"""Anchor-Referenced Batch Normalization (AR-BN) for multi-site training under class-site confounding.

Every BatchNorm layer of an ImageNet-pretrained ResNet-18 is replaced by a site-aware layer that normalizes all
images of a site with statistics computed ONLY on that site's anchor (shared-class) images, then applies a shared
affine transform. The site correction is therefore estimated where the class-site graph identifies it (anchors),
propagated to every class of the site, and applied at every depth during training. Combined with cell-balanced
batches (equal numbers per class-site cell), which train on the balanced design and remove prevalence cues.

Methods written to the same result files as e2e_cxr.py (same splits and seeds, so comparisons are paired):
  "AR-BN + balanced cells (proposed)", "AR-BN", "Domain-specific BN (all samples)", "Balanced cells, standard BN".
AR-BN needs anchors at every training site: with no bridge it is not estimable and is recorded as such.

Usage (in the folder of e2e_cxr.py, semireal_cxr.py and cxr224.npy):
  python e2e_arbn.py --exp selftest
  python e2e_arbn.py --exp constructed        # about 6 h
  python e2e_arbn.py --exp natural            # about 4 h
  python e2e_arbn.py --summary                # paired comparison with ERM and ComBat on learned features
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="selftest", choices=["selftest", "constructed", "natural"])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--max_epochs", type=int, default=6)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--bridges", default="0,100,1000,-1")
ap.add_argument("--cache", default="cxr224.npy")
ap.add_argument("--methods", default="all")
ap.add_argument("--summary", action="store_true")
args = ap.parse_args()
PROPOSED = "AR-BN + balanced cells (proposed)"
METHODS = [PROPOSED, "AR-BN", "Domain-specific BN (all samples)", "Balanced cells, standard BN"]

if args.summary:
    from scipy.stats import t as tdist
    for exp in ("constructed", "natural"):
        fn = f"results_e2e_es_{exp}.jsonl"
        if not os.path.exists(fn):
            continue
        rows = [json.loads(l) for l in open(fn)]
        print(f"\n=== {exp} (paired with the earlier ERM/ComBat runs; corrected resampled t-test, test fraction 0.3)")
        for b in sorted({r["bridge_req"] for r in rows}, key=lambda v: (v < 0, v)):
            g = [r for r in rows if r["bridge_req"] == b]
            meths = list(dict.fromkeys(r["method"] for r in g))
            key = "auc" if exp == "constructed" else "mean_within"
            ref = {r["rep"]: r for r in g if r["method"] == "ComBat on learned features"}
            erm = {r["rep"]: r for r in g if r["method"] == "ERM"}
            print(f"bridge {'all' if b < 0 else b}:")
            for m in meths:
                gm = [r for r in g if r["method"] == m]
                if "not_estimable" in gm[0]:
                    print(f"  {m:42s} not estimable (no anchors at a training site)")
                    continue
                v = 100 * np.array([r[key] for r in gm])
                line = f"  {m:42s} {key} {v.mean():5.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:4.1f} (n={len(v)})"
                if exp == "natural":
                    gap = 100 * np.mean([r["pooled"] - r["mean_within"] for r in gm])
                    line += f" | gap {gap:+5.1f}"
                for name, refd in (("ComBat", ref), ("ERM", erm)):
                    if refd and m not in ("ComBat on learned features", "ERM"):
                        d = np.array([100 * (r[key] - refd[r["rep"]][key]) for r in gm if r["rep"] in refd])
                        if len(d) > 1 and d.std() > 0:
                            se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
                            p = 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1)
                            line += f" | vs {name} {d.mean():+5.1f} (p={p:.2g}, wins {int((d > 0).sum())}/{len(d)})"
                print(line)
    sys.exit()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.argv = [sys.argv[0]]
import semireal_cxr as C  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMGS = np.load(args.cache, mmap_mode="r")
assert len(IMGS) == len(C.Y)
SITES = ("NIH", "CheXpert")
SITE_ID = np.array([SITES.index(s) for s in C.S])
MEAN, STD = 0.449, 0.226


class Ctx:
    """Batch layout read by every site-aware BN layer. Training batches hold exactly BLOCK images per site, sites in
    order 0..S-1, anchors in the first n_anchor rows of each block; statistics are computed for all sites in one
    vectorized operation (no loops, fixed shapes, which keeps Apple MPS fast and stable). Evaluation: running
    statistics gathered per sample from `site_vec`."""
    n_anchor = 16
    site_vec = None


BLOCK = 32


class SiteBN(nn.Module):
    def __init__(self, bn, n_sites, momentum=0.1, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer("rm", bn.running_mean.detach().clone().repeat(n_sites, 1))
        self.register_buffer("rv", bn.running_var.detach().clone().repeat(n_sites, 1))
        self.n_sites, self.momentum, self.eps = n_sites, momentum, eps

    def forward(self, x):
        n, c, h, w = x.shape
        if self.training:
            x5 = x.reshape(self.n_sites, BLOCK, c, h, w)
            xa = x5[:, :Ctx.n_anchor]
            mu = xa.mean(dim=(1, 3, 4))
            var = xa.var(dim=(1, 3, 4), unbiased=False)
            with torch.no_grad():
                self.rm.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
                self.rv.mul_(1 - self.momentum).add_(self.momentum * var.detach())
            out = ((x5 - mu[:, None, :, None, None]) / torch.sqrt(var[:, None, :, None, None] + self.eps)).reshape(n, c, h, w)
        else:
            mu, var = self.rm[Ctx.site_vec], self.rv[Ctx.site_vec]
            out = (x - mu[:, :, None, None]) / torch.sqrt(var[:, :, None, None] + self.eps)
        return out * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def replace_bn(module, n_sites):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, SiteBN(child, n_sites))
        else:
            replace_bn(child, n_sites)


def make_net(site_bn):
    f = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    f.fc = nn.Identity()
    if site_bn:
        replace_bn(f, len(SITES))
    net = nn.ModuleDict({"f": f, "head": nn.Linear(512, 2)})
    return net.to(DEV)


def batch_tensor(idx):
    x = torch.from_numpy(np.ascontiguousarray(IMGS[idx])).float().div_(255.0)
    return ((x - MEAN) / STD).unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(DEV)


def sample_batch(rng, idx_by_site_cls, mode):
    """Exactly BLOCK images per site, sites in order 0..S-1, 16 anchors (class 0) first when the site has any.
    balanced: the other 16 rows come from the site's other class when present, otherwise from its anchors; per-sample
    loss weights then make every class-site cell contribute equally. site: the other 16 rows are drawn from the whole
    site (prevalence preserved)."""
    order = []
    for site in sorted(idx_by_site_cls):
        cls = idx_by_site_cls[site]
        anchors = cls.get(0)
        allv = np.concatenate(list(cls.values()))
        a = rng.choice(anchors, 16, replace=len(anchors) < 16) if anchors is not None else rng.choice(allv, 16, replace=len(allv) < 16)
        if mode == "balanced":
            other = cls.get(1)
            rest = rng.choice(other, 16, replace=len(other) < 16) if other is not None else rng.choice(allv, 16, replace=len(allv) < 16)
        else:
            rest = rng.choice(allv, 16, replace=len(allv) < 16)
        order.append(np.concatenate([a, rest]))
    bi = np.concatenate(order).astype(int)
    return bi


def split_validation(idx, seed, frac=0.1):
    """Hold out 10% of the training patients (seeded) for early stopping; both classes must remain in the fit set."""
    pats = np.unique(C.G[idx])
    val_p = set(np.random.default_rng(seed + 999).choice(pats, max(1, int(frac * len(pats))), replace=False))
    vmask = np.array([g in val_p for g in C.G[idx]])
    return idx[~vmask], idx[vmask]

def train(idx, method, epochs, seed):
    """Early stopping as in e2e_cxr.py: cross-entropy on 10% held-out training patients, best epoch restored."""
    import copy as _copy
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx, val_idx = split_validation(idx, seed)
    site_bn = method != "Balanced cells, standard BN"
    net = make_net(site_bn)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    idx_by_site_cls = {}
    for site in np.unique(SITE_ID[idx]):
        m = SITE_ID[idx] == site
        idx_by_site_cls[site] = {c: idx[m & (C.Y[idx] == c)] for c in (0, 1) if (m & (C.Y[idx] == c)).any()}
    mode = "balanced" if method in (PROPOSED, "Balanced cells, standard BN") else "site"
    Ctx.n_anchor = BLOCK if method == "Domain-specific BN (all samples)" else 16
    assert sorted(idx_by_site_cls) == list(range(len(SITES))), "every site must contribute training images"
    cw = None
    if mode == "site":
        cnt = np.bincount(C.Y[idx], minlength=2).clip(1)
        cw = torch.tensor(len(idx) / (2 * cnt), dtype=torch.float32, device=DEV)
    per_epoch = max(1, len(idx) // 64)
    cw_val = torch.tensor(len(idx) / (2 * np.bincount(C.Y[idx], minlength=2).clip(1)), dtype=torch.float32, device=DEV)
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(1, args.max_epochs + 1):
      net.train()
      for _ in range(per_epoch):
        bi = sample_batch(rng, idx_by_site_cls, mode)
        x = batch_tensor(bi)
        y = torch.from_numpy(C.Y[bi]).long().to(DEV)
        logits = net["head"](net["f"](x))
        if mode == "balanced":
            keys = C.Y[bi] * 10 + SITE_ID[bi]
            _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            wv = 1.0 / cnt[inv]
            wv = torch.tensor(wv / wv.mean(), dtype=torch.float32, device=DEV)
            loss = (F.cross_entropy(logits, y, reduction="none") * wv).mean()
        else:
            loss = F.cross_entropy(logits, y, weight=cw)
        opt.zero_grad()
        loss.backward()
        opt.step()
      net.eval()
      with torch.no_grad():
        vl, n = 0.0, 0
        for k in range(0, len(val_idx), 256):
            b = val_idx[k:k + 256]
            Ctx.site_vec = torch.from_numpy(SITE_ID[b]).long().to(DEV)
            lg = net["head"](net["f"](batch_tensor(b)))
            vl += float(F.cross_entropy(lg, torch.from_numpy(C.Y[b]).long().to(DEV), weight=cw_val, reduction="sum"))
            n += len(b)
      vl /= max(n, 1)
      if vl < best - 1e-4:
        best, best_state, best_ep, bad = vl, _copy.deepcopy(net.state_dict()), ep, 0
      else:
        bad += 1
        if bad >= args.patience:
            break
    net.load_state_dict(best_state)
    net.eval()
    net.best_epoch = best_ep
    return net


@torch.no_grad()
def scores(net, idx):
    out = []
    for k in range(0, len(idx), 256):
        b = idx[k:k + 256]
        Ctx.site_vec = torch.from_numpy(SITE_ID[b]).long().to(DEV)
        lg = net["head"](net["f"](batch_tensor(b)))
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.concatenate(out)


def splits(rep, n_bridge, exp):
    rng = np.random.default_rng(200 + rep)                # identical to semireal_cxr.run and e2e_cxr.py
    tr, te = C.patient_split(rng)
    if exp == "natural":
        return np.where(tr)[0], np.where(te)[0], -1
    keep = tr & (((C.Y == 1) & (C.S == C.LINKED)) | ((C.Y == 0) & (C.S == C.TEST)))
    nf = np.where(tr & (C.Y == 0) & (C.S == C.LINKED))[0]
    nb = len(nf) if n_bridge < 0 else min(n_bridge, len(nf))
    keep[rng.choice(nf, nb, replace=False)] = True
    return np.where(keep)[0], np.where(te & (C.S == C.TEST))[0], nb


def evaluate(sc, te_idx, exp):
    y = C.Y[te_idx]
    if exp == "natural":
        s_ = C.S[te_idx]
        within = [roc_auc_score(y[s_ == k], sc[s_ == k]) for k in SITES]
        return dict(pooled=float(roc_auc_score(y, sc)), within_NIH=float(within[0]), within_CheXpert=float(within[1]),
                    mean_within=float(np.mean(within)))
    return dict(auc=float(roc_auc_score(y, sc)))


if __name__ == "__main__":
    if args.exp == "selftest":
        args.max_epochs = 2
    exp = "constructed" if args.exp == "selftest" else args.exp
    out_fn = f"results_e2e_es_{args.exp}.jsonl"
    done = set()
    if os.path.exists(out_fn):
        for line in open(out_fn):
            r = json.loads(line)
            done.add((r["bridge_req"], r["rep"], r["method"]))
    methods = METHODS if args.methods == "all" else [m for m in METHODS if m in args.methods.split(";")]
    bridges = [100] if args.exp == "selftest" else ([int(b) for b in args.bridges.split(",")] if exp == "constructed" else [-1])
    reps, epochs = (1, 1) if args.exp == "selftest" else (args.reps, args.epochs)
    print(f"device {DEV} | {args.exp} | bridges {bridges} | reps {reps} | epochs {epochs} | methods {methods}", flush=True)
    for b in bridges:
        for rep in range(reps):
            tr_idx, te_idx, nb = splits(rep, b, exp)
            if args.exp == "selftest":
                rs = np.random.default_rng(0)
                tr_idx = rs.choice(tr_idx, min(768, len(tr_idx)), replace=False)
                te_idx = rs.choice(te_idx, min(512, len(te_idx)), replace=False)
            anchors_everywhere = all(((SITE_ID[tr_idx] == s) & (C.Y[tr_idx] == 0)).any() for s in np.unique(SITE_ID[tr_idx]))
            for m in methods:
                if (b, rep, m) in done:
                    continue
                rec = dict(exp=exp, bridge_req=b, bridge=int(nb), rep=rep, method=m, n_train=int(len(tr_idx)))
                if m.startswith("AR-BN") and not anchors_everywhere:
                    rec["not_estimable"] = True
                else:
                    t0 = time.time()
                    net = train(tr_idx, m, epochs, seed=1000 * rep + 7)
                    sc = scores(net, te_idx)
                    os.makedirs("scores_e2e_es", exist_ok=True)       # per-sample test scores, for the AUC decomposition
                    slug = "".join(ch if ch.isalnum() else "_" for ch in m)
                    np.save(f"scores_e2e_es/{exp}_b{b}_r{rep}_{slug}.npy", np.c_[te_idx, sc])
                    rec.update(evaluate(sc, te_idx, exp))
                    rec["best_epoch"] = int(net.best_epoch)
                    rec["minutes"] = round((time.time() - t0) / 60, 1)
                with open(out_fn, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done.add((b, rep, m))
                print(f"  bridge {nb:>6} rep {rep} {m:36s} " +
                      ("not estimable" if rec.get("not_estimable") else
                       " ".join(f"{k}={v:.3f}" for k, v in rec.items() if isinstance(v, float) and k != "minutes")
                       + f"  ({rec['minutes']} min)"), flush=True)
    if args.exp == "selftest":
        print("SELFTEST FINISHED")
