"""End-to-end experiments on NIH and CheXpert: an ImageNet-pretrained ResNet-18 fine-tuned on 224x224 radiographs.

E1 (--exp constructed): effusion from NIH only in training, no-finding from CheXpert plus a bridge of NIH no-finding
   images; test on held-out CheXpert patients (same patient-wise splits and seeds as semireal_cxr.py).
E2 (--exp natural): the natural pool (both classes, both hospitals); pooled vs within-hospital test AUC.
Methods trained end to end: ERM, DANN (gradient-reversal site adversary) and, for E1, FOND (authors' loss, default
hyperparameters). Post-hoc on the learned ERM features: logistic head without harmonization, ComBat (diagnosis
covariate), anchor centring and anchor-referenced CORAL (anchors: no-finding images of each hospital).
Results are appended to results_e2e_es_<exp>.jsonl (early stopping); finished configurations are skipped, so runs can be resumed.

Usage:
  python e2e_cxr.py --exp selftest                  # 2-3 minutes: tiny subsets, checks the whole pipeline
  python e2e_cxr.py --exp constructed               # E1 (several hours)
  python e2e_cxr.py --exp natural                   # E2
  python e2e_cxr.py --summary                       # print the tables from the result files
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="selftest", choices=["selftest", "constructed", "natural"])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--max_epochs", type=int, default=6)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--bridges", default="0,100,1000,-1")
ap.add_argument("--cache", default="cxr224.npy")
ap.add_argument("--summary", action="store_true")
args = ap.parse_args()

if args.summary:
    from scipy.stats import t as tdist
    for exp in ("constructed", "natural"):
        fn = f"results_e2e_es_{exp}.jsonl"
        if not os.path.exists(fn):
            continue
        rows = [json.loads(l) for l in open(fn)]
        print(f"\n=== {exp}: {len(rows)} results")
        keys = sorted({(r["bridge"]) for r in rows}, key=lambda b: (b < 0, b))
        methods = list(dict.fromkeys(r["method"] for r in rows))
        for b in keys:
            print(f"bridge {b}:")
            ref = {r["rep"]: r for r in rows if r["bridge"] == b and r["method"] == "ComBat on learned features"}
            for m in methods:
                g = [r for r in rows if r["bridge"] == b and r["method"] == m]
                if not g:
                    continue
                if exp == "constructed":
                    v = 100 * np.array([r["auc"] for r in g])
                    line = f"  {m:42s} AUC {v.mean():5.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:4.1f} (n={len(v)})"
                    if ref and m != "ComBat on learned features":
                        d = np.array([100 * (r["auc"] - ref[r["rep"]]["auc"]) for r in g if r["rep"] in ref])
                        if len(d) > 1:
                            se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
                            p = 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1)
                            line += f" | minus ComBat {d.mean():+5.1f} (corrected p={p:.2g})"
                else:
                    pooled = 100 * np.mean([r["pooled"] for r in g])
                    within = 100 * np.mean([r["mean_within"] for r in g])
                    line = f"  {m:42s} pooled {pooled:5.1f} | mean within {within:5.1f} | gap {pooled - within:+5.1f} (n={len(g)})"
                print(line)
    sys.exit()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.argv = [sys.argv[0]]                                  # semireal_cxr reads its default data paths
import semireal_cxr as C                                  # noqa: E402  (same Y, S, G arrays and patient_split)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMGS = np.load(args.cache, mmap_mode="r")
assert len(IMGS) == len(C.Y), f"cache has {len(IMGS)} images but labels have {len(C.Y)}"
SITES = ("NIH", "CheXpert")
SITE_ID = np.array([SITES.index(s) for s in C.S])
MEAN, STD = 0.449, 0.226                                  # ImageNet mean/std averaged over channels (grayscale)
torch.manual_seed(0)


def batch_tensor(idx):
    x = torch.from_numpy(np.ascontiguousarray(IMGS[idx])).float().div_(255.0)
    x = (x - MEAN) / STD
    return x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(DEV)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.f.fc = nn.Identity()
        self.head = nn.Linear(512, 2)
        self.site = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        h = self.f(x)
        return h, self.head(h)


def fond_loss(h, logits, y, d, net, shared, temperature=0.07, alpha=1.0, beta=1.0, xdom=1.0, err=1.0):
    """FOND objective (authors' formulation): weighted supervised-contrastive loss + fairness term."""
    z = F.normalize(net.proj(h))
    n = len(y)
    self_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    same_y = y.view(-1, 1) == y.view(1, -1)
    same_d = d.view(-1, 1) == d.view(1, -1)
    a = (~(same_y & ~same_d)).float() + (same_y & ~same_d).float() * alpha
    b = (~(~same_y & same_d)).float() + (~same_y & same_d).float() * beta
    logit = z @ z.T / temperature
    logit = logit - logit.max(1, keepdim=True)[0].detach()
    exp = torch.exp(logit) * self_mask * b
    logp = logit - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (same_y & self_mask).float()
    supcon = (-(pos * a * logp).sum(1) / (pos.sum(1) + 1e-6)).mean()
    fair = torch.tensor(0.0, device=z.device)
    if shared and len(shared) < 2:                         # authors' code: the term is zero when it is undefined
        w_oc = torch.zeros(2, device=z.device)
        w_oc[shared] = 1.0
        if w_oc[y].sum() > 0 and (1 - w_oc)[y].sum() > 0:
            fair = torch.abs(F.cross_entropy(logits, y, weight=w_oc) - F.cross_entropy(logits, y, weight=1 - w_oc))
    return xdom * supcon + err * fair


def split_validation(idx, seed, frac=0.1):
    """Hold out 10% of the training patients (seeded) for early stopping; both classes must remain in the fit set."""
    pats = np.unique(C.G[idx])
    val_p = set(np.random.default_rng(seed + 999).choice(pats, max(1, int(frac * len(pats))), replace=False))
    vmask = np.array([g in val_p for g in C.G[idx]])
    return idx[~vmask], idx[vmask]


def train(idx, method, epochs, seed, shared=None):
    """Early stopping: class-weighted cross-entropy on 10% held-out training patients after each epoch; the weights
    of the best epoch are restored; stop after PATIENCE epochs without improvement or MAX_EPOCHS epochs."""
    import copy as _copy
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fit_idx, val_idx = split_validation(idx, seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    y_all = torch.from_numpy(C.Y).long()
    d_all = torch.from_numpy(SITE_ID).long()
    cw = torch.tensor(len(fit_idx) / (2 * np.bincount(C.Y[fit_idx], minlength=2).clip(1)), dtype=torch.float32, device=DEV)
    per_epoch = max(1, len(fit_idx) // 64)
    total = args.max_epochs * per_epoch
    by_dom = [fit_idx[SITE_ID[fit_idx] == k] for k in range(2)]
    best, best_state, best_ep, bad, step = np.inf, None, 0, 0, 0
    for ep in range(1, args.max_epochs + 1):
        net.train()
        for _ in range(per_epoch):
            if method == "FOND" and all(len(v) > 0 for v in by_dom):
                bi = np.concatenate([rng.choice(v, 32, replace=len(v) < 32) for v in by_dom])
            else:
                bi = rng.choice(fit_idx, 64, replace=False)
            x = batch_tensor(bi)
            tb = torch.from_numpy(bi)
            y, d = y_all[tb].to(DEV), d_all[tb].to(DEV)
            h, logits = net(x)
            loss = F.cross_entropy(logits, y, weight=cw)
            if method == "DANN":
                lam = 2.0 / (1.0 + np.exp(-10 * step / total)) - 1.0
                loss = loss + F.cross_entropy(net.site(GradReverse.apply(h, lam)), d)
            elif method == "FOND":
                loss = loss + fond_loss(h, logits, y, d, net, shared)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        net.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for k in range(0, len(val_idx), 256):
                b = val_idx[k:k + 256]
                _, lg = net(batch_tensor(b))
                vl += float(F.cross_entropy(lg, torch.from_numpy(C.Y[b]).long().to(DEV), weight=cw, reduction="sum"))
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
def embed(net, idx):
    hs, ss = [], []
    for k in range(0, len(idx), 256):
        h, lg = net(batch_tensor(idx[k:k + 256]))
        hs.append(h.cpu().numpy())
        ss.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.vstack(hs), np.concatenate(ss)


def sqrtm(Cm, power):
    w, V = np.linalg.eigh(Cm)
    return V @ np.diag(np.clip(w, 1e-8, None) ** power) @ V.T


def posthoc(Htr, ytr, Str, Hte, Ste):
    """Logistic heads on learned features: none, ComBat, anchor centring, anchor-referenced CORAL."""
    from neuroCombat import neuroCombat
    lr = lambda: LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")  # noqa: E731
    out = {"Linear head on learned features": lr().fit(Htr, ytr).decision_function(Hte)}
    if len(np.unique(Str)) > 1 and all(((Str == s) & (ytr == 0)).sum() > 5 for s in np.unique(Str)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = neuroCombat(dat=Htr.T, covars=pd.DataFrame({"site": Str, "dx": ytr}), batch_col="site",
                              categorical_cols=["dx"])
        est = fit["estimates"]
        lev = list(np.asarray(est["batches"]).astype(str))
        var, mean = np.asarray(est["var.pooled"]).reshape(-1, 1), np.asarray(est["stand.mean"])[:, [0]]
        Hc = np.empty_like(Hte)
        for s in np.unique(Ste):
            bi = lev.index(str(s))
            zz = (Hte[Ste == s].T - mean) / np.sqrt(var)
            zz = (zz - np.asarray(est["gamma.star"])[bi][:, None]) / np.sqrt(np.asarray(est["delta.star"])[bi][:, None])
            Hc[Ste == s] = (zz * np.sqrt(var) + mean).T
        out["ComBat on learned features"] = lr().fit(fit["data"].T, ytr).decision_function(Hc)
        ref_mu = Htr[ytr == 0].mean(0)
        ref_C = LedoitWolf().fit(Htr[ytr == 0]).covariance_
        st = {s: (Htr[(Str == s) & (ytr == 0)].mean(0), LedoitWolf().fit(Htr[(Str == s) & (ytr == 0)]).covariance_)
              for s in np.unique(Str)}

        def cent(H, S_):
            return H - np.array([st[s][0] for s in S_]) + ref_mu

        def coral(H, S_):
            o = np.empty_like(H)
            for s in np.unique(S_):
                m = S_ == s
                o[m] = (H[m] - st[s][0]) @ sqrtm(st[s][1], -0.5) @ sqrtm(ref_C, 0.5) + ref_mu
            return o
        out["Anchor centring on learned features"] = lr().fit(cent(Htr, Str), ytr).decision_function(cent(Hte, Ste))
        out["Anchor-referenced CORAL on learned features"] = lr().fit(coral(Htr, Str), ytr).decision_function(coral(Hte, Ste))
    return out


def splits(rep, n_bridge, exp):
    rng = np.random.default_rng(200 + rep)                # identical to semireal_cxr.run
    tr, te = C.patient_split(rng)
    if exp == "natural":
        return np.where(tr)[0], np.where(te)[0], -1
    keep = tr & (((C.Y == 1) & (C.S == C.LINKED)) | ((C.Y == 0) & (C.S == C.TEST)))
    nf = np.where(tr & (C.Y == 0) & (C.S == C.LINKED))[0]
    nb = len(nf) if n_bridge < 0 else min(n_bridge, len(nf))
    keep[rng.choice(nf, nb, replace=False)] = True
    return np.where(keep)[0], np.where(te & (C.S == C.TEST))[0], nb


def evaluate(scores, te_idx, exp):
    y = C.Y[te_idx]
    if exp == "natural":
        s_ = C.S[te_idx]
        within = [roc_auc_score(y[s_ == k], scores[s_ == k]) for k in SITES]
        return dict(pooled=float(roc_auc_score(y, scores)), within_NIH=float(within[0]),
                    within_CheXpert=float(within[1]), mean_within=float(np.mean(within)))
    return dict(auc=float(roc_auc_score(y, scores)))


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
    bridges = [int(b) for b in args.bridges.split(",")] if exp == "constructed" else [-1]
    reps, epochs = (1, 1) if args.exp == "selftest" else (args.reps, args.epochs)
    if args.exp == "selftest":
        bridges = [100]
    print(f"device {DEV} | experiment {args.exp} | bridges {bridges} | reps {reps} | epochs {epochs}", flush=True)
    for b in bridges:
        for rep in range(reps):
            tr_idx, te_idx, nb = splits(rep, b, exp)
            if args.exp == "selftest":
                rs = np.random.default_rng(0)
                tr_idx = rs.choice(tr_idx, min(768, len(tr_idx)), replace=False)
                te_idx = rs.choice(te_idx, min(512, len(te_idx)), replace=False)
            shared = [c for c in (0, 1) if len(set(C.S[tr_idx][C.Y[tr_idx] == c])) > 1]
            methods = ["ERM", "DANN"] + (["FOND"] if exp == "constructed" else [])
            for m in methods:
                if (b, rep, m) in done and m != "ERM":
                    continue
                need_posthoc = m == "ERM" and not all((b, rep, k) in done for k in ("ERM", "ComBat on learned features"))
                if (b, rep, m) in done and not need_posthoc:
                    continue
                t0 = time.time()
                net = train(tr_idx, m, epochs, seed=1000 * rep + 7, shared=shared)
                Htr, _ = embed(net, tr_idx)
                Hte, s_te = embed(net, te_idx)
                rows = {m: s_te}
                if m == "ERM":
                    rows.update(posthoc(Htr, C.Y[tr_idx], C.S[tr_idx], Hte, C.S[te_idx]))
                os.makedirs("scores_e2e_es", exist_ok=True)
                for name, sc in rows.items():                 # per-sample test scores, for the AUC decomposition
                    slug = "".join(ch if ch.isalnum() else "_" for ch in name)
                    np.save(f"scores_e2e_es/{exp}_b{b}_r{rep}_{slug}.npy", np.c_[te_idx, sc])
                with open(out_fn, "a") as f:
                    for name, sc in rows.items():
                        if (b, rep, name) in done:
                            continue
                        r = dict(exp=exp, bridge_req=b, bridge=int(nb), rep=rep, method=name, n_train=int(len(tr_idx)), best_epoch=int(net.best_epoch),
                                 **evaluate(sc, te_idx, exp))
                        f.write(json.dumps(r) + "\n")
                        done.add((b, rep, name))
                        print(f"  bridge {nb:>6} rep {rep} {name:42s} " +
                              " ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
                print(f"  ({m} trained in {(time.time() - t0) / 60:.1f} min)", flush=True)
    if args.exp == "selftest":
        print("SELFTEST FINISHED: the pipeline ran end to end (numbers on tiny subsets are not meaningful).")
