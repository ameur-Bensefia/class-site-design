"""FOND baseline (Kaai et al., IEEE TPAMI 2025) on the HAM10000 and chest-radiograph semi-real experiments.

The FOND objective (masks, weighted supervised-contrastive loss with alpha/beta, fairness loss) is copied from the
authors' code (src/networks/fond.py, FONDBase.update). The only change is the featurizer: the frozen 128-dimensional
representations of our experiments replace images, so a small MLP replaces ResNet-18. Splits, seeds, PCA and
confounded training sets are taken from semireal_ham.py / semireal_cxr.py, so results are paired with the other
methods. Hyperparameters: 5 draws from the authors' search ranges (first draw = defaults), learning rate range shifted
for frozen features; selection by balanced accuracy on a 20% validation split of the training sites.

Usage:
  python fond_baseline.py selftest            # quick check that training works (about 1 minute)
  python fond_baseline.py ham  [replicates]   # HAM10000
  python fond_baseline.py cxr  [replicates]   # NIH / CheXpert
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

MODE = sys.argv[1] if len(sys.argv) > 1 else "selftest"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
STEPS, BATCH = 1500, 32
torch.set_num_threads(max(1, torch.get_num_threads()))


# ============================================================ FOND objective (authors' code, unchanged logic)
class FONDHead(nn.Module):
    def __init__(self, d_in, num_classes, hp, shared_classes):
        super().__init__()
        enc = 256
        self.featurizer = nn.Sequential(nn.Linear(d_in, enc), nn.ReLU(), nn.Linear(enc, enc))   # replaces ResNet-18
        self.classifier = nn.Linear(enc, num_classes)
        self.projector = nn.Sequential(nn.Linear(enc, enc), nn.ReLU(), nn.Linear(enc, 256))
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.hp = hp
        oc = torch.zeros(num_classes, dtype=torch.bool)
        oc[shared_classes] = True
        self.oc_weight, self.noc_weight = oc.float(), (~oc).float()
        self.opt = torch.optim.Adam(self.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    @staticmethod
    def get_masks(Y, D):
        self_mask = ~torch.eye(Y.shape[0], dtype=torch.bool)
        same_Y = torch.eq(Y.view(-1, 1), Y.view(-1, 1).T)
        same_D = torch.eq(D.view(-1, 1), D.view(-1, 1).T)
        return {"self_mask": self_mask, "same_class_mask": same_Y,
                "diff_domain_same_class_mask": same_Y * ~same_D,
                "same_domain_diff_class_mask": ~same_Y * same_D}

    def supcon_loss(self, projections, positive_mask, negative_mask, alpha, beta, epsilon=1e-6):
        proj_dot = torch.div(projections @ projections.T, self.hp["temperature"])
        logits_max, _ = torch.max(proj_dot, dim=1, keepdim=True)
        logits = proj_dot - logits_max.detach()
        exp_logits = torch.exp(logits) * negative_mask * beta
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (positive_mask * alpha * log_prob).sum(1) / (positive_mask.sum(1) + epsilon)
        return (-(self.hp["temperature"] / self.hp["base_temperature"]) * mean_log_prob_pos).mean()

    def update(self, X, Y, D):
        feats = self.featurizer(X)
        projections = F.normalize(self.projector(feats))
        classifs = self.classifier(feats)
        masks = self.get_masks(Y, D)
        alpha = (~masks["diff_domain_same_class_mask"]) + masks["diff_domain_same_class_mask"] * self.hp["xda_alpha"]
        beta = (~masks["same_domain_diff_class_mask"]) + masks["same_domain_diff_class_mask"] * self.hp["xda_beta"]
        xdom_loss = self.supcon_loss(projections, masks["same_class_mask"] * masks["self_mask"], masks["self_mask"],
                                     alpha, beta)
        oc_loss = F.cross_entropy(classifs, Y, weight=self.oc_weight)
        noc_loss = F.cross_entropy(classifs, Y, weight=self.noc_weight)
        error_loss = oc_loss - noc_loss
        if torch.isnan(error_loss):
            error_loss = torch.tensor(0.0)
        loss = F.cross_entropy(classifs, Y) + self.hp["xdom_lmbd"] * xdom_loss + self.hp["error_lmbd"] * torch.abs(error_loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    @torch.no_grad()
    def score(self, X):
        z = self.classifier(self.featurizer(torch.as_tensor(X, dtype=torch.float32)))
        return (z[:, 1] - z[:, 0]).numpy()


# ============================================================ hyperparameters (authors' ranges; lr shifted)
def hparams(draw, rng):
    if draw == 0:
        return dict(temperature=0.07, base_temperature=0.07, xdom_lmbd=1.0, error_lmbd=1.0, xda_alpha=1.0,
                    xda_beta=1.0, lr=1e-3, weight_decay=0.0)
    return dict(temperature=0.07 * rng.uniform(0.75, 1.25), base_temperature=0.07,
                xdom_lmbd=10 ** rng.uniform(-1, 3), error_lmbd=10 ** rng.uniform(-1, 3),
                xda_alpha=10 ** rng.uniform(0, 1), xda_beta=10 ** rng.uniform(0, 1),
                lr=10 ** rng.uniform(-4, -2.5), weight_decay=10 ** rng.uniform(-6, -2))


def train(X, y, d, hp, shared_classes, seed, erm=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hp = dict(hp, xdom_lmbd=0.0, error_lmbd=0.0) if erm else hp
    model = FONDHead(X.shape[1], 2, hp, shared_classes)
    doms = [np.where(d == k)[0] for k in np.unique(d)]
    Xt, yt, dt = (torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long),
                  torch.as_tensor(d, dtype=torch.long))
    losses = []
    for _ in range(STEPS):                                   # equal number of samples per domain (DomainBed)
        idx = np.concatenate([rng.choice(ix, BATCH, replace=len(ix) < BATCH) for ix in doms])
        losses.append(model.update(Xt[idx], yt[idx], dt[idx]))
    return model, losses


def balanced_acc(y, s):
    p = (s > 0).astype(int)
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def fond_and_erm(Xtr, ytr, Str, Xte, yte, seed):
    """Select FOND hyperparameters on a 20% validation split of the training sites; return test AUCs."""
    sites = {s: i for i, s in enumerate(sorted(set(Str)))}
    d = np.array([sites[s] for s in Str])
    shared = [c for c in (0, 1) if len(set(Str[ytr == c])) > 1]
    rng = np.random.default_rng(seed)
    val = rng.random(len(ytr)) < 0.2
    best = (-1, None)
    for draw in range(5):
        hp = hparams(draw, rng)
        m, _ = train(Xtr[~val], ytr[~val], d[~val], hp, shared, seed + draw)
        v = balanced_acc(ytr[val], m.score(Xtr[val]))
        if v > best[0]:
            best = (v, hp)
    fond, _ = train(Xtr, ytr, d, best[1], shared, seed + 99)
    erm, _ = train(Xtr, ytr, d, hparams(0, rng), shared, seed + 99, erm=True)
    return dict(FOND=float(roc_auc_score(yte, fond.score(Xte))),
                MLP_ERM=float(roc_auc_score(yte, erm.score(Xte))), FOND_hp=best[1])


# ============================================================ same splits as the semi-real scripts
def split_ham(rep, n_bridge):
    import semireal_ham as H
    rng = np.random.default_rng(100 + rep)
    tr, te = H.lesion_split(rng)
    keep = np.zeros(len(H.Y), bool)
    keep |= tr & (H.Y == 1) & (H.S == H.LINKED)
    keep |= tr & (H.Y == 0) & (H.S != H.LINKED)
    nev = np.where(tr & (H.Y == 0) & (H.S == H.LINKED))[0]
    nb = len(nev) if n_bridge < 0 else min(n_bridge, len(nev))
    keep[rng.choice(nev, nb, replace=False)] = True
    test = te & (H.S == H.TEST)
    from sklearn.decomposition import PCA
    X = PCA(128, random_state=rep).fit(H.X0[tr]).transform(H.X0)
    return X[keep], H.Y[keep], H.S[keep], X[test], H.Y[test], nb


def split_cxr(rep, n_bridge):
    import semireal_cxr as C
    rng = np.random.default_rng(200 + rep)
    tr, te = C.patient_split(rng)
    keep = tr & (((C.Y == 1) & (C.S == C.LINKED)) | ((C.Y == 0) & (C.S == C.TEST)))
    nf = np.where(tr & (C.Y == 0) & (C.S == C.LINKED))[0]
    nb = len(nf) if n_bridge < 0 else min(n_bridge, len(nf))
    keep[rng.choice(nf, nb, replace=False)] = True
    test = te & (C.S == C.TEST)
    from sklearn.decomposition import PCA
    X = PCA(128, random_state=rep).fit(C.X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(C.X0)
    return X[keep], C.Y[keep], C.S[keep], X[test], C.Y[test], nb


def selftest():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    d = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 16)) + 1.5 * y[:, None] * (np.arange(16) < 4)
    S = np.array(["a", "b"])[d]
    m, losses = train(X, y, d, hparams(0, rng), [0, 1], 0)
    auc = roc_auc_score(y, m.score(X))
    ok = np.mean(losses[-100:]) < np.mean(losses[:100]) and auc > 0.8
    print(f"loss first 100 steps {np.mean(losses[:100]):.3f} -> last 100 steps {np.mean(losses[-100:]):.3f}; "
          f"training AUC {auc:.3f}")
    r = fond_and_erm(X, y, S, X, y, 1)
    print(f"selection + final fits ran: FOND AUC {r['FOND']:.3f}, MLP ERM AUC {r['MLP_ERM']:.3f}")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")


if __name__ == "__main__":
    if MODE == "selftest":
        selftest()
        sys.exit()
    sys.argv = [sys.argv[0]]                      # the semi-real modules read their default paths
    split = split_ham if MODE == "ham" else split_cxr
    bridges = (0, 10, 25, 50, 100, 300, -1) if MODE == "ham" else (0, 25, 50, 100, 300, 1000, -1)
    out = []
    for nb in bridges:
        for r in range(REPS):
            Xtr, ytr, Str, Xte, yte, n_used = split(r, nb)
            res = fond_and_erm(Xtr, ytr, Str, Xte, yte, 1000 * r + 7)
            res["n_bridge"] = int(n_used)
            out.append(res)
        print("bridge", nb, "done", flush=True)
    json.dump(out, open(f"fond_{MODE}_results.json", "w"), indent=1)
    prev = json.load(open(f"semireal_{MODE}_results.json"))
    key = "Anchor transform (nevi, Ledoit-Wolf)" if MODE == "ham" else "Anchor transform (no finding, LW)"
    get = (lambda o, k: o[k]["auc"]) if MODE == "ham" else (lambda o, k: o[k])
    print(f"\nAUC %, {REPS} replicates (paired with semireal_{MODE}_results.json)")
    print(f"{'bridge':>7s} {'MLP ERM':>9s} {'FOND':>9s} {'Proposed':>9s} {'Proposed - FOND [95% CI]':>28s} {'wins':>6s}")
    for k, nb in enumerate(bridges):
        g, p = out[k * REPS:(k + 1) * REPS], prev[k * REPS:(k + 1) * REPS]
        fond = 100 * np.array([o["FOND"] for o in g])
        erm = 100 * np.array([o["MLP_ERM"] for o in g])
        line = f"{g[0]['n_bridge']:>7d} {erm.mean():9.1f} {fond.mean():9.1f} "
        if key in p[0]:
            prop = 100 * np.array([get(o, key) for o in p])
            dlt = prop - fond
            h = 2.262 * dlt.std(ddof=1) / np.sqrt(len(dlt))
            line += f"{prop.mean():9.1f} {dlt.mean():+12.1f} [{dlt.mean() - h:+.1f}, {dlt.mean() + h:+.1f}] {int(np.sum(dlt > 0)):>4d}/{len(dlt)}"
        else:
            line += f"{'n/a':>9s} {'(no bridge)':>28s}"
        print(line)
