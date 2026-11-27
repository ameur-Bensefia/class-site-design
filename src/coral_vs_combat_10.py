"""Anchor-referenced CORAL versus ComBat on frozen ImageNet ResNet-18 features, constructed confounding, on the same ten
replicates as the frozen-versus-fine-tuned comparison (same splits as e2e_cxr.py / frozen_vs_e2e.py): paired difference, corrected resampled t-test
(test fraction 0.3) and Holm adjustment across bridge sizes.  Usage: python coral_vs_combat_10.py   (about 15 minutes)"""
import os
import sys

import numpy as np
from scipy.stats import t as tdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_p1 import group_split, load_cxr  # noqa: E402
from p2_experiments import heads  # noqa: E402
from frozen_vs_e2e import holm  # noqa: E402

_, Y, S, G = load_cxr()
X0 = np.load("cxr_r18_frozen.npy")
res = {}
for rep in range(10):
    for b in (100, 1000, -1):
        rng = np.random.default_rng(200 + rep)
        tr, te = group_split(G, rng)
        keep = tr & (((Y == 1) & (S == "NIH")) | ((Y == 0) & (S == "CheXpert")))
        nf = np.where(tr & (Y == 0) & (S == "NIH"))[0]
        keep[rng.choice(nf, len(nf) if b < 0 else min(b, len(nf)), replace=False)] = True
        test = te & (S == "CheXpert")
        X = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
        sc = heads(X[keep], Y[keep], S[keep], X[test], S[test])
        for k in ("ComBat", "Anchor-referenced CORAL"):
            res.setdefault((k, b), []).append(roc_auc_score(Y[test], sc[k]))
    print(f"  replicate {rep + 1}/10 done", flush=True)
rows = []
for b in (100, 1000, -1):
    c, a = 100 * np.array(res[("ComBat", b)]), 100 * np.array(res[("Anchor-referenced CORAL", b)])
    d = a - c
    se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
    h = tdist.ppf(0.975, len(d) - 1) * se
    rows.append((b, c.mean(), a.mean(), d.mean(), h, 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1), int((d > 0).sum())))
adj = holm([r[5] for r in rows])
print("\nFrozen ResNet-18, 10 paired replicates: anchor-referenced CORAL minus ComBat")
print(f"{'bridge':>7s} {'ComBat':>7s} {'CORAL':>7s} {'difference [95% CI]':>24s} {'p':>8s} {'Holm p':>8s} {'wins':>6s}")
for (b, c, a, dm, h, p, w), pa in zip(rows, adj):
    print(f"{'all' if b < 0 else b:>7} {c:7.1f} {a:7.1f} {dm:+8.1f} [{dm - h:+5.1f}, {dm + h:+5.1f}] {p:8.2g} {pa:8.2g} {w:>3d}/10")
