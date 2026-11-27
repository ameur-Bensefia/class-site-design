"""AUC decomposition for the end-to-end natural-pool runs, from the per-sample scores saved in scores_e2e/
(within-site pairs, cross-site pairs, equal-prevalence pooled AUC), as in audit_p1.py auc.
Usage: python e2e_decompose.py"""
import glob
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_p1 import decompose, load_cxr  # noqa: E402

_, Y, S, _ = load_cxr()
files = sorted(glob.glob("scores_e2e_es/natural_b-1_r*_*.npy"))
if not files:
    sys.exit("no natural-pool score files found in scores_e2e_es/")
res = {}
for fn in files:
    m = re.match(r"scores_e2e_es/natural_b-1_r(\d+)_(.+)\.npy", fn)
    rep, method = int(m.group(1)), m.group(2).replace("_", " ").strip()
    arr = np.load(fn)
    idx, sc = arr[:, 0].astype(int), arr[:, 1]
    y, s = Y[idx], S[idx]
    wp, cp, _ = decompose(y, sc, s)
    within = np.mean([roc_auc_score(y[s == k], sc[s == k]) for k in np.unique(s)])
    rng, bal, target = np.random.default_rng(rep), [], y.mean()
    for _ in range(30):
        keep = []
        for k in np.unique(s):
            ip, ineg = np.where((s == k) & (y == 1))[0], np.where((s == k) & (y == 0))[0]
            n_p = min(len(ip), int(round(len(ineg) * target / (1 - target))))
            n_n = min(len(ineg), int(round(n_p * (1 - target) / target)))
            keep += list(rng.choice(ip, n_p, replace=False)) + list(rng.choice(ineg, n_n, replace=False))
        keep = np.array(keep)
        bal.append(roc_auc_score(y[keep], sc[keep]))
    res.setdefault(method, []).append((roc_auc_score(y, sc), within, wp, cp, np.mean(bal)))
print(f"{'method':44s} {'n':>3s} {'pooled':>7s} {'within mean':>12s} {'within pairs':>13s} {'cross-site pairs':>17s} {'equal-prevalence':>17s}")
for m, v in res.items():
    v = 100 * np.array(v)
    print(f"{m:44s} {len(v):3d} " + " ".join(f"{x:>{w}.1f}" for x, w in zip(v.mean(0), (7, 12, 13, 17, 17))))
