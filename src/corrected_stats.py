"""Corrected resampled t-tests (Nadeau and Bengio, 2003) for all paired comparisons of the semi-real experiments.
Replicates reuse overlapping data, so the variance of the mean difference is inflated by (1/J + n_test/n_train).
Reads semireal_ham_results.json, semireal_cxr_results.json and, if present, fond_ham_results.json / fond_cxr_results.json.
Usage: python corrected_stats.py
"""
import json
import os

import numpy as np
from scipy.stats import t as tdist

FRAC_TEST = 0.3
REPS = 10


def corrected(d):
    J = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / J + FRAC_TEST / (1 - FRAC_TEST)))
    h = tdist.ppf(0.975, J - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, 2 * tdist.sf(abs(d.mean()) / se, J - 1)


for ds, key in (("ham", "Anchor transform (nevi, Ledoit-Wolf)"), ("cxr", "Anchor transform (no finding, LW)")):
    main = json.load(open(f"semireal_{ds}_results.json"))
    fond = json.load(open(f"fond_{ds}_results.json")) if os.path.exists(f"fond_{ds}_results.json") else None
    get = (lambda o, k: o[k]["auc"]) if ds == "ham" else (lambda o, k: o[k])
    print(f"\n=== {ds.upper()}: proposed minus reference, AUC points, corrected 95% CI and p-value")
    n_blocks = len(main) // REPS
    for b in range(n_blocks):
        g = main[b * REPS:(b + 1) * REPS]
        if key not in g[0]:
            continue
        prop = 100 * np.array([get(o, key) for o in g])
        refs = {"ERM": 100 * np.array([get(o, "ERM") for o in g])}
        if "ComBat (dx covariate)" in g[0]:
            refs["ComBat"] = 100 * np.array([get(o, "ComBat (dx covariate)") for o in g])
        if fond is not None:
            refs["FOND"] = 100 * np.array([o["FOND"] for o in fond[b * REPS:(b + 1) * REPS]])
        cells = []
        for name, r in refs.items():
            m, lo, hi, p = corrected(prop - r)
            cells.append(f"{name} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] p={p:.2g}")
        print(f"  bridge {g[0]['n_bridge']:>6d}: " + " | ".join(cells))
