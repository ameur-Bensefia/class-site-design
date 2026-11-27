"""Allocation of additional samples to class-site cells (design planning).

Given current cell counts and a budget of B additional samples, choose how many to add to each class-site cell so as
to minimize the effective resistance between the two disease classes, i.e. the variance of the disease contrast
(Theorem 1). R_eff is convex in the conductances (Ghosh et al., 2008); the marginal value of one sample in cell e is
the squared potential drop across e for a unit current between the two classes. Allocation: greedy in 200 increments
using exact gains (near-optimal for this convex problem). Compared with: uniform over cells, proportional to current
counts, and everything to the smallest cell. Metric: effective sample size 1/R_eff.

Usage: python allocation.py [datasets_dir]      (or --demo to run on the HAM10000 and NIH/CheXpert counts only)
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import sys

import numpy as np

DEF = DATASETS_DIR
BUDGETS = (500, 2000, 10000)


def r_eff(counts):
    """counts: dict {(class, site): n}; classes 0 and 1. Effective resistance between the two class nodes."""
    sites = sorted({s for _, s in counts})
    idx = {("c", 0): 0, ("c", 1): 1, **{("s", s): 2 + i for i, s in enumerate(sites)}}
    L = np.zeros((len(idx), len(idx)))
    for (c, s), n in counts.items():
        if n <= 0:
            continue
        i, j = idx[("c", c)], idx[("s", s)]
        L[i, i] += n; L[j, j] += n; L[i, j] -= n; L[j, i] -= n
    e = np.zeros(len(idx)); e[0], e[1] = 1, -1
    if np.linalg.matrix_rank(L) < len(idx) - 1:
        # the two classes may still be connected even if some site is isolated; check via the pseudo-inverse residual
        Lp = np.linalg.pinv(L)
        if not np.allclose(L @ Lp @ e, e, atol=1e-8):
            return np.inf
        return float(e @ Lp @ e)
    return float(e @ np.linalg.pinv(L) @ e)


ALLOWED = (0, 1)


def candidate_cells(counts):
    """Feasible cells: every allowed class at every site already in the study (ALLOWED = (0,) when only shared-class,
    e.g. normal, images can be collected)."""
    sites = sorted({s for _, s in counts})
    return [(c, s) for s in sites for c in ALLOWED]


def greedy(counts, budget, steps=200):
    cur = dict(counts)
    cells = candidate_cells(counts)
    inc = max(1, budget // steps)
    spent = 0
    while spent < budget:
        step = min(inc, budget - spent)
        best, best_r = None, np.inf
        for e in cells:
            trial = dict(cur); trial[e] = trial.get(e, 0) + step
            r = r_eff(trial)
            if r < best_r:
                best, best_r = e, r
        cur[best] = cur.get(best, 0) + step
        spent += step
    return cur


def uniform(counts, budget):
    cells = candidate_cells(counts)
    cur = dict(counts)
    for k, e in enumerate(cells):
        cur[e] = cur.get(e, 0) + budget // len(cells) + (1 if k < budget % len(cells) else 0)
    return cur


def proportional(counts, budget):
    elig = {e: n for e, n in counts.items() if e[0] in ALLOWED}
    tot = sum(elig.values())
    cur = dict(counts)
    for e, n in elig.items():
        cur[e] = n + int(round(budget * n / tot))
    return cur


def smallest(counts, budget):
    cells = candidate_cells(counts)
    e = min(cells, key=lambda k: counts.get(k, 0))
    cur = dict(counts); cur[e] = cur.get(e, 0) + budget
    return cur


def describe(before, after, top=3):
    add = {e: after.get(e, 0) - before.get(e, 0) for e in after}
    add = sorted(((n, e) for e, n in add.items() if n > 0), reverse=True)[:top]
    return ", ".join(f"{n} x class {e[0]} at {e[1]}" for n, e in add)


def report(name, counts):
    r0 = r_eff(counts)
    print(f"\n=== {name}: current effective sample size 1/R_eff = {1 / r0:,.0f}"
          f" ({sum(counts.values()):,} samples in {len(counts)} cells)")
    print(f"{'budget':>7s} {'optimal':>9s} {'uniform':>9s} {'proportional':>13s} {'smallest cell':>14s}   optimal allocation (top cells)")
    for B in BUDGETS:
        res = {}
        for nm, f in (("optimal", greedy), ("uniform", uniform), ("proportional", proportional), ("smallest", smallest)):
            after = f(counts, B)
            res[nm] = (1 / r_eff(after), after)
        print(f"{B:7,d} {res['optimal'][0]:9,.0f} {res['uniform'][0]:9,.0f} {res['proportional'][0]:13,.0f} "
              f"{res['smallest'][0]:14,.0f}   {describe(counts, res['optimal'][1])}")


def counts_from_npz(path, min_cell=0):
    z = np.load(path, allow_pickle=True)
    y, s = z["y"].astype(int), z["site"].astype(str)
    return {(c, k): int(((y == c) & (s == k)).sum()) for k in np.unique(s) for c in (0, 1)
            if ((y == c) & (s == k)).sum() > min_cell}


if __name__ == "__main__":
    ham = {(0, "rosendahl"): 803, (1, "rosendahl"): 342, (0, "vidir_modern"): 1832, (1, "vidir_modern"): 680,
           (0, "vidir_molemax"): 3720, (1, "vidir_molemax"): 24, (0, "vienna_dias"): 350, (1, "vienna_dias"): 67}
    cxr = {(0, "NIH"): 16368, (1, "NIH"): 3632, (0, "CheXpert"): 3580, (1, "CheXpert"): 16421}
    if "--demo" in sys.argv:
        report("NIH + CheXpert (natural)", cxr)
        report("HAM10000 (natural)", ham)
        hc = {(1, "rosendahl"): 243, (0, "rosendahl"): 100, (0, "vidir_modern"): 1300, (0, "vidir_molemax"): 2600,
              (0, "vienna_dias"): 241}
        report("HAM10000, constructed (melanoma at rosendahl only, 100 nevi as bridge)", hc)
        ALLOWED = (0,)
        report("HAM10000, constructed, only nevi (shared class) can be added", hc)
        report("NIH + CheXpert (natural), only no-finding images can be added", cxr)
        sys.exit()
    DIR = (sys.argv[1] if len(sys.argv) > 1 else DEF).rstrip("/")
    report("NIH + CheXpert (natural)", cxr)
    report("HAM10000 (natural)", counts_from_npz(f"{DIR}/ham10000.npz"))
    report("PTB-XL (natural, all sites)", counts_from_npz(f"{DIR}/ptbxl.npz"))
    report("ABIDE I+II (natural)", counts_from_npz(f"{DIR}/abide.npz"))
    hc = {(1, "rosendahl"): 243, (0, "rosendahl"): 100, (0, "vidir_modern"): 1300, (0, "vidir_molemax"): 2600,
          (0, "vienna_dias"): 241}
    report("HAM10000, constructed (melanoma at rosendahl only, 100 nevi as bridge)", hc)
    ALLOWED = (0,)
    report("HAM10000, constructed, only nevi (shared class) can be added", hc)
    report("NIH + CheXpert (natural), only no-finding images can be added", cxr)
    report("HAM10000 (natural), only nevi can be added", counts_from_npz(f"{DIR}/ham10000.npz"))
