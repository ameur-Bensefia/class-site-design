"""Semi-real test of disease-site separation on HAM10000 (real images, real acquisition sources).

Induced confounding: in training, melanoma (MEL) comes only from source 'rosendahl'; nevus (NEV) comes from every
source, with a variable number of NEV at rosendahl (the bridge between the two classes). Test: MEL and NEV from
'vidir_modern', a source where training contained only nevi (site-swapped test on real data). Images are split by
lesion. Metric: AUC of MEL vs NEV on the test source (threshold-free), plus balanced accuracy.

Usage: python semireal_ham.py [path/to/ham10000.npz] [replicates]
"""
import os
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import json
import sys

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

DEFAULT = os.path.join(DATASETS_DIR, "ham10000.npz")
PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LINKED, TEST = "rosendahl", "vidir_modern"
BRIDGES = (0, 10, 25, 50, 100, 300, -1)          # -1 = all rosendahl nevi of the training split

z = np.load(PATH, allow_pickle=True)
X0, Y, S, G = z["X"], z["y"].astype(int), z["site"].astype(str), z["group"].astype(str)


def lesion_split(rng, frac_test=0.3):
    lesions = np.unique(G)
    test_les = set(rng.choice(lesions, int(frac_test * len(lesions)), replace=False))
    te = np.array([g in test_les for g in G])
    return ~te, te


def sqrtm(C, power):
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.clip(w, 1e-8, None) ** power) @ V.T


def anchor_stats(X, mask, method):
    mu = X[mask].mean(0)
    if method == "mean":
        return mu, None
    C = LedoitWolf().fit(X[mask]).covariance_
    return mu, C


def harmonize(X, S, ref, stats, method):
    out = np.empty_like(X)
    mu_r, C_r = ref
    for s in np.unique(S):
        m = S == s
        mu, C = stats[s]
        if method == "mean":
            out[m] = X[m] - mu + mu_r
        else:
            out[m] = (X[m] - mu) @ sqrtm(C, -0.5) @ sqrtm(C_r, 0.5) + mu_r
    return out


def evaluate(clf, Xte, yte):
    sc = clf.decision_function(Xte)
    return dict(auc=float(roc_auc_score(yte, sc)), ba=float(balanced_accuracy_score(yte, (sc > 0).astype(int))))


def combat_apply(Xnew, site, est):
    """Apply trained neuroCombat estimates to new samples of a known training site (same formula as the package's
    neuroCombatFromTraining, which fails with recent NumPy versions)."""
    levels = list(np.asarray(est["batches"]).astype(str))
    b = levels.index(str(site))
    var = np.asarray(est["var.pooled"]).reshape(-1, 1)
    mean = np.asarray(est["stand.mean"])[:, [0]]
    z = (Xnew.T - mean) / np.sqrt(var)
    z = (z - np.asarray(est["gamma.star"])[b][:, None]) / np.sqrt(np.asarray(est["delta.star"])[b][:, None])
    return (z * np.sqrt(var) + mean).T


def combat_auc(Xtr, ytr, Str, Xte, yte, test_site, lr):
    """Official ComBat (neuroCombat, empirical Bayes) fitted on training data with the diagnosis as a biological
    covariate to preserve; the same estimates are applied to the test site. Returns None when the design is
    confounded (diagnosis collinear with site), where ComBat cannot be fitted."""
    import warnings

    import pandas as pd
    from neuroCombat import neuroCombat
    covars = pd.DataFrame({"site": Str, "dx": ytr.astype(int)})
    D = pd.get_dummies(covars["site"]).astype(float).values
    design = np.c_[D, covars["dx"].values]
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = neuroCombat(dat=Xtr.T, covars=covars, batch_col="site", categorical_cols=["dx"])
            Xh = fit["data"].T
            Xte_h = combat_apply(Xte, test_site, fit["estimates"])
        clf = lr().fit(Xh, ytr)
        return float(roc_auc_score(yte, clf.decision_function(Xte_h)))
    except Exception:
        return None


def run(rep, n_bridge):
    rng = np.random.default_rng(100 + rep)
    tr, te = lesion_split(rng)
    # confounded training set
    keep = np.zeros(len(Y), bool)
    keep |= tr & (Y == 1) & (S == LINKED)                                  # MEL only at the linked source
    keep |= tr & (Y == 0) & (S != LINKED)                                  # NEV at every other source
    nev_linked = np.where(tr & (Y == 0) & (S == LINKED))[0]
    nb = len(nev_linked) if n_bridge < 0 else min(n_bridge, len(nev_linked))
    keep[rng.choice(nev_linked, nb, replace=False)] = True                 # bridge: NEV at the linked source
    test = te & (S == TEST)
    oracle = tr & ((Y == 0) | ((Y == 1) & np.isin(S, [LINKED, TEST, "vienna_dias"])))
    pca = PCA(128, random_state=rep).fit(X0[tr])
    X = pca.transform(X0)
    Xtr, ytr, Str = X[keep], Y[keep], S[keep]
    Xte, yte = X[test], Y[test]
    res = dict(n_bridge=nb, R_eff=(1 / nb + 1 / max(1, ((Str == TEST) & (ytr == 0)).sum())) if nb > 0 else float("inf"))
    lr = lambda: LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")  # noqa: E731
    res["ERM"] = evaluate(lr().fit(Xtr, ytr), Xte, yte)
    # site harmonization ignoring the class: centre each source on the mean of all its training images
    site_mu = {s: Xtr[Str == s].mean(0) for s in np.unique(Str)}
    Xh = Xtr - np.array([site_mu[s] for s in Str])
    res["Site centring (class-blind)"] = evaluate(lr().fit(Xh, ytr), Xte - site_mu[TEST], yte)
    if nb > 0:
        for method, name in (("mean", "Anchor centring (nevi)"), ("lw", "Anchor transform (nevi, Ledoit-Wolf)")):
            anchors = ytr == 0
            stats = {s: anchor_stats(Xtr, anchors & (Str == s), method) for s in np.unique(Str)}
            ref = anchor_stats(Xtr, anchors, method)
            Xa = harmonize(Xtr, Str, ref, stats, method)
            clf = lr().fit(Xa, ytr)
            Xte_a = harmonize(Xte, np.full(len(Xte), TEST), ref, {TEST: stats[TEST]}, method)
            res[name] = evaluate(clf, Xte_a, yte)
    cb = combat_auc(Xtr, ytr, Str, Xte, yte, TEST, lr)
    if cb is not None:
        res["ComBat (dx covariate)"] = dict(auc=cb, ba=float("nan"))
    res["Oracle (melanoma from three sources)"] = evaluate(lr().fit(X[oracle], Y[oracle]), Xte, yte)
    return res


if __name__ == "__main__":
    out = []
    for nb in BRIDGES:
        for r in range(REPS):
            out.append(run(r, nb))
        print("bridge", nb, "done", flush=True)
    json.dump(out, open("semireal_ham_results.json", "w"), indent=1)
    methods = ["ERM", "Site centring (class-blind)", "Anchor centring (nevi)", "Anchor transform (nevi, Ledoit-Wolf)",
               "ComBat (dx covariate)", "Oracle (melanoma from three sources)"]
    print(f"\nTest: melanoma vs nevus at '{TEST}' (training saw only nevi there). AUC (%), mean over {REPS} replicates")
    print(f"{'nevi at ' + LINKED:>18s} {'R_eff':>7s} " + " ".join(f"{m[:24]:>25s}" for m in methods))
    for nb in BRIDGES:
        g = [o for o in out if (o["n_bridge"] == nb) or (nb < 0 and o is out[-1] and False)]
        g = [o for o in out if BRIDGES.index(nb) * REPS <= out.index(o) < (BRIDGES.index(nb) + 1) * REPS]
        cells = []
        for m in methods:
            v = [100 * o[m]["auc"] for o in g if m in o]
            cells.append(f"{np.mean(v):6.1f}±{1.96 * np.std(v, ddof=1) / np.sqrt(len(v)):4.1f}" if v else "   n/a (no bridge)")
        nbs = g[0]["n_bridge"]
        R = np.mean([o["R_eff"] for o in g])
        print(f"{nbs:>18d} {R:7.4f} " + " ".join(f"{c:>25s}" for c in cells))

    print("\nPaired difference, anchor transform (LW) minus ComBat (AUC points):")
    for k, nb in enumerate(BRIDGES):
        g = out[k * REPS:(k + 1) * REPS]
        if "ComBat (dx covariate)" in g[0] and "Anchor transform (nevi, Ledoit-Wolf)" in g[0]:
            d = 100 * np.array([o["Anchor transform (nevi, Ledoit-Wolf)"]["auc"] - o["ComBat (dx covariate)"]["auc"] for o in g])
            h = 2.262 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {g[0]['n_bridge']:>6d} nevi: {d.mean():+5.1f} [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}]  wins {int(np.sum(d > 0))}/{len(d)}")
