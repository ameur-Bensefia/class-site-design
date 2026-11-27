"""Semi-real test of disease-site separation on chest radiographs (NIH ChestX-ray14 and CheXpert).

Induced confounding: in training, pleural effusion (EFF) comes only from NIH; 'no finding' (NF) comes from both
hospitals, with a variable number of NIH NF images as the bridge. Test: EFF and NF from CheXpert patients, a hospital
where training contained only NF. Splits are by patient within each hospital. Metric: AUC (threshold-free).

Usage: python semireal_cxr.py [datasets_dir] [replicates]
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import json
import sys

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEF = DATASETS_DIR + "/"
DIR = sys.argv[1] if len(sys.argv) > 1 else DEF
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
BRIDGES = (0, 25, 50, 100, 300, 1000, -1)          # NIH 'no finding' images in training; -1 = all of the split

parts = []
for name, site in (("nih_cxr", "NIH"), ("chexpert", "CheXpert")):
    z = np.load(DIR.rstrip("/") + f"/{name}.npz", allow_pickle=True)
    parts.append((z["X"], z["y"].astype(int), np.full(len(z["y"]), site), np.char.add(site + ":", z["group"].astype(str))))
X0 = np.vstack([p[0] for p in parts])
Y = np.concatenate([p[1] for p in parts])
S = np.concatenate([p[2] for p in parts])
G = np.concatenate([p[3] for p in parts])
LINKED, TEST = "NIH", "CheXpert"


def patient_split(rng, frac_test=0.3):
    pats = np.unique(G)
    test_p = set(rng.choice(pats, int(frac_test * len(pats)), replace=False))
    te = np.array([g in test_p for g in G])
    return ~te, te


def sqrtm(C, power):
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.clip(w, 1e-8, None) ** power) @ V.T


def anchor_stats(X, mask, method):
    mu = X[mask].mean(0)
    return (mu, None) if method == "mean" else (mu, LedoitWolf().fit(X[mask]).covariance_)


def harmonize(X, S_, ref, stats, method):
    out = np.empty_like(X)
    mu_r, C_r = ref
    for s in np.unique(S_):
        m = S_ == s
        mu, C = stats[s]
        out[m] = X[m] - mu + mu_r if method == "mean" else (X[m] - mu) @ sqrtm(C, -0.5) @ sqrtm(C_r, 0.5) + mu_r
    return out


def auc(clf, Xte, yte):
    return float(roc_auc_score(yte, clf.decision_function(Xte)))


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
    rng = np.random.default_rng(200 + rep)
    tr, te = patient_split(rng)
    keep = tr & (((Y == 1) & (S == LINKED)) | ((Y == 0) & (S == TEST)))      # EFF at NIH only, NF at CheXpert
    nf_linked = np.where(tr & (Y == 0) & (S == LINKED))[0]
    nb = len(nf_linked) if n_bridge < 0 else min(n_bridge, len(nf_linked))
    keep[rng.choice(nf_linked, nb, replace=False)] = True                  # bridge: NF at NIH
    test = te & (S == TEST)
    oracle = tr                                                            # both classes from both hospitals
    pca = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)])
    X = pca.transform(X0)
    Xtr, ytr, Str = X[keep], Y[keep], S[keep]
    Xte, yte = X[test], Y[test]
    lr = lambda: LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")  # noqa: E731
    res = dict(n_bridge=int(nb))
    res["ERM"] = auc(lr().fit(Xtr, ytr), Xte, yte)
    site_mu = {s: Xtr[Str == s].mean(0) for s in np.unique(Str)}
    res["Site centring (class-blind)"] = auc(lr().fit(Xtr - np.array([site_mu[s] for s in Str]), ytr),
                                             Xte - site_mu[TEST], yte)
    if nb > 0:
        for method, name in (("mean", "Anchor centring (no finding)"), ("lw", "Anchor transform (no finding, LW)")):
            anchors = ytr == 0
            stats = {s: anchor_stats(Xtr, anchors & (Str == s), method) for s in np.unique(Str)}
            ref = anchor_stats(Xtr, anchors, method)
            clf = lr().fit(harmonize(Xtr, Str, ref, stats, method), ytr)
            res[name] = auc(clf, harmonize(Xte, np.full(len(Xte), TEST), ref, {TEST: stats[TEST]}, method), yte)
    cb = combat_auc(Xtr, ytr, Str, Xte, yte, TEST, lr)
    if cb is not None:
        res["ComBat (dx covariate)"] = cb
    res["Oracle (both classes, both hospitals)"] = auc(lr().fit(X[oracle], Y[oracle]), Xte, yte)
    return res


if __name__ == "__main__":
    out = []
    for nb in BRIDGES:
        for r in range(REPS):
            out.append(run(r, nb))
        print("bridge", nb, "done", flush=True)
    json.dump(out, open("semireal_cxr_results.json", "w"), indent=1)
    methods = ["ERM", "Site centring (class-blind)", "Anchor centring (no finding)",
               "Anchor transform (no finding, LW)", "ComBat (dx covariate)", "Oracle (both classes, both hospitals)"]
    print(f"\nTest: effusion vs no finding at CheXpert (training saw only 'no finding' there). AUC %, {REPS} replicates")
    print(f"{'NIH NF images':>14s} " + " ".join(f"{m[:26]:>27s}" for m in methods))
    for k, nb in enumerate(BRIDGES):
        g = out[k * REPS:(k + 1) * REPS]
        cells = []
        for m in methods:
            v = [100 * o[m] for o in g if m in o]
            cells.append(f"{np.mean(v):6.1f}±{1.96 * np.std(v, ddof=1) / np.sqrt(len(v)):4.1f}" if v else "n/a (no bridge)")
        print(f"{g[0]['n_bridge']:>14d} " + " ".join(f"{c:>27s}" for c in cells))
    print("\nPaired difference, anchor transform (LW) minus ERM:")
    for k, nb in enumerate(BRIDGES):
        g = out[k * REPS:(k + 1) * REPS]
        if "Anchor transform (no finding, LW)" in g[0]:
            d = 100 * np.array([o["Anchor transform (no finding, LW)"] - o["ERM"] for o in g])
            h = 2.262 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {g[0]['n_bridge']:>6d} NIH NF: {d.mean():+5.1f} [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}]  "
                  f"wins {int(np.sum(d > 0))}/{len(d)}")
        if "ComBat (dx covariate)" in g[0] and "Anchor transform (no finding, LW)" in g[0]:
            d = 100 * np.array([o["Anchor transform (no finding, LW)"] - o["ComBat (dx covariate)"] for o in g])
            h = 2.262 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"           vs ComBat: {d.mean():+5.1f} [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}]  wins {int(np.sum(d > 0))}/{len(d)}")
