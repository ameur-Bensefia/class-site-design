"""Audits of the frozen-feature pipeline: ComBat at inference, patient-level design checks and pooled-AUC decomposition.

  python audit_p1.py combat     ComBat at inference: (i) what the trained estimates contain, (ii) the test transform uses
                                no test diagnosis (invariance under permutation of test labels), (iii) frozen-feature
                                comparison of three ComBat variants on NIH/CheXpert (natural pool and constructed
                                confounding): empirical Bayes (as in the paper), no empirical Bayes, and location-only
                                without empirical Bayes (the estimator covered by Theorem 1).
  python audit_p1.py patients   effective sample size 1/R_eff computed on samples (images/patches) and on independent
                                units (patients, lesions or participants) per class-site cell.
  python audit_p1.py auc        decomposition of the pooled AUC on the natural NIH/CheXpert and HAM10000 pools:
                                within-site pairs, cross-site pairs, a predictor using the site only, and a test set
                                resampled to equal prevalence across sites.
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DIR = DATASETS_DIR
MODE = sys.argv[1] if len(sys.argv) > 1 else "combat"
if len(sys.argv) > 2:
    DIR = sys.argv[2].rstrip("/")


# ------------------------------------------------------------------ data
def load_cxr():
    parts = []
    for name, site in (("nih_cxr", "NIH"), ("chexpert", "CheXpert")):
        z = np.load(f"{DIR}/{name}.npz", allow_pickle=True)
        parts.append((z["X"], z["y"].astype(int), np.full(len(z["y"]), site),
                      np.char.add(site + ":", z["group"].astype(str))))
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))


def load_npz(name):
    z = np.load(f"{DIR}/{name}.npz", allow_pickle=True)
    return z["X"], z["y"].astype(int), z["site"].astype(str), z["group"].astype(str)


def group_split(G, rng, frac=0.3):
    g = np.unique(G)
    te_g = set(rng.choice(g, int(frac * len(g)), replace=False))
    te = np.array([x in te_g for x in G])
    return ~te, te


def lr():
    return LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")


def sqrtm(C, p):
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.clip(w, 1e-8, None) ** p) @ V.T


# ------------------------------------------------------------------ ComBat: fit and test-time transform
def combat_fit(Xtr, ytr, Str, eb=True, mean_only=False):
    from neuroCombat import neuroCombat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return neuroCombat(dat=Xtr.T, covars=pd.DataFrame({"site": Str, "dx": ytr}), batch_col="site",
                           categorical_cols=["dx"], eb=eb, mean_only=mean_only)


def combat_apply(Xnew, Snew, est):
    """Test-time transform. Inputs: features and site of the new samples, and the trained estimates.
    The diagnosis of the new samples is NOT an input, so it cannot influence the transform."""
    lev = list(np.asarray(est["batches"]).astype(str))
    var = np.asarray(est["var.pooled"]).reshape(-1, 1)
    mean = np.asarray(est["stand.mean"])[:, [0]]
    out = np.empty_like(Xnew)
    for s in np.unique(Snew):
        b = lev.index(str(s))
        z = (Xnew[Snew == s].T - mean) / np.sqrt(var)
        z = (z - np.asarray(est["gamma.star"])[b][:, None]) / np.sqrt(np.asarray(est["delta.star"])[b][:, None])
        out[Snew == s] = (z * np.sqrt(var) + mean).T
    return out


def audit_combat():
    X0, Y, S, G = load_cxr()
    rng = np.random.default_rng(300)
    tr, te = group_split(G, rng)
    X = PCA(min(128, X0.shape[1] - 1), random_state=0).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
    fit = combat_fit(X[tr], Y[tr], S[tr])
    est = fit["estimates"]
    sm = np.asarray(est["stand.mean"])
    print("(i) trained estimates:", sorted(est.keys()))
    print(f"    stand.mean shape {sm.shape}; max difference between its columns = {np.abs(sm - sm[:, [0]]).max():.2e} "
          "(0 means it holds the grand mean only, with no diagnosis effect)")
    if "mod.mean" in est:
        mm = np.asarray(est["mod.mean"])
        print(f"    mod.mean (diagnosis effect added back to TRAINING data only) shape {mm.shape}, "
              f"mean |value| {np.abs(mm).mean():.3f}")
    a = combat_apply(X[te], S[te], est)
    b = combat_apply(X[te], S[te], est)               # labels are not an argument; permuting them cannot matter
    yperm = np.random.default_rng(1).permutation(Y[te])
    print(f"(ii) transform of test features is identical across calls: {np.array_equal(a, b)}; the diagnosis is not an "
          f"input of combat_apply (permuted labels differ from true labels in {np.mean(yperm != Y[te]):.0%} of cases "
          "and are never used)")
    # (iii) three variants, natural pool and constructed confounding, 10 replicates
    print("\n(iii) frozen ResNet-50 features, NIH/CheXpert, 10 replicates")
    variants = {"ComBat EB (paper)": dict(eb=True), "ComBat no EB": dict(eb=False),
                "ComBat location-only, no EB (Theorem 1)": dict(eb=False, mean_only=True)}
    res = {k: {"nat_within": [], "nat_gap": [], **{f"b{b}": [] for b in (100, 1000, -1)}} for k in variants}
    for rep in range(10):
        rng = np.random.default_rng(300 + rep)
        tr, te = group_split(G, rng)
        X = PCA(min(128, X0.shape[1] - 1), random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
        for k, kw in variants.items():
            f = combat_fit(X[tr], Y[tr], S[tr], **kw)
            sc = lr().fit(f["data"].T, Y[tr]).decision_function(combat_apply(X[te], S[te], f["estimates"]))
            within = np.mean([roc_auc_score(Y[te][S[te] == s], sc[S[te] == s]) for s in ("NIH", "CheXpert")])
            res[k]["nat_within"].append(within)
            res[k]["nat_gap"].append(roc_auc_score(Y[te], sc) - within)
        rng = np.random.default_rng(200 + rep)            # constructed confounding, as in semireal_cxr.py
        tr, te = group_split(G, rng)
        Xc = PCA(min(128, X0.shape[1] - 1), random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
        nf = np.where(tr & (Y == 0) & (S == "NIH"))[0]
        perm = rng.permutation(nf)
        test = te & (S == "CheXpert")
        for bsz in (100, 1000, -1):
            keep = tr & (((Y == 1) & (S == "NIH")) | ((Y == 0) & (S == "CheXpert")))
            keep[perm[:len(nf) if bsz < 0 else bsz]] = True
            for k, kw in variants.items():
                f = combat_fit(Xc[keep], Y[keep], S[keep], **kw)
                sc = lr().fit(f["data"].T, Y[keep]).decision_function(combat_apply(Xc[test], S[test], f["estimates"]))
                res[k][f"b{bsz}"].append(roc_auc_score(Y[test], sc))
        print(f"  replicate {rep + 1} done", flush=True)
    print(f"\n{'variant':42s} {'natural within':>15s} {'natural gap':>12s} {'bridge 100':>11s} {'bridge 1000':>12s} {'bridge all':>11s}")
    for k, v in res.items():
        print(f"{k:42s} {100 * np.mean(v['nat_within']):15.1f} {100 * np.mean(v['nat_gap']):+12.1f} "
              f"{100 * np.mean(v['b100']):11.1f} {100 * np.mean(v['b1000']):12.1f} {100 * np.mean(v['b-1']):11.1f}")
    base = np.array(res["ComBat EB (paper)"]["b-1"])
    for k in list(variants)[1:]:
        for key in ("b100", "b1000", "b-1", "nat_within"):
            d = 100 * (np.array(res[k][key]) - np.array(res["ComBat EB (paper)"][key]))
            se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
            p = 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1) if se > 0 else 1.0
            print(f"  {k} minus EB, {key}: {d.mean():+.2f} points (corrected p={p:.2g})")
    _ = base


# ------------------------------------------------------------------ patient-level design audit
def r_eff(counts):
    sites = sorted({s for _, s in counts})
    idx = {("c", 0): 0, ("c", 1): 1, **{("s", s): 2 + i for i, s in enumerate(sites)}}
    L = np.zeros((len(idx), len(idx)))
    for (c, s), n in counts.items():
        if n > 0:
            i, j = idx[("c", c)], idx[("s", s)]
            L[i, i] += n; L[j, j] += n; L[i, j] -= n; L[j, i] -= n
    e = np.zeros(len(idx)); e[0], e[1] = 1, -1
    return float(e @ np.linalg.pinv(L) @ e)


def audit_patients():
    rows = []
    sets = [("NIH + CheXpert", load_cxr(), "patients"), ("HAM10000", load_npz("ham10000"), "lesions"),
            ("PTB-XL", load_npz("ptbxl"), "patients"), ("ABIDE I+II", load_npz("abide"), "participants")]
    try:
        z = np.load("camelyon17.npz", allow_pickle=True)
        sets.append(("Camelyon17 (100,000-patch subsample)", (z["X"], z["y"].astype(int), z["site"].astype(str),
                                                               z["group"].astype(str)), "patients"))
    except FileNotFoundError:
        print("camelyon17.npz not found in the current folder; skipping Camelyon17")
    for name, (X, Y, S, G), unit in sets:
        samp = {(c, s): int(((Y == c) & (S == s)).sum()) for s in np.unique(S) for c in (0, 1)}
        units = {(c, s): len(np.unique(G[(Y == c) & (S == s)])) for s in np.unique(S) for c in (0, 1)}
        nA, nB = sum(v for (c, _), v in samp.items() if c == 1), sum(v for (c, _), v in samp.items() if c == 0)
        uA, uB = sum(v for (c, _), v in units.items() if c == 1), sum(v for (c, _), v in units.items() if c == 0)
        rows.append(dict(data=name, unit=unit, samples=len(Y), units=len(np.unique(G)),
                         n_eff_samples=round(1 / r_eff(samp)), n_eff_units=round(1 / r_eff(units)),
                         efficiency_samples=round((1 / r_eff(samp)) / (nA * nB / (nA + nB)), 3),
                         efficiency_units=round((1 / r_eff(units)) / (uA * uB / (uA + uB)), 3)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nn_eff = 1/R_eff; efficiency = n_eff divided by its upper bound nA*nB/(nA+nB) (Corollary 1, revised).")


# ------------------------------------------------------------------ AUC decomposition on natural pools
def decompose(y, sc, site):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    same = site[pos][:, None] == site[neg][None, :]
    gt = (sc[pos][:, None] > sc[neg][None, :]) + 0.5 * (sc[pos][:, None] == sc[neg][None, :])
    return float(gt[same].mean()), float(gt[~same].mean()), float(same.mean())


def audit_auc():
    for name, data in (("NIH + CheXpert", load_cxr()), ("HAM10000", load_npz("ham10000"))):
        X0, Y, S, G = data
        sites = np.unique(S)
        out = {}
        for rep in range(10):
            rng = np.random.default_rng(300 + rep if name.startswith("NIH") else 400 + rep)
            tr, te = group_split(G, rng)
            sub = rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)
            X = PCA(min(128, X0.shape[1] - 1), random_state=rep).fit(X0[tr][sub]).transform(X0)
            Xtr, ytr, Str, Xte, yte, Ste = X[tr], Y[tr], S[tr], X[te], Y[te], S[te]
            scores = {"Standard training": lr().fit(Xtr, ytr).decision_function(Xte)}
            f = combat_fit(Xtr, ytr, Str)
            scores["ComBat (diagnosis covariate)"] = lr().fit(f["data"].T, ytr).decision_function(combat_apply(Xte, Ste, f["estimates"]))
            mu = {s: Xtr[Str == s].mean(0) for s in sites}
            scores["Class-blind site centring"] = lr().fit(Xtr - np.array([mu[s] for s in Str]), ytr).decision_function(
                Xte - np.array([mu[s] for s in Ste]))
            ref_mu, ref_C = Xtr[ytr == 0].mean(0), LedoitWolf().fit(Xtr[ytr == 0]).covariance_
            st = {s: (Xtr[(Str == s) & (ytr == 0)].mean(0), LedoitWolf().fit(Xtr[(Str == s) & (ytr == 0)]).covariance_) for s in sites}
            def coral(A, Sa):
                o = np.empty_like(A)
                for s in sites:
                    m = Sa == s
                    o[m] = (A[m] - st[s][0]) @ sqrtm(st[s][1], -0.5) @ sqrtm(ref_C, 0.5) + ref_mu
                return o
            scores["Anchor-referenced CORAL"] = lr().fit(coral(Xtr, Str), ytr).decision_function(coral(Xte, Ste))
            prev = {s: ytr[Str == s].mean() for s in sites}
            scores["Site only (training prevalence)"] = np.array([prev[s] for s in Ste])   # ties counted as one half by roc_auc_score
            for m, sc in scores.items():
                within_pairs, cross_pairs, frac_same = decompose(yte, sc, Ste)
                within_mean = np.mean([roc_auc_score(yte[Ste == s], sc[Ste == s]) for s in sites
                                       if len(np.unique(yte[Ste == s])) == 2])
                bal = []                                   # equal prevalence across sites: subsample positives or negatives
                target = yte.mean()
                for _ in range(30):
                    keep = []
                    for s in sites:
                        idx_p, idx_n = np.where((Ste == s) & (yte == 1))[0], np.where((Ste == s) & (yte == 0))[0]
                        n_p = min(len(idx_p), int(round(len(idx_n) * target / (1 - target))))
                        n_n = min(len(idx_n), int(round(n_p * (1 - target) / target)))
                        keep += list(rng.choice(idx_p, n_p, replace=False)) + list(rng.choice(idx_n, n_n, replace=False))
                    keep = np.array(keep)
                    bal.append(roc_auc_score(yte[keep], sc[keep]))
                d = out.setdefault(m, {k: [] for k in ("pooled", "within_mean", "within_pairs", "cross_pairs", "balanced", "frac_same")})
                for k, v in (("pooled", roc_auc_score(yte, sc)), ("within_mean", within_mean), ("within_pairs", within_pairs),
                             ("cross_pairs", cross_pairs), ("balanced", np.mean(bal)), ("frac_same", frac_same)):
                    d[k].append(v)
        print(f"\n=== {name}, natural pool, frozen features, 10 replicates (AUC %)")
        print(f"{'method':34s} {'pooled':>7s} {'within (mean)':>13s} {'within pairs':>13s} {'cross-site pairs':>17s} "
              f"{'equal-prevalence pooled':>24s}")
        for m, d in out.items():
            print(f"{m:34s} {100 * np.mean(d['pooled']):7.1f} {100 * np.mean(d['within_mean']):13.1f} "
                  f"{100 * np.mean(d['within_pairs']):13.1f} {100 * np.mean(d['cross_pairs']):17.1f} "
                  f"{100 * np.mean(d['balanced']):24.1f}")
        print(f"(fraction of positive-negative test pairs from the same site: {100 * np.mean(out['Standard training']['frac_same']):.0f}%)")


if __name__ == "__main__":
    {"combat": audit_combat, "patients": audit_patients, "auc": audit_auc}[MODE]()
