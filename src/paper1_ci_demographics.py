"""Paper 1 (Radiology: AI) — 95% confidence intervals for the pooled-AUC decomposition and cohort demographics.

Run from src/ (it reuses audit_p1.py and p2_experiments.py):
    python paper1_ci_demographics.py
Writes paper1_ci_results.csv and paper1_demographics.txt, and prints both. About 10-20 minutes.
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = DATA_ROOT
DS = DATASETS_DIR
NIH_META = os.path.join(ROOT, "NIH ChestX", "Data_Entry_2017.csv")
CXP_META = [os.path.join(ROOT, "chexpert", f) for f in ("train.csv", "valid.csv")]
REPS, TEST_FRAC, EQ_DRAWS = 10, 0.3, 30

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ 1. demographics of the retained cohort
def demographics():
    out = []
    z = np.load(os.path.join(DS, "nih_cxr.npz"), allow_pickle=True)
    m = pd.read_csv(NIH_META).set_index("Image Index")
    kept = m.loc[[os.path.basename(str(p)) for p in z["paths"]]]
    pat = kept.sort_values("Follow-up #").groupby("Patient ID").first()
    out.append(("NIH ChestX-ray14", pat["Patient Age"].astype(float), pat["Patient Gender"].map({"M": "Male", "F": "Female"})))
    z = np.load(os.path.join(DS, "chexpert.npz"), allow_pickle=True)
    meta = pd.concat([pd.read_csv(f) for f in CXP_META if os.path.exists(f)], ignore_index=True)
    key = lambda p: str(p).replace("\\", "/").split("CheXpert-v1.0-small/")[-1].split("chexpert/")[-1]  # noqa: E731
    meta["key"] = meta["Path"].map(key)
    meta["patient"] = meta["Path"].str.extract(r"(patient\d+)")[0]
    kept = meta.set_index("key").loc[[key(p) for p in z["paths"]]]
    pat = kept.groupby("patient").first()
    out.append(("CheXpert", pat["Age"].astype(float), pat["Sex"].replace({"Unknown": np.nan})))
    lines = []
    for name, age, sex in out:
        lines.append(f"{name}: {len(age):,} patients; age (at first retained image) mean {age.mean():.1f} ± {age.std():.1f} years, range {age.min():.0f}–{age.max():.0f}")
        for s in ("Female", "Male"):
            a = age[sex == s]
            lines.append(f"  {s}: {len(a):,} ({100 * len(a) / len(age):.1f}%), age mean {a.mean():.1f} ± {a.std():.1f}, range {a.min():.0f}–{a.max():.0f}")
        n_other = len(age) - (sex == "Female").sum() - (sex == "Male").sum()
        if n_other: lines.append(f"  Sex unknown: {n_other}")
        implausible = int((age > 105).sum() + (age < 0).sum())
        if implausible: lines.append(f"  Note: {implausible} implausible ages (>105 or <0) in the source metadata")
    return "\n".join(lines)


# ------------------------------------------------------------------ 2. pooled-AUC decomposition with CIs
def decompose(y, s, score):
    pos, neg = y == 1, y == 0
    hosp = np.unique(s)
    within_auc, w_pairs, w_num = [], 0, 0.0
    for h in hosp:
        m = s == h
        if len(np.unique(y[m])) == 2:
            a = roc_auc_score(y[m], score[m]); n = pos[m].sum() * neg[m].sum()
            within_auc.append(a); w_num += a * n; w_pairs += n
    total = pos.sum() * neg.sum()
    pooled = roc_auc_score(y, score)
    cross = (pooled * total - w_num) / (total - w_pairs)
    return pooled, float(np.mean(within_auc)), w_num / w_pairs, cross


def equal_prevalence(y, s, score, rng):
    p = y.mean(); vals = []
    for _ in range(EQ_DRAWS):
        idx = []
        for h in np.unique(s):
            ip, ineg = np.where((s == h) & (y == 1))[0], np.where((s == h) & (y == 0))[0]
            k1 = min(len(ip), int(round(len(ineg) * p / (1 - p)))); k0 = min(len(ineg), int(round(k1 * (1 - p) / p)))
            idx += list(rng.choice(ip, k1, replace=False)) + list(rng.choice(ineg, k0, replace=False))
        idx = np.array(idx); vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.mean(vals))


def corrected_ci(v):
    v = np.asarray(v, dtype=float) * 100
    se = np.sqrt(v.var(ddof=1) * (1 / len(v) + TEST_FRAC / (1 - TEST_FRAC)))
    h = tdist.ppf(0.975, len(v) - 1) * se
    return v.mean(), v.mean() - h, v.mean() + h


def main():
    print(demographics()); open("paper1_demographics.txt", "w").write(demographics() + "\n")
    from audit_p1 import group_split, load_cxr
    from p2_experiments import heads
    X0, Y, S, G = load_cxr()
    rec = []
    for rep in range(REPS):
        rng = np.random.default_rng(200 + rep)
        tr, te = group_split(G, rng)
        X = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
        sc = heads(X[tr], Y[tr], S[tr], X[te], S[te])
        if rep == 0: print("methods returned by heads():", list(sc))
        prev = {h: Y[tr][S[tr] == h].mean() for h in np.unique(S[tr])}
        sc["Hospital only"] = np.array([prev[h] for h in S[te]])
        for name, score in sc.items():
            pooled, within_mean, within_pairs, cross = decompose(Y[te], S[te], np.asarray(score))
            eq = equal_prevalence(Y[te], S[te], np.asarray(score), rng)
            rec.append(dict(rep=rep, method=name, pooled=pooled, within_mean=within_mean, within_pairs=within_pairs, cross_pairs=cross, equal_prevalence=eq))
        print(f"  replicate {rep + 1}/{REPS} done", flush=True)
    df = pd.DataFrame(rec); df.to_csv("paper1_ci_per_replicate.csv", index=False)
    rows = []
    for name, g in df.groupby("method", sort=False):
        row = {"method": name}
        for col in ("pooled", "within_mean", "within_pairs", "cross_pairs", "equal_prevalence"):
            m, lo, hi = corrected_ci(g[col]); row[col] = f"{m:.1f} ({lo:.1f}, {hi:.1f})"
        gap = corrected_ci(g["pooled"] - g["within_mean"]); row["pooled_minus_within"] = f"{gap[0]:.1f} ({gap[1]:.1f}, {gap[2]:.1f})"
        rows.append(row)
    res = pd.DataFrame(rows); res.to_csv("paper1_ci_results.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\nMeans with 95% CIs (corrected resampled t, 10 patient-level splits), AUC in %:")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
