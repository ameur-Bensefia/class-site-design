"""Paired comparison, same backbone (ResNet-18) and same splits: frozen ImageNet features versus end-to-end fine-tuning
with early stopping (results_e2e_es_constructed.jsonl), constructed confounding, 10 replicates.
For each method and bridge size: AUC frozen, AUC fine-tuned, paired difference (corrected resampled t-test, test
fraction 0.3), wins, and Holm adjustment across bridge sizes within each method.
Usage: python frozen_vs_e2e.py      (needs cxr_r18_frozen.npy from 'python p2_experiments.py r18')"""
import json
import os
import sys

import numpy as np
from scipy.stats import t as tdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_p1 import group_split, load_cxr  # noqa: E402
from p2_experiments import heads  # noqa: E402

E2E_NAME = {"ERM": "ERM", "ComBat": "ComBat on learned features",
            "Anchor-referenced CORAL": "Anchor-referenced CORAL on learned features"}
BRIDGES = (0, 100, 1000, -1)
REPS = 10


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


if __name__ == "__main__":
    _, Y, S, G = load_cxr()
    X0 = np.load("cxr_r18_frozen.npy")
    e2e = {}
    for line in open("results_e2e_es_constructed.jsonl"):
        r = json.loads(line)
        if "auc" in r:
            e2e[(r["method"], r["bridge_req"], r["rep"])] = r["auc"]
    frozen = {}
    for rep in range(REPS):
        for b in BRIDGES:
            rng = np.random.default_rng(200 + rep)              # identical to semireal_cxr / e2e_cxr
            tr, te = group_split(G, rng)
            keep = tr & (((Y == 1) & (S == "NIH")) | ((Y == 0) & (S == "CheXpert")))
            nf = np.where(tr & (Y == 0) & (S == "NIH"))[0]
            nb = len(nf) if b < 0 else min(b, len(nf))
            keep[rng.choice(nf, nb, replace=False)] = True
            test = te & (S == "CheXpert")
            X = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
            for k, sc in heads(X[keep], Y[keep], S[keep], X[test], S[test]).items():
                frozen[(k, b, rep)] = roc_auc_score(Y[test], sc)
        print(f"  replicate {rep + 1}/{REPS} done", flush=True)
    print("\nSame backbone (ResNet-18), constructed confounding, 10 paired replicates: frozen minus fine-tuned (early stopping)")
    print(f"{'method':26s} {'bridge':>7s} {'frozen':>7s} {'fine-tuned':>11s} {'difference [95% CI]':>24s} {'p':>8s} {'Holm p':>8s} {'wins':>6s}")
    for k, name in E2E_NAME.items():
        rows = []
        for b in BRIDGES:
            pairs = [(frozen[(k, b, r)], e2e[(name, b, r)]) for r in range(REPS) if (k, b, r) in frozen and (name, b, r) in e2e]
            if len(pairs) < 2:
                continue
            f, e = 100 * np.array(pairs).T
            d = f - e
            se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
            h = tdist.ppf(0.975, len(d) - 1) * se
            p = 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1)
            rows.append((b, f.mean(), e.mean(), d.mean(), h, p, int((d > 0).sum()), len(d)))
        adj = holm([r[5] for r in rows])
        for (b, fm, em, dm, h, p, w, n), pa in zip(rows, adj):
            print(f"{k:26s} {'all' if b < 0 else b:>7} {fm:7.1f} {em:11.1f} {dm:+8.1f} [{dm - h:+5.1f}, {dm + h:+5.1f}] {p:8.2g} {pa:8.2g} {w:>3d}/{n}")
