"""Two sweeps on the bridged design: number of anchors (controls) per site, and strength of the nonlinear
interaction between site and content. Key methods only."""
import json
import sys

import numpy as np

import feasibility as F


def one(seed, n_anchor, nonlinear):
    rng = np.random.default_rng(seed)
    w = F.World(rng, nonlinear=nonlinear)
    S = [w.new_site() for _ in range(3)]
    design = {0: ["C", "A"], 1: ["C", "B"]}
    Xtr, ytr, s_tr = F.build(w, design, S, {"C": n_anchor, "A": 300, "B": 300})
    Xcf, ycf, scf = F.build(w, {1: ["A"], 0: ["B"]}, S, 300)
    Xo, yo, so = F.build(w, {0: ["C", "A", "B"], 1: ["C", "A", "B"]}, S, 300)
    out = {}
    erm = F.train_net(rng, Xtr, ytr, s_tr)
    out["ERM"] = F.ab_pairwise(ycf, erm.logits(Xcf))
    dann = F.train_net(rng, Xtr, ytr, s_tr, adv="all")
    out["DANN"] = F.ab_pairwise(ycf, dann.logits(Xcf))
    Xa, ref = F.anchor_standardize(Xtr, s_tr, ytr)
    prop = F.train_net(rng, Xa, ytr, s_tr, adv="anchors")
    # transform counterfactual samples with the anchors of the site whose style they carry
    Xcf_t = np.empty_like(Xcf)
    for d in (0, 1):
        anchors = Xtr[(s_tr == d) & (ytr == 0)]
        Xd = np.vstack([anchors, Xcf[scf == d]])
        mask = np.r_[np.ones(len(anchors), bool), np.zeros((scf == d).sum(), bool)]
        tr, _ = F.anchor_standardize(Xd, np.full(len(Xd), d), np.zeros(len(Xd), int), ref_stats=ref, anchor_mask=mask)
        Xcf_t[scf == d] = tr[len(anchors):]
    out["Proposed"] = F.ab_pairwise(ycf, prop.logits(Xcf_t))
    ora = F.train_net(rng, Xo, yo, so)
    out["Oracle"] = F.ab_pairwise(ycf, ora.logits(Xcf))
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = {"anchors": {}, "nonlinear": {}}
    for na in (10, 25, 50, 100, 300):
        res["anchors"][na] = [one(2000 + r, na, 0.6) for r in range(reps)]
        print("anchors", na, flush=True)
    for nl in (0.0, 0.3, 0.6, 1.0):
        res["nonlinear"][nl] = [one(3000 + r, 300, nl) for r in range(reps)]
        print("nonlinear", nl, flush=True)
    json.dump(res, open("sweeps_results.json", "w"), indent=1)
    for k, block in res.items():
        print(f"\n=== sweep over {k}: counterfactual A-vs-B accuracy (%), chance = 50")
        for lvl, runs in block.items():
            print(f"  {k}={lvl:>5}: " + "  ".join(f"{m} {100 * np.mean([r[m] for r in runs]):5.1f}" for m in runs[0]))
