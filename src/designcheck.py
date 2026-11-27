"""Design check for multi-site studies, from case counts only (no images, no model).

Input: a CSV file with columns class, site, n (one row per observed class-site cell).
Output, for every pair of classes: whether the contrast is estimable (connected classes), the effective sample size
1/R_eff, the design efficiency relative to n_A n_B / (n_A + n_B), and the number of independent cycles of the graph.
For two classes, it also reports the share of positive-negative pairs from the same site (pi_w) and the pooled AUC that
a predictor using only the site can achieve (A_site), and suggests where to add samples.

Usage:
    python designcheck.py counts.csv [--positive CLASS] [--budget 1000] [--step 50]

Example counts.csv:
    class,site,n
    effusion,NIH,3632
    no_finding,NIH,16368
    effusion,CheXpert,16421
    no_finding,CheXpert,3580
"""
import argparse
import itertools

import numpy as np
import pandas as pd


def graph(cells):
    """Bipartite class-site graph: node list, edge list (class index, site index, conductance)."""
    classes = sorted(cells["class"].unique()); sites = sorted(cells["site"].unique())
    nodes = [("class", c) for c in classes] + [("site", s) for s in sites]
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[("class", r["class"])], idx[("site", r["site"])], float(r["n"])) for _, r in cells.iterrows() if r["n"] > 0]
    return classes, sites, nodes, idx, edges


def laplacian(n_nodes, edges):
    L = np.zeros((n_nodes, n_nodes))
    for u, v, w in edges:
        L[u, u] += w; L[v, v] += w; L[u, v] -= w; L[v, u] -= w
    return L


def components(n_nodes, edges):
    parent = list(range(n_nodes))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for u, v, _ in edges:
        parent[find(u)] = find(v)
    return [find(i) for i in range(n_nodes)]


def effective_resistance(Lp, u, v):
    e = np.zeros(Lp.shape[0]); e[u], e[v] = 1.0, -1.0
    return float(e @ Lp @ e)


def potentials(Lp, u, v):
    e = np.zeros(Lp.shape[0]); e[u], e[v] = 1.0, -1.0
    return Lp @ e


def source_only_auc(cells, positive):
    """pi_w (share of within-site positive-negative pairs) and A_site (pooled AUC of a site-only predictor)."""
    pos = cells[cells["class"] == positive].set_index("site")["n"]
    neg = cells[cells["class"] != positive].groupby("site")["n"].sum()
    sites = sorted(set(pos.index) | set(neg.index))
    n1 = np.array([pos.get(s, 0.0) for s in sites], float); n0 = np.array([neg.get(s, 0.0) for s in sites], float)
    p = np.where(n1 + n0 > 0, n1 / np.maximum(n1 + n0, 1e-12), 0.0)
    total = n1.sum() * n0.sum()
    pi_w = float((n1 * n0).sum() / total)
    a = sum(n1[d] * n0[e] * (1.0 if p[d] > p[e] else 0.5 if p[d] == p[e] else 0.0) for d in range(len(sites)) for e in range(len(sites)))
    return pi_w, float(a / total)


def greedy_allocation(cells, a, b, budget, step):
    """Add samples in increments to the class-site cell with the largest squared potential difference (Proposition 5)."""
    cells = cells.copy()
    classes = sorted(cells["class"].unique()); sites = sorted(cells["site"].unique())
    full = pd.DataFrame([(c, s) for c in classes for s in sites], columns=["class", "site"])
    cells = full.merge(cells, on=["class", "site"], how="left").fillna({"n": 0.0})
    added = {}
    for _ in range(int(budget // step)):
        _, _, nodes, idx, edges = graph(cells)
        # include empty cells with a tiny conductance so that their marginal value is defined
        all_edges = [(idx[("class", r["class"])], idx[("site", r["site"])], max(float(r["n"]), 1e-6)) for _, r in cells.iterrows()]
        Lp = np.linalg.pinv(laplacian(len(nodes), all_edges))
        phi = potentials(Lp, idx[("class", a)], idx[("class", b)])
        gain = [(phi[idx[("class", r["class"])]] - phi[idx[("site", r["site"])]]) ** 2 for _, r in cells.iterrows()]
        k = int(np.argmax(gain))
        cells.loc[k, "n"] += step
        key = (cells.loc[k, "class"], cells.loc[k, "site"]); added[key] = added.get(key, 0) + step
    return cells, added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counts"); ap.add_argument("--positive", default=None, help="positive class, for pi_w and A_site (two classes)")
    ap.add_argument("--budget", type=float, default=0, help="number of samples to allocate (0: no allocation)")
    ap.add_argument("--step", type=float, default=50)
    args = ap.parse_args()
    cells = pd.read_csv(args.counts)[["class", "site", "n"]].groupby(["class", "site"], as_index=False)["n"].sum()
    classes, sites, nodes, idx, edges = graph(cells)
    comp = components(len(nodes), edges)
    n_comp = len(set(comp)); cycles = len(edges) - len(nodes) + n_comp
    Lp = np.linalg.pinv(laplacian(len(nodes), edges))
    totals = cells.groupby("class")["n"].sum()
    print(f"{len(classes)} classes, {len(sites)} sites, {len(edges)} observed cells, {n_comp} connected component(s), {cycles} independent cycle(s)")
    for a, b in itertools.combinations(classes, 2):
        u, v = idx[("class", a)], idx[("class", b)]
        if comp[u] != comp[v]:
            print(f"  {a} vs {b}: NOT ESTIMABLE (no bridge between the classes)"); continue
        r = effective_resistance(Lp, u, v); bound = totals[a] * totals[b] / (totals[a] + totals[b])
        print(f"  {a} vs {b}: effective sample size 1/R_eff = {1 / r:,.1f}; efficiency = {(1 / r) / bound:.2f}")
    if len(classes) == 2:
        positive = args.positive or classes[0]
        pi_w, a_site = source_only_auc(cells, positive)
        print(f"  within-site share of positive-negative pairs pi_w = {pi_w:.3f}; pooled AUC achievable by the site alone = {100 * a_site:.1f}%")
        if args.budget > 0:
            new, added = greedy_allocation(cells, classes[0], classes[1], args.budget, args.step)
            _, _, nodes2, idx2, edges2 = graph(new)
            r2 = effective_resistance(np.linalg.pinv(laplacian(len(nodes2), edges2)), idx2[("class", classes[0])], idx2[("class", classes[1])])
            print(f"  greedy allocation of {args.budget:,.0f} samples: effective sample size {1 / r2:,.1f}")
            for (c, s), k in sorted(added.items(), key=lambda t: -t[1]):
                print(f"    add {k:,.0f} {c} at {s}")


if __name__ == "__main__":
    main()
