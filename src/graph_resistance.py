"""Does the effective resistance of the class-site graph govern how well disease is separated from site?

Several sites; disease A only at the first site, disease B only at the last. Shared classes connect the sites
either as a star (one class present at every site) or as a chain (each shared class links two neighbouring
sites). Site effects are estimated jointly on the whole graph by least squares on class-by-site means
(relative measurements), samples are harmonized, and a classifier is evaluated on a site-swapped test
(A rendered with the style of B's site and vice versa).

Prediction for an additive site effect with isotropic noise of variance s2 in D dimensions:
  E || (est. delta_A_site - est. delta_B_site) - (true difference) ||^2 = s2 * D * R_eff(site_A, site_B),
where R_eff is computed on the bipartite class-site graph with edge conductance n(c, d).
"""
import json
import sys
from itertools import product

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

D, DC, DS = 32, 12, 8


class World:
    def __init__(self, rng, classes, style_scale=2.5, nonlinear=0.0, effect=1.0):
        self.Wc = rng.normal(0, 1, (DC, D)) / np.sqrt(DC)
        self.Ws = rng.normal(0, 1, (DS, D)) / np.sqrt(DS)
        self.B = rng.normal(0, 1, (D, D)) / np.sqrt(D)
        self.mu = {c: rng.normal(0, 0.6, DC) for c in classes}
        self.mu["A"], self.mu["B"] = np.zeros(DC), np.zeros(DC)
        self.mu["A"][:3], self.mu["B"][3:6] = effect, effect           # disease directions
        self.style_scale, self.nl, self.rng = style_scale, nonlinear, rng

    def site(self):
        return self.rng.normal(0, self.style_scale, DS)

    def sample(self, c, style, n):
        z = self.mu[c] + self.rng.normal(0, 1, (n, DC))
        h = z @ self.Wc + (style + self.rng.normal(0, 0.2, (n, DS))) @ self.Ws
        return h + self.nl * np.tanh(h @ self.B) + self.rng.normal(0, 0.3, (n, D))


def design(topology, n_sites, n_anchor):
    """Return {(class, site): n}. A at site 0, B at the last site."""
    E = {("A", 0): 300, ("B", n_sites - 1): 300}
    if topology == "star":
        for d in range(n_sites):
            E[("S", d)] = n_anchor
    elif topology == "chain":
        for d in range(n_sites - 1):
            E[(f"S{d}", d)] = n_anchor
            E[(f"S{d}", d + 1)] = n_anchor
    return E


def effective_resistance(E, u, v):
    """Effective resistance between nodes u and v of the bipartite graph (conductance = sample count)."""
    nodes = sorted({("c", c) for c, _ in E} | {("s", d) for _, d in E})
    idx = {x: i for i, x in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for (c, d), n in E.items():
        i, j = idx[("c", c)], idx[("s", d)]
        L[i, i] += n
        L[j, j] += n
        L[i, j] -= n
        L[j, i] -= n
    Lp = np.linalg.pinv(L)
    e = np.zeros(len(nodes))
    e[idx[u]], e[idx[v]] = 1, -1
    return float(e @ Lp @ e)


def estimate_site_effects(means, counts, sites, classes):
    """Weighted least squares: m(c,d) = mu_c + delta_d, delta_0 = 0. Returns delta (n_sites x D)."""
    ci = {c: i for i, c in enumerate(classes)}
    nC, nS = len(classes), len(sites)
    rows, w, targets = [], [], []
    for (c, d), m in means.items():
        r = np.zeros(nC + nS - 1)
        r[ci[c]] = 1
        if d > 0:
            r[nC + d - 1] = 1
        rows.append(r)
        w.append(counts[(c, d)])
        targets.append(m)
    A, W, Y = np.array(rows), np.sqrt(np.array(w))[:, None], np.array(targets)
    theta, *_ = np.linalg.lstsq(A * W, Y * W, rcond=None)
    delta = np.zeros((nS, D))
    delta[1:] = theta[nC:]
    return delta


def run(seed, topology, n_sites, n_anchor, nonlinear=0.0):
    rng = np.random.default_rng(seed)
    E = design(topology, n_sites, n_anchor)
    classes = sorted({c for c, _ in E})
    w = World(rng, classes, nonlinear=nonlinear)
    S = [w.site() for _ in range(n_sites)]
    X, y, s = [], [], []
    for (c, d), n in E.items():
        X.append(w.sample(c, S[d], n))
        y += [classes.index(c)] * n
        s += [d] * n
    X, y, s = np.vstack(X), np.array(y), np.array(s)
    means = {(c, d): X[(y == classes.index(c)) & (s == d)].mean(0) for (c, d) in E}
    delta = estimate_site_effects(means, E, list(range(n_sites)), classes)
    # true relative site effect between A's and B's sites (additive part, measured with many noiseless samples)
    probe = {d: w.sample(classes[0], S[d], 4000).mean(0) for d in (0, n_sites - 1)}
    true_diff = probe[0] - probe[n_sites - 1]
    est_diff = delta[0] - delta[n_sites - 1]
    est_err = float(np.sum((est_diff - true_diff) ** 2))
    Xh = X - delta[s]
    clf = LogisticRegression(max_iter=4000).fit(Xh, y)
    iA, iB = classes.index("A"), classes.index("B")
    XA = w.sample("A", S[n_sites - 1], 400) - delta[n_sites - 1]   # A rendered with B's site style
    XB = w.sample("B", S[0], 400) - delta[0]                        # B rendered with A's site style
    LA, LB = clf.decision_function(XA), clf.decision_function(XB)
    acc = 0.5 * (np.mean(LA[:, iA] > LA[:, iB]) + np.mean(LB[:, iB] > LB[:, iA]))
    R = effective_resistance(E, ("s", 0), ("s", n_sites - 1))
    return dict(topology=topology, n_sites=n_sites, n_anchor=n_anchor, nonlinear=nonlinear, R_eff=R,
                est_err=est_err, cf_A_vs_B=float(acc))


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    configs = [(t, k, n) for t, k, n in product(("star", "chain"), (3, 4, 6), (10, 25, 50, 100, 300))]
    out = []
    for t, k, n in configs:
        for r in range(reps):
            out.append(run(5000 + r, t, k, n))
    json.dump(out, open("graph_resistance_results.json", "w"), indent=1)
    # noise variance of class means: per-dimension variance of x within (class, site)
    print(f"{'topology':8s} {'sites':>5s} {'anchors':>7s} {'R_eff':>8s} {'est. error':>10s} {'CF A-vs-B':>9s}")
    for t, k, n in configs:
        g = [o for o in out if (o["topology"], o["n_sites"], o["n_anchor"]) == (t, k, n)]
        print(f"{t:8s} {k:5d} {n:7d} {g[0]['R_eff']:8.4f} {np.mean([o['est_err'] for o in g]):10.3f} "
              f"{100 * np.mean([o['cf_A_vs_B'] for o in g]):8.1f}%")
    R = np.array([o["R_eff"] for o in out])
    err = np.array([o["est_err"] for o in out])
    acc = np.array([o["cf_A_vs_B"] for o in out])
    slope = float(np.sum(R * err) / np.sum(R * R))
    print(f"\nSpearman(R_eff, estimation error) = {spearmanr(R, err)[0]:.3f}")
    print(f"Spearman(R_eff, CF A-vs-B accuracy) = {spearmanr(R, acc)[0]:.3f}")
    print(f"Proportional fit: est. error ~ {slope:.2f} x R_eff (R^2 = "
          f"{1 - np.sum((err - slope * R) ** 2) / np.sum((err - err.mean()) ** 2):.3f})")
