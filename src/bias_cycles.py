"""Misspecification bias and its detectability on the class-site graph.

Model: x = mu_c + delta_d + b_(c,d) + noise, where b_(c,d) is a class-by-site interaction (departure from the additive
model). Site effects are estimated jointly by weighted least squares on cell means (conductance = sample count).

Predictions tested:
  P1. Bias of the estimated site contrast delta_u - delta_v equals sum_e i_e * b_e, where i is the unit electrical
      current between u and v in the bipartite graph (exact in expectation).
  P2. The weighted residual of the fit (cycle inconsistency) follows chi2 with D * (number of independent cycles)
      degrees of freedom under the additive model; it is identically zero on a tree, so no interaction is ever
      detectable on an acyclic graph, whatever the bias.
  P3. On graphs with cycles, the test detects interactions, with power increasing with their size and the number
      of independent cycles.
"""
import json
import sys

import numpy as np
from scipy.stats import chi2

D = 8


def design(kind, K=4, n_anchor=100):
    E = {("A", 0): 300, ("B", K - 1): 300}
    for d in range(K - 1):
        E[(f"S{d}", d)] = n_anchor
        E[(f"S{d}", d + 1)] = n_anchor
        if kind in ("ladder", "ladder+common"):
            E[(f"T{d}", d)] = n_anchor
            E[(f"T{d}", d + 1)] = n_anchor
    if kind == "ladder+common":
        for d in range(K):
            E[("C", d)] = n_anchor
    return E


def incidence(E):
    classes = sorted({c for c, _ in E})
    sites = sorted({d for _, d in E})
    nodes = [("c", c) for c in classes] + [("s", d) for d in sites]
    idx = {x: i for i, x in enumerate(nodes)}
    edges = list(E)
    Bm = np.zeros((len(edges), len(nodes)))
    for k, (c, d) in enumerate(edges):
        Bm[k, idx[("c", c)]] = 1          # cell mean = class potential + site potential
        Bm[k, idx[("s", d)]] = 1
    w = np.array([E[e] for e in edges], float)
    return edges, nodes, idx, Bm, w


def wls(Bm, w, M):
    """Weighted least squares of cell means M (edges x D) on class + site potentials; returns potentials, fitted."""
    W = np.sqrt(w)[:, None]
    theta, *_ = np.linalg.lstsq(Bm * W, M * W, rcond=None)
    return theta, Bm @ theta


def current(Bm, w, idx, u, v):
    """Contrast weights: estimate(theta_u - theta_v) = sum_e i_e * m_e for the minimum-norm WLS solution.
    For a contrast that is estimable, these weights are invariant to the gauge."""
    W = np.diag(w)
    G = Bm.T @ W @ Bm
    e = np.zeros(Bm.shape[1])
    e[idx[u]], e[idx[v]] = 1, -1
    return W @ Bm @ np.linalg.pinv(G) @ e


def n_cycles(E):
    _, nodes, _, Bm, _ = incidence(E)
    return len(E) - np.linalg.matrix_rank(Bm)


def run(kind, tau, seed, n_anchor=100, K=4):
    rng = np.random.default_rng(seed)
    E = design(kind, K, n_anchor)
    edges, nodes, idx, Bm, w = incidence(E)
    classes = sorted({c for c, _ in E})
    mu = {c: rng.normal(0, 1, D) for c in classes}
    delta = {d: rng.normal(0, 2.5, D) for d in range(K)}
    b = {e: rng.normal(0, tau, D) for e in edges}                    # class-by-site interaction
    M = np.array([mu[c] + delta[d] + b[(c, d)] + rng.normal(0, 1, (E[(c, d)], D)).mean(0) for c, d in edges])
    theta, fit = wls(Bm, w, M)
    u, v = ("s", 0), ("s", K - 1)
    est = theta[idx[u]] - theta[idx[v]]
    truth = delta[0] - delta[K - 1]
    i = current(Bm, w, idx, u, v)
    pred_bias = i @ np.array([b[e] for e in edges])
    rss = float(np.sum(w[:, None] * (M - fit) ** 2))                # noise variance 1 per coordinate, known
    dof = int(D * n_cycles(E))
    p = float(chi2.sf(rss, dof)) if dof > 0 else 1.0
    return dict(kind=kind, tau=tau, cycles=int(n_cycles(E)), obs_err=(est - truth).tolist(),
                pred_bias=pred_bias.tolist(), rss=rss, dof=dof, p=p,
                path_current=float(np.sum(np.abs(i))))


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = []
    for kind in ("chain", "ladder", "ladder+common"):
        for tau in (0.0, 0.05, 0.1, 0.2, 0.4):
            for r in range(reps):
                out.append(run(kind, tau, 7000 + r))
    json.dump(out, open("bias_cycles_results.json", "w"))
    print(f"{'graph':14s} {'cycles':>6s} {'tau':>5s} {'|bias| obs':>10s} {'|bias| pred':>11s} {'corr':>6s} {'reject':>7s}")
    for kind in ("chain", "ladder", "ladder+common"):
        for tau in (0.0, 0.05, 0.1, 0.2, 0.4):
            g = [o for o in out if o["kind"] == kind and o["tau"] == tau]
            obs, pred = np.array([o["obs_err"] for o in g]), np.array([o["pred_bias"] for o in g])
            # predicted bias explains the observed error up to noise: correlate coordinate-wise
            corr = np.corrcoef(obs.ravel(), pred.ravel())[0, 1] if tau > 0 else float("nan")
            print(f"{kind:14s} {g[0]['cycles']:6d} {tau:5.2f} {np.mean(np.linalg.norm(obs, axis=1)):10.3f} "
                  f"{np.mean(np.linalg.norm(pred, axis=1)):11.3f} {corr:6.2f} "
                  f"{100 * np.mean([o['p'] < 0.05 for o in g]):6.1f}%")
    # exactness of P1: regress (observed error - noise-only part) on predicted bias, pooled
    print("\nTotal current along the A-site to B-site contrast (sum |i_e|):",
          {k: round(next(o['path_current'] for o in out if o['kind'] == k), 2) for k in ("chain", "ladder", "ladder+common")})
