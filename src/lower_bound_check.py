"""Simulated check of the lower bound: on random connected class-site designs, the variance of three unbiased
estimators of a class contrast is compared with the Cramer-Rao / Gauss-Markov bound sigma^2 * R_eff(u, v)."""
import numpy as np
from collections import deque

def laplacian(cells, nC, nS):
    L = np.zeros((nC + nS, nC + nS))
    for (c, s), n in cells.items():
        i, j = c, nC + s
        L[i, i] += n; L[j, j] += n; L[i, j] -= n; L[j, i] -= n
    return L

def contrast_weights(cells, nC, nS, u, v, weighted=True):
    """Weights w_e such that the (W)LS estimate of mu_u - mu_v equals sum_e w_e m_e (site potentials as -delta)."""
    keys = list(cells)
    B = np.zeros((len(keys), nC + nS))
    for k, (c, s) in enumerate(keys):
        B[k, c] = 1; B[k, nC + s] = -1                      # m_e = mu_c - (-delta_s): difference of potentials
    W = np.diag([cells[k] if weighted else 1.0 for k in keys])
    e = np.zeros(nC + nS); e[u], e[v] = 1, -1
    return keys, W @ B @ np.linalg.pinv(B.T @ W @ B) @ e

def shortest_path_weights(cells, nC, nS, u, v):
    """Unbiased estimator along one shortest class-site path: alternating sum of cell means."""
    adj = {}
    for (c, s) in cells:
        adj.setdefault(("c", c), []).append(("s", s)); adj.setdefault(("s", s), []).append(("c", c))
    prev, q = {("c", u): None}, deque([("c", u)])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in prev:
                prev[y] = x; q.append(y)
    path, x = [], ("c", v)
    while x is not None:
        path.append(x); x = prev[x]
    path = path[::-1]
    w = {}
    for a, b in zip(path[:-1], path[1:]):                    # step class->site: +m ; site->class: -m
        if a[0] == "c": w[(a[1], b[1])] = w.get((a[1], b[1]), 0) + 1
        else: w[(b[1], a[1])] = w.get((b[1], a[1]), 0) - 1
    return w

rng = np.random.default_rng(0)
ratios = {"WLS (paper)": [], "Unweighted LS": [], "Shortest path": []}
designs = 0
while designs < 60:
    nC, nS = rng.integers(3, 7), rng.integers(3, 9)
    cells = {(c, s): int(rng.integers(10, 301)) for c in range(nC) for s in range(nS) if rng.random() < 0.45}
    L = laplacian(cells, nC, nS)
    if np.linalg.matrix_rank(L) != nC + nS - 1:              # keep connected designs only
        continue
    u, v = rng.choice(nC, 2, replace=False)
    e = np.zeros(nC + nS); e[u], e[v] = 1, -1
    R = float(e @ np.linalg.pinv(L) @ e)                     # bound: sigma^2 * R_eff with sigma = 1
    mu, delta = rng.normal(0, 2, nC), rng.normal(0, 3, nS)
    keys, w_wls = contrast_weights(cells, nC, nS, u, v, True)
    _, w_ols = contrast_weights(cells, nC, nS, u, v, False)
    wp = shortest_path_weights(cells, nC, nS, u, v)
    w_sp = np.array([wp.get(k, 0.0) for k in keys])
    est = {k: [] for k in ratios}
    truth = mu[u] - mu[v]
    for _ in range(3000):
        m = np.array([mu[c] + delta[s] + rng.normal(0, 1 / np.sqrt(cells[(c, s)])) for (c, s) in keys])
        est["WLS (paper)"].append(w_wls @ m); est["Unweighted LS"].append(w_ols @ m); est["Shortest path"].append(w_sp @ m)
    for k in ratios:
        a = np.array(est[k])
        assert abs(a.mean() - truth) < 6 * a.std() / np.sqrt(len(a)) + 1e-9, (k, a.mean(), truth)   # unbiased
        ratios[k].append(a.var(ddof=1) / R)
    designs += 1
print(f"{designs} random connected designs, 3,000 replicates each; ratio = empirical variance / bound sigma^2 R_eff")
for k, v in ratios.items():
    v = np.array(v)
    print(f"  {k:15s} mean ratio {v.mean():5.3f}  (min {v.min():5.3f}, max {v.max():6.3f})  "
          f"designs with ratio > 1.05: {int((v > 1.05).sum())}/{len(v)}")
