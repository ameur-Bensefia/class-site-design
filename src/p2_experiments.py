"""Frozen ResNet-18 features, learning curve, allocation validation, cycle-test calibration and external source.

  python p2_experiments.py r18       frozen ImageNet ResNet-18 features of the same 224x224 radiographs used end to end,
                                     run through the frozen-feature pipeline (natural pool, 10 replicates; constructed
                                     confounding, 5 replicates, same splits as e2e_cxr.py): same backbone, frozen vs fine-tuned.
  python p2_experiments.py curve     end-to-end ERM on the natural pool for 10 epochs: validation and test AUC per epoch.
  python p2_experiments.py alloc     (a) greedy allocation versus the continuous convex optimum (certificate of the gap);
                                     (b) reservoir validation: samples actually drawn from a held-out reservoir by each
                                     strategy, and the empirical error of the disease contrast compared with the prediction.
  python p2_experiments.py cyclecal  cycle test calibrated by a residual bootstrap that resamples patients within sites
                                     and re-estimates the whole procedure; effect size independent of sample size.
  python p2_experiments.py external  HAM10000 to an unseen acquisition source (ISIC 2018 test, source 'external').
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

DIR = DATASETS_DIR
MODE = sys.argv[1] if len(sys.argv) > 1 else "alloc"
if len(sys.argv) > 2:
    DIR = sys.argv[2].rstrip("/")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_p1 import combat_apply, combat_fit, group_split, load_cxr, load_npz, lr, r_eff, sqrtm  # noqa: E402

PAPER_E2E = {"natural": {"ERM": (89.7, 5.3), "ComBat": (89.6, 5.2), "Anchor-referenced CORAL": (90.1, 4.8)},
             "constructed": {"ERM": (59.2, 58.6, 62.7, 72.2), "ComBat": (None, 67.3, 72.5, 79.2),
                             "Anchor-referenced CORAL": (None, 62.6, 68.1, 76.4)}}


def heads(Xtr, ytr, Str, Xte, Ste):
    """Scores of four frozen-feature methods (anchors: class 0)."""
    out = {"ERM": lr().fit(Xtr, ytr).decision_function(Xte)}
    sites = np.unique(Str)
    if all(((Str == s) & (ytr == 0)).sum() > 5 for s in sites) and len(sites) > 1:
        f = combat_fit(Xtr, ytr, Str)
        out["ComBat"] = lr().fit(f["data"].T, ytr).decision_function(combat_apply(Xte, Ste, f["estimates"]))
        ref_mu, ref_C = Xtr[ytr == 0].mean(0), LedoitWolf().fit(Xtr[ytr == 0]).covariance_
        st = {s: (Xtr[(Str == s) & (ytr == 0)].mean(0), LedoitWolf().fit(Xtr[(Str == s) & (ytr == 0)]).covariance_) for s in sites}

        def coral(A, Sa):
            o = np.empty_like(A)
            for s in np.unique(Sa):
                m = Sa == s
                o[m] = (A[m] - st[s][0]) @ sqrtm(st[s][1], -0.5) @ sqrtm(ref_C, 0.5) + ref_mu
            return o
        out["Anchor-referenced CORAL"] = lr().fit(coral(Xtr, Str), ytr).decision_function(coral(Xte, Ste))
    return out


# ------------------------------------------------------------------ same backbone: frozen ResNet-18
def run_r18():
    import torch
    import torchvision
    _, Y, S, G = load_cxr()
    feat_fn = "cxr_r18_frozen.npy"
    if not os.path.exists(feat_fn):
        imgs = np.load("cxr224.npy", mmap_mode="r")
        assert len(imgs) == len(Y)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m.eval().to(dev)
        F = np.zeros((len(Y), 512), np.float32)
        with torch.no_grad():
            for k in range(0, len(Y), 256):
                x = torch.from_numpy(np.ascontiguousarray(imgs[k:k + 256])).float().div_(255.0)
                x = ((x - 0.449) / 0.226).unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(dev)
                F[k:k + 256] = m(x).cpu().numpy()
                if k % 10240 == 0:
                    print(f"  features {k}/{len(Y)}", flush=True)
        np.save(feat_fn, F)
    X0 = np.load(feat_fn)
    nat = {}
    for rep in range(10):                                  # natural pool, seeds as in natural_pool_cxr.py
        rng = np.random.default_rng(300 + rep)
        tr, te = group_split(G, rng)
        X = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
        for k, sc in heads(X[tr], Y[tr], S[tr], X[te], S[te]).items():
            w = np.mean([roc_auc_score(Y[te][S[te] == s], sc[S[te] == s]) for s in ("NIH", "CheXpert")])
            nat.setdefault(k, []).append((w, roc_auc_score(Y[te], sc) - w))
    con = {}
    for rep in range(5):                                   # constructed confounding, identical to semireal_cxr / e2e_cxr
        for b in (0, 100, 1000, -1):
            rng = np.random.default_rng(200 + rep)
            tr, te = group_split(G, rng)
            keep = tr & (((Y == 1) & (S == "NIH")) | ((Y == 0) & (S == "CheXpert")))
            nf = np.where(tr & (Y == 0) & (S == "NIH"))[0]
            nb = len(nf) if b < 0 else min(b, len(nf))
            keep[rng.choice(nf, nb, replace=False)] = True
            test = te & (S == "CheXpert")
            X = PCA(128, random_state=rep).fit(X0[tr][rng.choice(tr.sum(), min(8000, tr.sum()), replace=False)]).transform(X0)
            for k, sc in heads(X[keep], Y[keep], S[keep], X[test], S[test]).items():
                con.setdefault((k, b), []).append(roc_auc_score(Y[test], sc))
    print("\nSame backbone (ResNet-18): frozen ImageNet features vs end-to-end fine-tuning (paper, Tables 9-10)")
    print(f"{'method':26s} {'natural: within / gap (frozen)':>32s} {'(end to end)':>14s}")
    for k, v in nat.items():
        v = np.array(v)
        ref = PAPER_E2E["natural"].get(k)
        print(f"{k:26s} {100 * v[:, 0].mean():20.1f} / {100 * v[:, 1].mean():+5.1f}       "
              + (f"{ref[0]:.1f} / {ref[1]:+.1f}" if ref else ""))
    print(f"\n{'method':26s} {'constructed, frozen ResNet-18 (bridge 0 / 100 / 1000 / all)':>60s}   end to end")
    for k in ("ERM", "ComBat", "Anchor-referenced CORAL"):
        cells = [f"{100 * np.mean(con[(k, b)]):5.1f}" if (k, b) in con else "  n.e." for b in (0, 100, 1000, -1)]
        ref = PAPER_E2E["constructed"][k]
        print(f"{k:26s} {' / '.join(cells):>60s}   " + " / ".join("n.e." if r is None else f"{r:.1f}" for r in ref))
    from scipy.stats import t as tdist
    print("\nPaired comparison on frozen ResNet-18, anchor-referenced CORAL minus ComBat (corrected resampled t):")
    for b in (100, 1000, -1):
        d = 100 * (np.array(con[("Anchor-referenced CORAL", b)]) - np.array(con[("ComBat", b)]))
        se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
        p = 2 * tdist.sf(abs(d.mean()) / se, len(d) - 1)
        print(f"  constructed, bridge {'all' if b < 0 else b}: {d.mean():+5.1f} [{d.mean() - tdist.ppf(0.975, len(d) - 1) * se:+5.1f}, "
              f"{d.mean() + tdist.ppf(0.975, len(d) - 1) * se:+5.1f}] p={p:.2g}, wins {int((d > 0).sum())}/{len(d)}")
    for k in ("ERM", "Anchor-referenced CORAL"):
        dw = 100 * (np.array([v[0] for v in nat[k]]) - np.array([v[0] for v in nat["ComBat"]]))
        dg = 100 * (np.array([v[1] for v in nat[k]]) - np.array([v[1] for v in nat["ComBat"]]))
        for name, d in (("within-hospital AUC", dw), ("gap", dg)):
            se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 0.3 / 0.7))
            print(f"  natural, {k} minus ComBat, {name}: {d.mean():+5.2f} (p={2 * tdist.sf(abs(d.mean()) / se, len(d) - 1):.2g})")


# ------------------------------------------------------------------ convergence curve
def run_curve():
    import torch
    import torch.nn.functional as Fn
    import torchvision
    _, Y, S, G = load_cxr()
    imgs = np.load("cxr224.npy", mmap_mode="r")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(200)
    tr, te = group_split(G, rng)
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    pats = np.unique(G[tr_idx])
    val_p = set(np.random.default_rng(5).choice(pats, int(0.1 * len(pats)), replace=False))
    val_mask = np.array([g in val_p for g in G[tr_idx]])
    fit_idx, val_idx = tr_idx[~val_mask], tr_idx[val_mask]

    def tens(idx):
        x = torch.from_numpy(np.ascontiguousarray(imgs[idx])).float().div_(255.0)
        return ((x - 0.449) / 0.226).unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(dev)
    torch.manual_seed(7)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(512, 2)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    cw = torch.tensor(len(fit_idx) / (2 * np.bincount(Y[fit_idx])), dtype=torch.float32, device=dev)

    @torch.no_grad()
    def score(idx):
        net.eval()
        out = np.concatenate([(lambda l: (l[:, 1] - l[:, 0]).cpu().numpy())(net(tens(idx[k:k + 256]))) for k in range(0, len(idx), 256)])
        net.train()
        return out
    print(f"{'epoch':>5s} {'validation AUC (pooled)':>24s} {'test pooled':>12s} {'test within (mean)':>19s}")
    r = np.random.default_rng(11)
    for ep in range(1, 11):
        net.train()
        perm = r.permutation(fit_idx)
        for k in range(0, len(perm) - 63, 64):
            bi = perm[k:k + 64]
            loss = Fn.cross_entropy(net(tens(bi)), torch.from_numpy(Y[bi]).long().to(dev), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        sv, st_ = score(val_idx), score(te_idx)
        w = np.mean([roc_auc_score(Y[te_idx][S[te_idx] == s], st_[S[te_idx] == s]) for s in ("NIH", "CheXpert")])
        print(f"{ep:5d} {100 * roc_auc_score(Y[val_idx], sv):24.1f} {100 * roc_auc_score(Y[te_idx], st_):12.1f} {100 * w:19.1f}", flush=True)


# ------------------------------------------------------------------ allocation: certificate and reservoir validation
def laplacian(counts, sites):
    idx = {("c", 0): 0, ("c", 1): 1, **{("s", s): 2 + i for i, s in enumerate(sites)}}
    L = np.zeros((len(idx), len(idx)))
    for (c, s), n in counts.items():
        if n > 0:
            i, j = idx[("c", c)], idx[("s", s)]
            L[i, i] += n; L[j, j] += n; L[i, j] -= n; L[j, i] -= n
    return L, idx


def greedy(counts, budget, cells, caps=None, steps=200):
    cur = dict(counts)
    inc, spent = max(1, budget // steps), 0
    while spent < budget:
        step = min(inc, budget - spent)
        best, best_r = None, np.inf
        for e in cells:
            room = np.inf if caps is None else caps[e] - (cur.get(e, 0) - counts.get(e, 0))
            if room <= 0:
                continue
            s_ = int(min(step, room))
            trial = dict(cur); trial[e] = trial.get(e, 0) + s_
            rr = r_eff(trial)
            if rr < best_r:
                best, best_r, best_s = e, rr, s_
        if best is None:
            break
        cur[best] = cur.get(best, 0) + best_s
        spent += best_s
    return cur


def convex_opt(counts, budget, cells, iters=3000):
    """Continuous relaxation: minimize R_eff(n + x) over x >= 0, sum x = budget, by exponentiated gradient on the
    scaled simplex using the exact gradient -(potential drop)^2 (Proposition 5). Gives a lower bound for any integer
    allocation of the same budget."""
    sites = sorted({s for _, s in counts})
    w = np.ones(len(cells)) / len(cells)
    best = np.inf
    for t in range(iters):
        x = budget * w
        cur = dict(counts)
        for e, xe in zip(cells, x):
            cur[e] = cur.get(e, 0) + xe
        L, idx = laplacian(cur, sites)
        e_ = np.zeros(len(idx)); e_[0], e_[1] = 1, -1
        phi = np.linalg.pinv(L) @ e_
        R = float(e_ @ phi)
        best = min(best, R)
        g = np.array([-(phi[idx[("c", c)]] - phi[idx[("s", s)]]) ** 2 for (c, s) in cells]) * budget
        w = w * np.exp(-0.5 * g / (np.abs(g).max() + 1e-12))
        w = w / w.sum()
    return best


def run_alloc():
    ham = load_npz("ham10000")
    cx = load_cxr()
    designs = [("NIH + CheXpert", cx), ("HAM10000", ham), ("PTB-XL", load_npz("ptbxl")), ("ABIDE I+II", load_npz("abide"))]
    print("(a) Greedy allocation versus the continuous convex optimum (lower bound), budget 10,000")
    for name, (_, Y, S, _) in designs:
        counts = {(c, s): int(((Y == c) & (S == s)).sum()) for s in np.unique(S) for c in (0, 1) if ((Y == c) & (S == s)).any()}
        cells = sorted({(c, s) for s in np.unique(S) for c in (0, 1)})
        rg = r_eff(greedy(counts, 10000, cells))
        rc = convex_opt(counts, 10000, cells)
        print(f"  {name:16s} greedy 1/R = {1 / rg:9.1f} | convex optimum 1/R = {1 / rc:9.1f} | greedy within {100 * (rg - rc) / rc:.2f}% of the optimum")
    print("\n(b) Reservoir validation: initial design = 50% of the data, reservoir = the rest; each strategy draws the same")
    print("    number of samples from the reservoir (cells capped by availability); error of the site-adjusted disease")
    print("    contrast (squared distance to the full-data estimate, 16 principal components), 40 replicates;")
    print("    for nested estimators the prediction is proportional to R_eff(subset) - R_eff(full)")
    for name, (X0, Y, S, G), B in (("HAM10000", ham, 600), ("NIH + CheXpert", cx, 2000)):
        Z = PCA(16, random_state=0).fit_transform((X0 - X0.mean(0)) / (X0.std(0) + 1e-8))
        sites = sorted(np.unique(S))
        cells = [(c, s) for s in sites for c in (0, 1)]

        def contrast(idx):
            cnt = {(c, s): int(((Y[idx] == c) & (S[idx] == s)).sum()) for (c, s) in cells}
            keys = [e for e in cells if cnt[e] > 0]
            nodes = [("c", 0), ("c", 1)] + [("s", s) for s in sites]
            ix = {x: i for i, x in enumerate(nodes)}
            Bm = np.zeros((len(keys), len(nodes)))
            for k, (c, s) in enumerate(keys):
                Bm[k, ix[("c", c)]] = 1; Bm[k, ix[("s", s)]] = 1
            M = np.array([Z[idx][(Y[idx] == c) & (S[idx] == s)].mean(0) for (c, s) in keys])
            W = np.sqrt(np.array([cnt[e] for e in keys]))[:, None]
            th, *_ = np.linalg.lstsq(Bm * W, M * W, rcond=None)
            return th[1] - th[0]
        full = contrast(np.arange(len(Y)))
        r_full = r_eff({(c, s): int(((Y == c) & (S == s)).sum()) for (c, s) in cells})
        errs = {k: [] for k in ("proportional", "uniform", "smallest cell", "greedy (graph)")}
        pred = {k: [] for k in errs}
        for rep in range(40):
            rng = np.random.default_rng(900 + rep)
            ug = np.unique(G)
            init_g = set(rng.choice(ug, len(ug) // 2, replace=False))
            init = np.array([g in init_g for g in G])
            base = {(c, s): int(((Y == c) & (S == s) & init).sum()) for (c, s) in cells}
            avail = {(c, s): np.where((Y == c) & (S == s) & ~init)[0] for (c, s) in cells}
            caps = {e: len(v) for e, v in avail.items()}
            tot = sum(base.values())
            plans = {"proportional": {e: min(caps[e], int(round(B * base[e] / tot))) for e in cells},
                     "uniform": {e: min(caps[e], B // len(cells)) for e in cells},
                     "smallest cell": {e: (min(caps[e], B) if e == min(cells, key=lambda k: base[k] if caps[k] > 0 else 1e9) else 0) for e in cells}}
            g = greedy(base, B, cells, caps=caps)
            plans["greedy (graph)"] = {e: g[e] - base[e] for e in cells}
            for k, plan in plans.items():
                add = np.concatenate([rng.choice(avail[e], plan[e], replace=False) for e in cells if plan[e] > 0] or [np.array([], int)])
                idx = np.concatenate([np.where(init)[0], add.astype(int)])
                errs[k].append(float(np.sum((contrast(idx) - full) ** 2)))
                # nested estimators: E||est(subset) - est(full)||^2 = tr(Sigma) * (R_subset - R_full)
                pred[k].append(r_eff({e: base[e] + plan[e] for e in cells}) - r_full)
        print(f"  {name} (budget {B}):")
        ref = np.mean(errs["proportional"])
        for k in errs:
            print(f"    {k:16s} empirical error {np.mean(errs[k]):.4f} (relative to proportional {np.mean(errs[k]) / ref:.2f}) | "
                  f"predicted (R_subset - R_full) relative {np.mean(pred[k]) / np.mean(pred['proportional']):.2f}")


# ------------------------------------------------------------------ cycle test with a structure-preserving bootstrap
def cycle_T(Z, Y, S, sites):
    cells, means, L, ns = [], [], [], []
    for s in sites:
        res = np.vstack([Z[(S == s) & (Y == g)] - Z[(S == s) & (Y == g)].mean(0) for g in (0, 1)])
        w, V = np.linalg.eigh(LedoitWolf().fit(res).covariance_)
        Wd = V @ np.diag(w ** -0.5) @ V.T
        for g in (0, 1):
            m = (S == s) & (Y == g)
            cells.append((g, s)); means.append(Z[m].mean(0)); L.append(np.sqrt(m.sum()) * Wd); ns.append(m.sum())
    k = Z.shape[1]
    nodes = [("c", 0), ("c", 1)] + [("s", s) for s in sites]
    ix = {x: i for i, x in enumerate(nodes)}
    A = np.zeros((len(cells) * k, len(nodes) * k)); b = np.zeros(len(cells) * k)
    for e, ((g, s), m, Le) in enumerate(zip(cells, means, L)):
        r = slice(e * k, (e + 1) * k)
        A[r, ix[("c", g)] * k:(ix[("c", g)] + 1) * k] = Le
        A[r, ix[("s", s)] * k:(ix[("s", s)] + 1) * k] = Le
        b[r] = Le @ m
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    fitted = {}
    for e, (g, s) in enumerate(cells):
        fitted[(g, s)] = th[ix[("c", g)] * k:(ix[("c", g)] + 1) * k] + th[ix[("s", s)] * k:(ix[("s", s)] + 1) * k]
    return float(np.sum((b - A @ th) ** 2)), k * (len(sites) - 1), fitted, int(sum(ns))


def run_cyclecal(B=200):
    sets = [("NIH + CheXpert", load_cxr()), ("HAM10000", load_npz("ham10000")), ("PTB-XL", load_npz("ptbxl")),
            ("ABIDE I+II", load_npz("abide"))]
    print(f"{'data':16s} {'k':>3s} {'T':>9s} {'dof':>5s} {'chi2 p':>10s} {'bootstrap null: mean T*, 95th pct':>34s} "
          f"{'bootstrap p':>12s} {'RMS standardized interaction':>29s}")
    for name, (X0, Y, S, G) in sets:
        sites = [s for s in np.unique(S) if all(((S == s) & (Y == g)).sum() >= 10 for g in (0, 1))]
        m = np.isin(S, sites)
        X0, Y, S, G = X0[m], Y[m], S[m], G[m]
        pcs = PCA(20, random_state=0).fit_transform((X0 - X0.mean(0)) / (X0.std(0) + 1e-8))
        for k in (5, 10):
            Z = pcs[:, :k]
            T, dof, fitted, N = cycle_T(Z, Y, S, sites)
            rng = np.random.default_rng(3)
            # residuals around each cell mean, grouped by patient within site
            res = np.empty_like(Z)
            for s in sites:
                for g in (0, 1):
                    mm = (S == s) & (Y == g)
                    res[mm] = Z[mm] - Z[mm].mean(0)
            by_site = {}                                          # patient -> row indices, per site (precomputed)
            for s in sites:
                site_idx = np.where(S == s)[0]
                order = np.argsort(G[site_idx], kind="stable")
                gs, starts = np.unique(G[site_idx][order], return_index=True)
                by_site[s] = np.split(site_idx[order], starts[1:])
            cells_idx = {(g, s): np.where((S == s) & (Y == g))[0] for s in sites for g in (0, 1)}
            Ts = []
            for _ in range(B):
                Zs = np.empty_like(Z)
                for s in sites:
                    pats = by_site[s]
                    mean_size = max(1.0, np.mean([len(p) for p in pats]))
                    for g in (0, 1):
                        cell = cells_idx[(g, s)]
                        rows = np.array([], dtype=int)
                        while len(rows) < len(cell):              # resample whole patients from the same site
                            n_draw = int((len(cell) - len(rows)) / mean_size) + 1
                            picks = rng.integers(0, len(pats), n_draw)
                            rows = np.concatenate([rows] + [pats[i] for i in picks])
                        Zs[cell] = fitted[(g, s)] + res[rows[:len(cell)]]
                Ts.append(cycle_T(Zs, Y, S, sites)[0])
            Ts = np.array(Ts)
            lam = max(T - Ts.mean(), 0.0)                       # noncentrality above the bootstrap null mean
            print(f"{name:16s} {k:3d} {T:9.1f} {dof:5d} {chi2.sf(T, dof):10.2g} {Ts.mean():15.1f}, {np.percentile(Ts, 95):8.1f}"
                  f"        {(1 + (Ts >= T).sum()) / (B + 1):12.3f} {np.sqrt(lam / N):29.3f}", flush=True)
    print("\nBootstrap p-values cannot be smaller than 1/(B+1); the RMS standardized interaction sqrt(lambda/N) is an effect "
          "size that does not grow with sample size (lambda: statistic minus its bootstrap null mean).")


# ------------------------------------------------------------------ external site (ISIC 2018 'external' source)
def run_external(reps=20):
    X0, Y, S, G = load_npz("ham10000")
    z = np.load(f"{DIR}/isic2018_test.npz", allow_pickle=True)
    Xe, Ye, Se = z["X"], z["y"].astype(int), z["site"].astype(str)
    pca = PCA(128, random_state=0).fit(X0)
    Xtr, Xt = pca.transform(X0), pca.transform(Xe)
    print(f"ISIC 2018 test: {dict(pd.Series(Se).value_counts())}; external source: "
          f"{int(((Se == 'external') & (Ye == 0)).sum())} nevi, {int(((Se == 'external') & (Ye == 1)).sum())} melanomas")
    erm = lr().fit(Xtr, Y)
    ext = Se == "external"

    def boot_ci(y, s, nb=1000):
        r = np.random.default_rng(0); v = []
        for _ in range(nb):
            i = r.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                v.append(roc_auc_score(y[i], s[i]))
        return np.percentile(v, [2.5, 97.5])
    s_erm = erm.decision_function(Xt[ext])
    lo, hi = boot_ci(Ye[ext], s_erm)
    print(f"Standard training, unseen source: AUC {100 * roc_auc_score(Ye[ext], s_erm):.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
    for s in sorted(set(Se) - {"external"}):
        mm = Se == s
        if len(np.unique(Ye[mm])) == 2:
            print(f"  for comparison, ISIC test images from training source {s}: AUC {100 * roc_auc_score(Ye[mm], erm.decision_function(Xt[mm])):.1f} (n={mm.sum()})")
    # new-site adaptation with anchors: half of the external nevi are used as anchors and excluded from evaluation
    res = {"Standard training (same evaluation images)": [], "Anchor centring with external anchors": [],
           "Anchor-referenced CORAL with external anchors": []}
    ref_mu = Xtr[Y == 0].mean(0)
    ref_C = LedoitWolf().fit(Xtr[Y == 0]).covariance_
    ex_idx = np.where(ext)[0]
    for rep in range(reps):
        r = np.random.default_rng(rep)
        nev = ex_idx[Ye[ex_idx] == 0]
        anchors = r.choice(nev, len(nev) // 2, replace=False)
        ev = np.setdiff1d(ex_idx, anchors)
        mu_a, C_a = Xt[anchors].mean(0), LedoitWolf().fit(Xt[anchors]).covariance_
        res["Standard training (same evaluation images)"].append(roc_auc_score(Ye[ev], erm.decision_function(Xt[ev])))
        res["Anchor centring with external anchors"].append(roc_auc_score(Ye[ev], erm.decision_function(Xt[ev] - mu_a + ref_mu)))
        res["Anchor-referenced CORAL with external anchors"].append(
            roc_auc_score(Ye[ev], erm.decision_function((Xt[ev] - mu_a) @ sqrtm(C_a, -0.5) @ sqrtm(ref_C, 0.5) + ref_mu)))
    print(f"\nNew-site adaptation, {reps} random anchor/evaluation splits of the external source:")
    for k, v in res.items():
        print(f"  {k:46s} AUC {100 * np.mean(v):.1f} (range {100 * np.min(v):.1f}-{100 * np.max(v):.1f})")
    print("ComBat cannot be applied to a source absent from training (no site parameters are estimated for it).")


if __name__ == "__main__":
    {"r18": run_r18, "curve": run_curve, "alloc": run_alloc, "cyclecal": run_cyclecal, "external": run_external}[MODE]()
