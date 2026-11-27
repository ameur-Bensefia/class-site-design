"""Feasibility study: can a classifier separate disease from site when the two are confounded?

Ground truth is known by construction: each sample has a class (C = control, A, B) and a site style.
Training sites are confounded (site 1 holds C and A, site 2 holds C and B). The decisive test is
counterfactual: A rendered with the style of site 2 and B with the style of site 1. A model that
recognizes disease classifies them correctly; a model that recognizes site gets them wrong.
"""
import json
import sys
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression

D, DC, DS, HID = 32, 8, 8, 64
CLASSES = ["C", "A", "B"]


# ------------------------------------------------------------------ data generator
class World:
    """Fixed generative mechanism; site style is shared across classes (the identifiability assumption)."""

    def __init__(self, rng, disease_effect=1.0, style_scale=2.5, nonlinear=0.6):
        self.Wc = rng.normal(0, 1, (DC, D)) / np.sqrt(DC)
        self.Ws = rng.normal(0, 1, (DS, D)) / np.sqrt(DS)
        self.B = rng.normal(0, 1, (D, D)) / np.sqrt(D)
        self.mu = {"C": np.zeros(DC), "A": np.zeros(DC), "B": np.zeros(DC)}
        self.mu["A"][:3] = disease_effect
        self.mu["B"][3:6] = disease_effect
        self.style_scale, self.nl = style_scale, nonlinear
        self.rng = rng

    def new_site(self):
        return self.rng.normal(0, self.style_scale, DS)

    def sample(self, cls, style, n):
        z = self.mu[cls] + self.rng.normal(0, 1, (n, DC))
        s = style + self.rng.normal(0, 0.2, (n, DS))
        h = z @ self.Wc + s @ self.Ws
        return h + self.nl * np.tanh(h @ self.B) + self.rng.normal(0, 0.3, (n, D))


def build(world, design, sites, n):
    """design: {site_index: [classes]} -> X, y, site."""
    X, y, s = [], [], []
    for d, cls_list in design.items():
        for c in cls_list:
            X.append(world.sample(c, sites[d], n[c] if isinstance(n, dict) else n))
            y += [CLASSES.index(c)] * len(X[-1])
            s += [d] * len(X[-1])
    return np.vstack(X), np.array(y), np.array(s)


# ------------------------------------------------------------------ identifiability criterion (graph)
def identifiable_pairs(design):
    """Class pairs connected in the bipartite class-site graph (union-find over classes sharing a site)."""
    parent = {c: c for c in CLASSES}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for cls_list in design.values():
        for a, b in combinations(cls_list, 2):
            parent[find(a)] = find(b)
    present = sorted({c for v in design.values() for c in v})
    return {f"{a}-{b}": find(a) == find(b) for a, b in combinations(present, 2)}


# ------------------------------------------------------------------ small network with gradient reversal (numpy)
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Net:
    """Encoder (tanh) -> class head; optional site adversary on a subset, with gradient reversal."""

    def __init__(self, rng, n_cls, n_site):
        g = lambda a, b: rng.normal(0, 1 / np.sqrt(a), (a, b))  # noqa: E731
        self.p = dict(W1=g(D, HID), b1=np.zeros(HID), W2=g(HID, n_cls), b2=np.zeros(n_cls),
                      Wa=g(HID, n_site), ba=np.zeros(n_site))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def hidden(self, X):
        return np.tanh(X @ self.p["W1"] + self.p["b1"])

    def logits(self, X):
        return self.hidden(X) @ self.p["W2"] + self.p["b2"]

    def predict(self, X):
        return self.logits(X).argmax(1)

    def step(self, X, y, s, adv_mask, lam, lr=3e-3, wd=1e-4):
        n = len(X)
        H = self.hidden(X)
        P = softmax(H @ self.p["W2"] + self.p["b2"])
        dZ = P.copy()
        dZ[np.arange(n), y] -= 1
        dZ /= n
        g = dict(W2=H.T @ dZ, b2=dZ.sum(0))
        dH = dZ @ self.p["W2"].T
        g["Wa"], g["ba"] = np.zeros_like(self.p["Wa"]), np.zeros_like(self.p["ba"])
        if lam > 0 and adv_mask.any():
            Ha, sa = H[adv_mask], s[adv_mask]
            Pa = softmax(Ha @ self.p["Wa"] + self.p["ba"])
            dA = Pa.copy()
            dA[np.arange(len(sa)), sa] -= 1
            dA /= len(sa)
            g["Wa"], g["ba"] = Ha.T @ dA, dA.sum(0)                 # adversary minimizes site loss
            dHa = np.zeros_like(H)
            dHa[adv_mask] = dA @ self.p["Wa"].T
            dH = dH - lam * dHa                                     # encoder maximizes it (reversal)
        dPre = dH * (1 - H ** 2)
        g["W1"], g["b1"] = X.T @ dPre, dPre.sum(0)
        self.t += 1
        for k in self.p:                                            # Adam
            gk = g[k] + (wd * self.p[k] if k.startswith("W") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def train_net(rng, X, y, s, adv="none", lam=1.0, iters=1500):
    net = Net(rng, 3, int(s.max()) + 1)
    mask = {"none": np.zeros(len(y), bool), "all": np.ones(len(y), bool), "anchors": y == 0}[adv]
    for it in range(iters):
        net.step(X, y, s, mask, lam * min(1.0, it / 300) if adv != "none" else 0.0)
    return net


# ------------------------------------------------------------------ harmonization transforms
def anchor_standardize(X, s, y, ref_stats=None, anchor_mask=None):
    """Control-anchored transform: per site, standardize by the mean and covariance of its anchors (controls),
    then map to the reference anchor distribution. Applied to every sample of the site (propagation)."""
    out = np.empty_like(X)
    am = (y == 0) if anchor_mask is None else anchor_mask
    stats = {}
    for d in np.unique(s):
        Xa = X[(s == d) & am]
        mu, C = Xa.mean(0), np.cov(Xa.T) + 1e-2 * np.eye(D)
        w, V = np.linalg.eigh(C)
        stats[d] = (mu, V @ np.diag(w ** -0.5) @ V.T)
    if ref_stats is None:
        Xa = X[am]
        mu_r, C_r = Xa.mean(0), np.cov(Xa.T) + 1e-2 * np.eye(D)
        w, V = np.linalg.eigh(C_r)
        ref_stats = (mu_r, V @ np.diag(w ** 0.5) @ V.T)
    for d, (mu, Wd) in stats.items():
        out[s == d] = (X[s == d] - mu) @ Wd @ ref_stats[1] + ref_stats[0]
    return out, ref_stats


def site_residualize(X, s):
    """Remove each site's mean computed on all its samples (harmonization without class covariates)."""
    out = X.copy()
    for d in np.unique(s):
        out[s == d] -= X[s == d].mean(0)
    return out


# ------------------------------------------------------------------ evaluation
def balanced_acc(y, p, classes=(0, 1, 2)):
    return float(np.mean([np.mean(p[y == c] == c) for c in classes if (y == c).any()]))


def ab_acc(y, p):
    m = (y == 1) | (y == 2)
    return float(np.mean(p[m] == y[m]))


def ab_pairwise(y, logits):
    """Disease discrimination: among A and B samples, is the A score above the B score for A (and vice versa)?
    Independent of how often the model answers 'control', which depends on class proportions."""
    m = (y == 1) | (y == 2)
    pred = np.where(logits[m, 1] > logits[m, 2], 1, 2)
    return float(np.mean([np.mean(pred[y[m] == c] == c) for c in (1, 2)]))


def run(seed, scenario, n_anchor_new=50, n_anchor_train=300):
    rng = np.random.default_rng(seed)
    w = World(rng)
    S = [w.new_site() for _ in range(3)]                    # sites 0, 1 for training; 2 is a new hospital
    if scenario == "bridged":
        design = {0: ["C", "A"], 1: ["C", "B"]}
    else:                                                   # no bridge: site 1 holds only B
        design = {0: ["C", "A"], 1: ["B"]}
    n = {"C": n_anchor_train, "A": 300, "B": 300}
    Xtr, ytr, str_ = build(w, design, S, n)
    # tests: factual (same confounded design), counterfactual (swapped styles), new site (all classes)
    Xf, yf, sf = build(w, design, S, 300)
    Xcf, ycf, scf = build(w, {1: ["A"], 0: ["B"]} if scenario == "bridged" else {1: ["A"], 0: ["B"]}, S, 300)
    Xc_cf, yc_cf, sc_cf = build(w, {0: ["C"], 1: ["C"]}, S, 300)
    Xcf, ycf, scf = np.vstack([Xcf, Xc_cf]), np.r_[ycf, yc_cf], np.r_[scf, sc_cf]
    Xn, yn, sn = build(w, {2: ["C", "A", "B"]}, S, 300)
    Xn_anchor, _, sn_anchor = build(w, {2: ["C"]}, S, n_anchor_new)   # a few controls from the new hospital
    # oracle: unconfounded training (every class in every site)
    Xo, yo, so = build(w, {0: ["C", "A", "B"], 1: ["C", "A", "B"]}, S, 300)

    res = {"identifiable": identifiable_pairs(design)}

    def score(name, logits, transform=lambda X, s, y: X):
        Lf, Lcf, Ln = (logits(transform(X_, s_, y_)) for X_, s_, y_ in ((Xf, sf, yf), (Xcf, scf, ycf), (Xn, sn, yn)))
        res[name] = dict(factual=balanced_acc(yf, Lf.argmax(1)),
                         counterfactual=balanced_acc(ycf, Lcf.argmax(1)),
                         cf_A_vs_B=ab_pairwise(ycf, Lcf),
                         new_site=balanced_acc(yn, Ln.argmax(1)),
                         new_site_A_vs_B=ab_pairwise(yn, Ln))

    lr = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    score("ERM (logistic)", lr.decision_function)
    erm = train_net(rng, Xtr, ytr, str_)
    score("ERM (network)", erm.logits)
    Xr = site_residualize(Xtr, str_)
    res_net = train_net(rng, Xr, ytr, str_)
    score("Site harmonization", res_net.logits, lambda X, s, y: site_residualize(X, s))
    dann = train_net(rng, Xtr, ytr, str_, adv="all")
    score("Site adversary (DANN)", dann.logits)
    anc = train_net(rng, Xtr, ytr, str_, adv="anchors")
    score("Anchor adversary", anc.logits)
    if scenario == "bridged":
        Xa_tr, ref = anchor_standardize(Xtr, str_, ytr)
        anz = train_net(rng, Xa_tr, ytr, str_, adv="anchors")

        def anchor_tf(X, s, y):
            # transform each site with its own anchors: training sites use training controls,
            # the new hospital uses its few labelled controls; test labels are never used
            out = np.empty_like(X)
            for d in np.unique(s):
                if d == 2:
                    ref_X, ref_s = Xn_anchor, sn_anchor
                else:
                    ref_X, ref_s = Xtr[(str_ == d) & (ytr == 0)], np.full(((str_ == d) & (ytr == 0)).sum(), d)
                Xa_d = np.vstack([ref_X, X[s == d]])
                sa_d = np.r_[ref_s, s[s == d]]
                mask = np.r_[np.ones(len(ref_X), bool), np.zeros((s == d).sum(), bool)]
                tr, _ = anchor_standardize(Xa_d, sa_d, np.zeros(len(sa_d), int), ref_stats=ref, anchor_mask=mask)
                out[s == d] = tr[len(ref_X):]
            return out
        score("Anchor transform + adversary (proposed)", anz.logits, anchor_tf)
    ora = train_net(rng, Xo, yo, so)
    score("Oracle (unconfounded training)", ora.logits)
    return res


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {}
    for scenario in ("bridged", "no_bridge"):
        out[scenario] = [run(1000 + r, scenario) for r in range(reps)]
        print(scenario, "done", flush=True)
    json.dump(out, open("feasibility_results.json", "w"), indent=1)
    for scenario, runs in out.items():
        print(f"\n=== {scenario}: identifiable pairs {runs[0]['identifiable']}")
        print(f"{'method':42s} {'factual':>9s} {'counterf.':>9s} {'CF A-vs-B':>9s} {'new site':>9s} {'new A-B':>9s}")
        for m in [k for k in runs[0] if k != "identifiable"]:
            vals = {k: np.array([r[m][k] for r in runs]) for k in ("factual", "counterfactual", "cf_A_vs_B", "new_site", "new_site_A_vs_B")}
            print(f"{m:42s} " + " ".join(f"{100 * v.mean():6.1f}±{100 * 1.96 * v.std(ddof=1) / np.sqrt(len(v)):3.1f}"
                                          for v in vals.values()))
