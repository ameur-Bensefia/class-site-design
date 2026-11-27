"""Design check on the natural (as-distributed) class-site designs of the public datasets: connectivity, effective
resistance of the disease contrast between the two class nodes, effective sample size 1/R_eff, independent cycles."""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import numpy as np, pandas as pd, sys
DIR = sys.argv[1] if len(sys.argv) > 1 else DATASETS_DIR

def audit(name, counts):
    """counts: {(class, site): n}. Bipartite graph, conductance = n; class contrast between the two class nodes."""
    classes = sorted({c for c, _ in counts}); sites = sorted({s for _, s in counts})
    nodes = [("c", c) for c in classes] + [("s", s) for s in sites]; idx = {x: i for i, x in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for (c, s), n in counts.items():
        if n <= 0: continue
        i, j = idx[("c", c)], idx[("s", s)]; L[i, i] += n; L[j, j] += n; L[i, j] -= n; L[j, i] -= n
    e = np.zeros(len(nodes)); e[idx[("c", classes[0])]], e[idx[("c", classes[1])]] = 1, -1
    comps = len(nodes) - np.linalg.matrix_rank(L)
    edges = sum(1 for n in counts.values() if n > 0)
    cycles = edges - len(nodes) + comps
    lap = np.linalg.pinv(L); R = float(e @ lap @ e)
    # connectivity of the two class nodes: R finite and the Laplacian null space does not separate them
    w, V = np.linalg.eigh(L); null = V[:, w < 1e-9]
    connected = np.allclose(null[idx[("c", classes[0])]], null[idx[("c", classes[1])]])
    n_tot = sum(counts.values()); per = {c: sum(n for (cc, _), n in counts.items() if cc == c) for c in classes}
    R_ideal = 1 / per[classes[0]] + 1 / per[classes[1]]          # same class totals, no site effects to estimate
    return dict(dataset=name, sites=len(sites), samples=n_tot, connected=connected, cycles=int(cycles),
                R_eff=R if connected else float("inf"), n_eff=(1 / R if connected else 0.0),
                efficiency=(R_ideal / R if connected else 0.0))

rows = []
z = np.load(f"{DIR}/ham10000.npz", allow_pickle=True)
rows.append(audit("HAM10000 (melanoma vs nevus)", pd.Series(1, index=pd.MultiIndex.from_arrays([z["y"], z["site"]])).groupby(level=[0, 1]).sum().to_dict()))
cx = {}
for nm, site in (("nih_cxr", "NIH"), ("chexpert", "CheXpert")):
    z = np.load(f"{DIR}/{nm}.npz", allow_pickle=True)
    for c in (0, 1): cx[(c, site)] = int((z["y"] == c).sum())
rows.append(audit("NIH + CheXpert (effusion vs no finding)", cx))
z = np.load(f"{DIR}/ptbxl.npz", allow_pickle=True)
rows.append(audit("PTB-XL, all sites (infarction vs normal)", pd.Series(1, index=pd.MultiIndex.from_arrays([z["y"], z["site"].astype(str)])).groupby(level=[0, 1]).sum().to_dict()))
z = np.load(f"{DIR}/abide.npz", allow_pickle=True)
rows.append(audit("ABIDE I+II (autism vs control)", pd.Series(1, index=pd.MultiIndex.from_arrays([z["y"], z["site"].astype(str)])).groupby(level=[0, 1]).sum().to_dict()))
cam = {(c, h): n for h, n in zip(range(5), (29718, 17452, 42527, 64919, 73361)) for c in (0, 1)}   # WILDS v1.0, balanced
rows.append(audit("Camelyon17 WILDS (tumour vs normal)", cam))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}" if v < 1 else f"{v:.1f}"))
