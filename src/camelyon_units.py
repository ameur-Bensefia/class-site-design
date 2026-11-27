"""Design audit of Camelyon17 (WILDS v1.0, full dataset) at the level of patches, patients and slides, from the
metadata file only (no image features needed).
Usage: python camelyon_units.py [path/to/metadata.csv]"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_p1 import r_eff  # noqa: E402

DEF = CAMELYON_METADATA
path = sys.argv[1] if len(sys.argv) > 1 else DEF
m = pd.read_csv(path)
print(f"{path}: {len(m):,} patches; columns {list(m.columns)}")
y, site = m["tumor"].astype(int).values, m["center"].astype(int).values
patient = m["patient"].astype(str).values
slide = (m["patient"].astype(str) + "_" + m["node"].astype(str)).values
rows = []
for unit_name, unit in (("patches", np.arange(len(m)).astype(str)), ("slides", slide), ("patients", patient)):
    counts = {(c, s): len(np.unique(unit[(y == c) & (site == s)])) for s in np.unique(site) for c in (0, 1)}
    nA = sum(v for (c, _), v in counts.items() if c == 1)
    nB = sum(v for (c, _), v in counts.items() if c == 0)
    neff = 1 / r_eff(counts)
    rows.append(dict(unit=unit_name, total_units=len(np.unique(unit)),
                     tumour_units=nA, normal_units=nB, n_eff=round(neff, 1),
                     efficiency=round(neff / (nA * nB / (nA + nB)), 3)))
print(pd.DataFrame(rows).to_string(index=False))
print("\nPer hospital (patients with tumour patches / with normal patches):")
for s in np.unique(site):
    print(f"  hospital {s}: {len(np.unique(patient[(site == s) & (y == 1)]))} / {len(np.unique(patient[(site == s) & (y == 0)]))}")
print("\nA patient contributing both tumour and normal patches counts once in each cell; units are not independent across cells.")
