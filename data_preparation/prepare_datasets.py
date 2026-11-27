"""Prepare the datasets used in the experiments: select the cohorts, extract the representations and save one .npz
file per dataset (X, y, group, sample_id, site, paths), which the analysis scripts in src/ read.

Usage:
    python prepare_datasets.py --root /path/to/raw/datasets --out /path/to/prepared [--cache /path/to/cache] [names ...]

names: any of nih_cxr, chexpert, ham10000, ptbxl, abide (default: all). Image datasets are embedded with a frozen
ImageNet ResNet-50 (IMAGENET1K_V2; resize 256, center crop 224, ImageNet normalization); a GPU or Apple MPS device is
used when available. The inclusion rules are those described in the papers (Supplemental Table S1).
"""
import argparse
import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csdprep import adapters, io  # noqa: E402

DATASETS = ["nih_cxr", "chexpert", "ham10000", "ptbxl", "abide"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("names", nargs="*", default=DATASETS, help="datasets to prepare (default: all)")
ap.add_argument("--root", required=True, help="folder containing one sub-folder per raw dataset")
ap.add_argument("--out", required=True, help="output folder for the .npz files (CSD_DATASETS_DIR)")
ap.add_argument("--cache", default=None, help="cache folder for embeddings (default: <out>/cache)")
ap.add_argument("--max_images_cxr", type=int, default=20000, help="images per chest radiograph dataset (default 20,000)")
args = ap.parse_args()
cache = args.cache or os.path.join(args.out, "cache")
os.makedirs(args.out, exist_ok=True); os.makedirs(cache, exist_ok=True)

rows = []
for name in args.names:
    if name not in DATASETS:
        print(f"[skip] unknown dataset {name}; choose from {DATASETS}"); continue
    t = time.time()
    opts = dict(cache_dir=cache)
    if name in ("nih_cxr", "chexpert"):
        opts["max_images"] = args.max_images_cxr
    try:
        ds = adapters.load(name, args.root, **opts)
        ds.name = name
        io.save(ds, args.out)
        s = ds.summary(); s["seconds"] = round(time.time() - t); rows.append(s)
        print(f"[ok] {name}: {s}", flush=True)
    except Exception as e:  # report and continue with the next dataset
        print(f"[error] {name}: {e.__class__.__name__}: {e}", flush=True)
        traceback.print_exc(limit=1)
if rows:
    out = pd.DataFrame(rows); out.to_csv(os.path.join(args.out, "datasets_summary.csv"), index=False)
    print(out.to_string(index=False))
