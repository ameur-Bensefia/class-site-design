"""Paths used by the analysis scripts. Set them with environment variables rather than editing the scripts:

    export CSD_DATA_ROOT=/path/to/raw/datasets          # folder with the original releases (NIH, CheXpert, ...)
    export CSD_DATASETS_DIR=/path/to/prepared/datasets  # folder with the .npz files written by prepare_datasets.py
"""
import os

DATA_ROOT = os.environ.get("CSD_DATA_ROOT", os.path.expanduser("~/data/class-site-design"))
DATASETS_DIR = os.environ.get("CSD_DATASETS_DIR", os.path.join(DATA_ROOT, "prepared"))
CAMELYON_METADATA = os.environ.get("CSD_CAMELYON_METADATA", os.path.join(DATA_ROOT, "WILDS", "camelyon17_v1.0", "metadata.csv"))
