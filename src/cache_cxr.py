"""Resize the 40,001 NIH and CheXpert radiographs used in the experiments to 224x224 grayscale, once, and store them in
a memory-mapped file in the same order as the embeddings (NIH first, then CheXpert), so that end-to-end training reads
pixels quickly. Usage: python cache_cxr.py [datasets_dir] [output_dir]   (about 10-20 minutes, about 2 GB)"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

DIR = sys.argv[1] if len(sys.argv) > 1 else DATASETS_DIR
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SIZE = 224


def load(path):
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None


if __name__ == "__main__":
    paths = []
    for name in ("nih_cxr", "chexpert"):
        z = np.load(os.path.join(DIR, f"{name}.npz"), allow_pickle=True)
        paths += [str(p) for p in z["paths"]]
    n = len(paths)
    out_path = os.path.join(OUT, "cxr224.u8")
    mm = np.lib.format.open_memmap(out_path.replace(".u8", ".npy"), mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE))
    t0, failed = time.time(), []
    with Pool(max(2, (os.cpu_count() or 4) - 2)) as pool:
        for i, img in enumerate(pool.imap(load, paths, chunksize=64)):
            if img is None:
                failed.append(i)
            else:
                mm[i] = img
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{n} images, {rate:.0f}/s, about {(n - i - 1) / rate / 60:.0f} min left", flush=True)
    mm.flush()
    print(f"saved {out_path.replace('.u8', '.npy')}: {n} images, {len(failed)} unreadable"
          + (f" (first indices: {failed[:10]})" if failed else "") + f", {time.time() - t0:.0f} s")
