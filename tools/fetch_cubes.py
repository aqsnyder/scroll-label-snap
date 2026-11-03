"""Download a reproducible random set of image/label cube pairs from the Vesuvius Challenge
surface-label bucket on Hugging Face (needs huggingface_hub 2.4 or newer).

Usage:
  python tools/fetch_cubes.py OUT_DIR [--count 100] [--seed 2026] [--exclude DIR ...] [--all]
                              [--bucket scrollprize/datasets] [--prefix surfaces/kaggle]

Lists images/ and labels/ under the prefix and keeps names present in both. Drops names found
in any --exclude folder (for example the cubes used during development), then picks --count
of the rest at random with --seed, or all of them with --all. Writes OUT_DIR/selection.json
and downloads the pairs into OUT_DIR/images and OUT_DIR/labels. Files already there with the
right size are skipped, so an interrupted download can simply be run again.
"""
import argparse
import json
import os
import sys

import numpy as np


def log(msg, path):
    print(msg, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--exclude", nargs="*", default=[], help="folders whose .tif names are left out")
    ap.add_argument("--all", action="store_true", help="take every pair that isn't excluded")
    ap.add_argument("--bucket", default="scrollprize/datasets")
    ap.add_argument("--prefix", default="surfaces/kaggle")
    ap.add_argument("--batch", type=int, default=20, help="files per download call")
    a = ap.parse_args()
    try:
        from huggingface_hub import download_bucket_files, list_bucket_tree
    except ImportError:
        sys.exit("needs huggingface_hub 2.4 or newer: pip install --upgrade huggingface_hub")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    logf = os.path.join(a.out_dir, "fetch.log")

    files = {}
    for sub in ("images", "labels"):
        items = list_bucket_tree(a.bucket, prefix=f"{a.prefix}/{sub}", recursive=True)
        files[sub] = {os.path.basename(it.path)[:-4]: it for it in items
                      if getattr(it, "type", "") == "file" and it.path.endswith(".tif")}
    names = sorted(set(files["images"]) & set(files["labels"]))
    excluded = set()
    for d in a.exclude:
        if os.path.isdir(d):
            excluded |= {f[:-4] for f in os.listdir(d) if f.endswith(".tif")}
    pool = [x for x in names if x not in excluded]
    if a.all:
        pick = pool
    else:
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.choice(pool, min(a.count, len(pool)), replace=False).tolist())
    with open(os.path.join(a.out_dir, "selection.json"), "w") as f:
        json.dump(dict(bucket=a.bucket, prefix=a.prefix, seed=a.seed, all=a.all, listed_pairs=len(names),
                       excluded=sorted(excluded), count=len(pick), names=pick), f, indent=1)

    todo = []
    for x in pick:
        for sub in ("images", "labels"):
            it = files[sub][x]
            local = os.path.join(a.out_dir, sub, x + ".tif")
            if not (os.path.exists(local) and os.path.getsize(local) == it.size):
                todo.append((it, local))
    log(f"{len(names)} pairs in the bucket, {len(excluded & set(names))} excluded, {len(pick)} picked "
        f"(seed {a.seed}); {len(todo)} files to download, "
        f"{sum(it.size for it, _ in todo) / 1e9:.1f} GB", logf)
    for k in range(0, len(todo), a.batch):
        download_bucket_files(a.bucket, files=todo[k:k + a.batch], raise_on_missing_files=True)
        log(f"downloaded {min(k + a.batch, len(todo))} of {len(todo)} files", logf)
    bad = [local for it, local in todo if not (os.path.exists(local) and os.path.getsize(local) == it.size)]
    log("all files present with the expected sizes" if not bad else f"{len(bad)} files missing or wrong size: {bad[:5]}", logf)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
