"""Snap the recto-surface labels of one or more cubes.

Usage:
  python tools/snap_cube.py OUT_DIR IMAGE_TIF LABEL_TIF [IMAGE_TIF LABEL_TIF ...] [--param NAME=VALUE ...]

--param overrides a snapper parameter (see scroll_label_snap/core.py Params), e.g. keep_topology=true.

Writes, per cube:
  OUT_DIR/labels/<name>.tif      snapped labels (same values as the input: 0/1/2)
  OUT_DIR/confidence/<name>.tif  per-voxel confidence of the snapped band (0-255)
  OUT_DIR/flags/<name>.tif       255 where the label sits on a face of the opposite polarity
  OUT_DIR/<name>.json            summary (samples, shift statistics, parameters)

A cube that fails is reported and the rest still run; the exit code is non-zero if any failed.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params, snap, shift_volumes  # noqa: E402


def main(out_dir, *paths, p=None):
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "confidence"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "flags"), exist_ok=True)
    p = p or Params()
    failed = []
    for img_path, lab_path in zip(paths[0::2], paths[1::2]):
        name = os.path.basename(img_path).rsplit(".", 1)[0]
        t0 = time.time()
        try:
            img = tifffile.imread(img_path)
            lab = tifffile.imread(lab_path)
            res = snap(img, lab, p)
        except Exception as e:  # keep going; report at the end
            failed.append(name)
            print(f"{name}: FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        # 255 in flags on band voxels whose label sits on a face of the opposite polarity: worth a look
        conf, flags = shift_volumes(lab.shape, res.labels == 1, res.pts,
                                    [(res.confidence * 255).astype(np.uint8), (res.reversed_face * 255).astype(np.uint8)])
        tifffile.imwrite(os.path.join(out_dir, "labels", name + ".tif"), res.labels,
                         compression="zlib")
        tifffile.imwrite(os.path.join(out_dir, "confidence", name + ".tif"), conf,
                         compression="zlib")
        tifffile.imwrite(os.path.join(out_dir, "flags", name + ".tif"), flags, compression="zlib")
        info = dict(res.info, seconds=round(time.time() - t0, 1))
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"{name}: {info['n_samples']} samples, {info['n_sheets']} sheets, "
              f"mean |shift| {info['mean_abs_shift']:.2f} vox, moved >=1 vox "
              f"{info['frac_moved_1vox']:.0%}, flagged {info['frac_reversed_face']:.0%}, {info['seconds']} s")
        if not info["n_movable"]:
            print(f"{name}: no samples away from the cube's edge, so the labels are unchanged", flush=True)
    if failed:
        sys.exit(f"{len(failed)} cubes failed: {', '.join(failed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("paths", nargs="+", metavar="TIF", help="image and label TIFFs, in pairs")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a snapper parameter (see scroll_label_snap/core.py Params), e.g. keep_topology=true")
    a = ap.parse_args()
    params = Params()
    for kv in a.param:
        k, v = kv.split("=", 1)
        if not hasattr(params, k):
            sys.exit(f"unknown parameter {k}")
        cur = getattr(params, k)
        params = replace(params, **{k: (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur or 0.0)(v)})
    main(a.out_dir, *a.paths, p=params)
