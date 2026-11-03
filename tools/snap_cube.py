"""Snap the recto-surface labels of one or more cubes.

Usage:
  python tools/snap_cube.py OUT_DIR IMAGE_TIF LABEL_TIF [IMAGE_TIF LABEL_TIF ...]

Writes, per cube:
  OUT_DIR/labels/<name>.tif      snapped labels (same values as the input: 0/1/2)
  OUT_DIR/confidence/<name>.tif  per-voxel confidence of the snapped band (0-255)
  OUT_DIR/flags/<name>.tif       255 where the label sits on a face of the opposite polarity
  OUT_DIR/<name>.json            summary (samples, shift statistics, parameters)
"""
import json
import os
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params, snap, shift_volume  # noqa: E402


def main(out_dir, *paths):
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "confidence"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "flags"), exist_ok=True)
    p = Params()
    for img_path, lab_path in zip(paths[0::2], paths[1::2]):
        name = os.path.basename(img_path).rsplit(".", 1)[0]
        t0 = time.time()
        img = tifffile.imread(img_path)
        lab = tifffile.imread(lab_path)
        res = snap(img, lab, p)
        conf = shift_volume(lab.shape, res.labels == 1, res.pts,
                            (res.confidence * 255).astype(np.uint8))
        tifffile.imwrite(os.path.join(out_dir, "labels", name + ".tif"), res.labels,
                         compression="zlib")
        tifffile.imwrite(os.path.join(out_dir, "confidence", name + ".tif"), conf,
                         compression="zlib")
        # 255 on band voxels whose label sits on a face of the opposite polarity: worth a look
        flags = shift_volume(lab.shape, res.labels == 1, res.pts, (res.reversed_face * 255).astype(np.uint8))
        tifffile.imwrite(os.path.join(out_dir, "flags", name + ".tif"), flags, compression="zlib")
        info = dict(res.info, seconds=round(time.time() - t0, 1))
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"{name}: {info['n_samples']} samples, {info['n_sheets']} sheets, "
              f"mean |shift| {info['mean_abs_shift']:.2f} vox, moved >=1 vox "
              f"{info['frac_moved_1vox']:.0%}, flagged {info['frac_reversed_face']:.0%}, {info['seconds']} s")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
