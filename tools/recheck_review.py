"""Where does a new snapper version put the labels at the points of a finished review?

For each moved item, measures the label band's position along the normal (as in
eval_snap.py) for the original, the reviewed snapped version and a new version, and
reports how many wins stay where the reviewed version put them and how many losses go
back to the original.

Usage:
  python tools/recheck_review.py KEY_JSON ANSWERS_JSON IMAGES_DIR LABELS_DIR OLD_DIR NEW_DIR [OUT_JSON]
"""
import json
import os
import sys

import numpy as np
import tifffile
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from scroll_label_snap.core import Params, prepare  # noqa: E402
from eval_snap import band_position  # noqa: E402
from score_review import verdict  # noqa: E402


def main(key_path, answers_path, images, labels, old_dir, new_dir, out_json=None):
    key = json.load(open(key_path))
    answers = json.load(open(answers_path))
    moved = {q: r for q, r in key.items() if r["stratum"] in ("big move", "moderate move") and q in answers}
    rows = []
    for cube in sorted({r["cube"] for r in moved.values()}):
        img = tifffile.imread(os.path.join(images, cube + ".tif"))
        lab = tifffile.imread(os.path.join(labels, cube + ".tif"))
        _, pts, n, _, _, _ = prepare(img, lab, Params())
        qs = [q for q in sorted(moved) if moved[q]["cube"] == cube]
        ks = [cKDTree(pts).query(np.array(moved[q]["point_zyx"]))[1] for q in qs]
        pos = [band_position(v, pts[ks], n[ks]) for v in
               (lab, tifffile.imread(os.path.join(old_dir, cube + ".tif")), tifffile.imread(os.path.join(new_dir, cube + ".tif")))]
        for m, q in enumerate(qs):
            rows.append(dict(item=q, cube=cube, verdict=verdict(moved[q], answers[q]),
                             old=round(float(pos[1][m] - pos[0][m]), 2), new=round(float(pos[2][m] - pos[0][m]), 2)))

    def near(a, b):
        return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 0.5

    wins = [r for r in rows if r["verdict"] == "snapped"]
    losses = [r for r in rows if r["verdict"] == "original"]
    print(f"wins (reviewed version preferred): {sum(near(r['new'], r['old']) for r in wins)} of {len(wins)} stay within "
          f"0.5 voxel; {sum(near(r['new'], 0) for r in wins)} go back to the original")
    print(f"losses (original preferred): {sum(near(r['new'], 0) for r in losses)} of {len(losses)} go back to the original; "
          f"{sum(near(r['new'], r['old']) for r in losses)} stay")
    for r in losses:
        print(f"   {r['item']} {r['cube']}: {r['old']:+.2f} -> {r['new']:+.2f}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:8])
