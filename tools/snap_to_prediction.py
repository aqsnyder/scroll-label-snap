"""Baseline for mesh snapping: snap fetched patches onto a surface prediction, with the method
Erwin Nieuwlaar used for his PHerc. 343 First Letters meshes (scroll_label_snap/baselines.py).

Usage:
  python tools/snap_to_prediction.py OUT_DIR PATCH_DIR [PATCH_DIR ...] [--prediction m7]
         [--max-off 15] [--jump 6] [--sigma 1]

  PATCH_DIR is a folder made by tools/fetch_segment.py with the prediction cropped next to the
  scan (--extra m7=URL gives patch_K/m7.tif).

Writes the same layout as tools/snap_tifxyz.py, so the evaluation and figure tools can compare
the two: OUT_DIR/<name>/snapped.tifxyz (the input grid, vertices moved), snapped_dense.tifxyz (the
same mesh resampled like the snapper's dense output) and snap.json. The recto side comes from
the scroll geometry as for the snapper, and the mesh's own offset from the papyrus face is
estimated from the CT, so a surface placed at the prediction's centre is measured at its own
face.
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.baselines import snap_to_prediction  # noqa: E402
from scroll_label_snap.core import Params, estimate_convention  # noqa: E402
from scroll_label_snap.mesh import (densify, mean_spacing, orientation_vote, patch_orientation,  # noqa: E402
                                    vertex_normals)
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--prediction", default="m7")
    ap.add_argument("--max-off", type=int, default=15)
    ap.add_argument("--jump", type=float, default=6.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    a = ap.parse_args()
    p = Params()
    infos = []
    for d in a.patches:
        name = os.path.basename(os.path.normpath(d))
        pj = json.load(open(os.path.join(d, "patch.json")))
        off = np.asarray(pj["offset_zyx"], np.float32)
        ls = float(pj.get("scale_to_level", 1.0))
        pred = tifffile.imread(os.path.join(d, f"{a.prediction}.tif"))
        ct = tifffile.imread(os.path.join(d, "volume.tif"))
        P, valid, meta = read_tifxyz(os.path.join(d, "segment.tifxyz"))
        Q = P.astype(np.float32) * ls - off
        S, offv, info = snap_to_prediction(pred, Q, valid, a.max_off, a.jump, a.sigma)
        N, ok = vertex_normals(S, valid)
        f = max(1, int(round(mean_spacing(S, ok) / p.spacing)))
        Sd, Nd, vd = densify(S, N, ok, f)
        ct_s = ndi.gaussian_filter(ct.astype(np.float32), p.sigma_ct)
        sign = patch_orientation(d, verbose=False) or orientation_vote(ct_s, Sd[vd], Nd[vd])[0]
        c_off, _ = estimate_convention(ct_s, Sd[vd], Nd[vd] * sign, p)
        out = (S + off) / ls
        dense = (Sd + off) / ls
        uuid = str(meta.get("uuid") or name).replace(".tifxyz", "")
        src = dict(meta, snapped_from=uuid, snapped_with=f"snap_to_prediction ({a.prediction})")
        scale = float(np.atleast_1d(meta.get("scale", [1.0]))[0])
        os.makedirs(os.path.join(a.out_dir, name), exist_ok=True)
        write_tifxyz(os.path.join(a.out_dir, name, "snapped.tifxyz"), out, ok, src, uuid=f"{uuid}_{a.prediction}")
        write_tifxyz(os.path.join(a.out_dir, name, "snapped_dense.tifxyz"), dense, vd, src,
                     scale=round(scale * f, 6), uuid=f"{uuid}_{a.prediction}_dense")
        info.update(name=name, prediction=a.prediction, convention_offset=float(c_off), normal_sign=int(sign),
                    factor=f, max_off=a.max_off, jump=a.jump, sigma=a.sigma,
                    prediction_fraction=float((pred > 0).mean()))
        json.dump(info, open(os.path.join(a.out_dir, name, "snap.json"), "w"), indent=1)
        infos.append(info)
        print(f"{name}: sheet within reach at {info['frac_with_sheet']:.0%} of vertices, snapped {info['frac_snapped']:.0%}, "
              f"median |offset| {info['median_abs_offset']:.2f}, rejected {info['frac_rejected']:.0%}; "
              f"face at {c_off:+.2f} voxels from the result", flush=True)
    json.dump(infos, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
