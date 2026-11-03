"""Snap VC3D surface meshes (tifxyz) onto the papyrus surface.

Usage:
  python tools/snap_tifxyz.py OUT_DIR PATCH_DIR [PATCH_DIR ...] [options]
  python tools/snap_tifxyz.py OUT_DIR --tifxyz SEGMENT.tifxyz --volume CROP.tif
         [--offset Z Y X] [--level-scale S] [options]

  PATCH_DIR is a folder made by tools/fetch_segment.py (segment.tifxyz, volume.tif, patch.json).
  With --tifxyz and --volume, CROP.tif must contain the whole mesh; --offset is where the crop
  starts in the volume (z, y, x) and --level-scale its resolution relative to the mesh's
  coordinates (1 = full resolution, 0.5 = OME-Zarr level 1).

Options:
  --target-offset T   put the mesh T voxels from the papyrus edge, measured along the normal
                      that points from papyrus toward the gap (0 = on the edge; negative =
                      inside the papyrus). Default: keep the mesh's own typical offset.
  --normals auto|grid|flipped
                      which way papyrus -> gap points at the recto face, relative to the grid's
                      own normal (cross product of the column and row directions). Default: from
                      the scroll geometry (the recto faces the scroll axis): from patch.json, or
                      the whole segment saved next to the patches, or --axis; else from the CT.
  --axis Y X          scroll axis position in the volume, for --tifxyz meshes

Writes, per input:
  OUT_DIR/<name>/snapped.tifxyz        the input grid with its vertices moved (drop-in replacement)
  OUT_DIR/<name>/snapped_dense.tifxyz  the snapped surface at about 3-voxel spacing
      Both carry extra channels: snap_shift (voxels moved along the normal) and
      snap_confidence (0-255). The dense one also has snap_flag (255 where the mesh sits on a
      face of the opposite polarity) and snap_limit (255 where the shift stopped at the edge of
      the search range, so the face may be farther away); the input-size one has snap_review
      (255 where either applies to most of the surface around the vertex). Those are the places
      worth a look.
  OUT_DIR/<name>/snap.json             summary
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params  # noqa: E402
from scroll_label_snap.mesh import axis_orientation, patch_orientation, snap_mesh  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def load_patch(d):
    pj = json.load(open(os.path.join(d, "patch.json")))
    return (os.path.join(d, "segment.tifxyz"), os.path.join(d, "volume.tif"),
            pj["offset_zyx"], pj.get("scale_to_level", 1.0))


def run_one(out_dir, name, seg, vol, offset, level_scale, normal_sign, p):
    t0 = time.time()
    P, valid, meta = read_tifxyz(seg)
    ct = tifffile.imread(vol)
    res = snap_mesh(ct, P, valid, p, offset_zyx=offset, level_scale=level_scale, normal_sign=normal_sign)
    f = res.factor
    scale = float(np.atleast_1d(meta.get("scale", [1.0]))[0])
    uuid = str(meta.get("uuid") or os.path.basename(os.path.normpath(seg))).replace(".tifxyz", "")
    src = dict(meta, snapped_from=uuid)
    conf_v = res.dense_confidence[::f, ::f]
    # a vertex is marked for review when most of the surface it carries is flagged
    review_v = ndi.uniform_filter((res.dense_flag | res.dense_limit).astype(np.float32), size=f)[::f, ::f] >= 0.5
    write_tifxyz(os.path.join(out_dir, name, "snapped.tifxyz"), res.P, res.valid, src,
                 channels=dict(snap_shift=res.vertex_shift.astype(np.float32),
                               snap_confidence=(conf_v * 255).astype(np.uint8),
                               snap_review=(review_v * 255).astype(np.uint8)),
                 uuid=f"{uuid}_snapped")
    write_tifxyz(os.path.join(out_dir, name, "snapped_dense.tifxyz"), res.dense_P, res.dense_valid, src,
                 scale=round(scale * f, 6), uuid=f"{uuid}_snapped_dense",
                 channels=dict(snap_shift=res.dense_shift.astype(np.float32),
                               snap_confidence=(res.dense_confidence * 255).astype(np.uint8),
                               snap_flag=(res.dense_flag * 255).astype(np.uint8),
                               snap_limit=(res.dense_limit * 255).astype(np.uint8)))
    info = {k: v for k, v in res.info.items() if k != "params"}
    vs = res.dense_shift[res.dense_valid]
    info.update(name=name, seconds=round(time.time() - t0, 1),
                dense_shift_percentiles={str(q): float(np.percentile(vs, q)) for q in (5, 25, 50, 75, 95)},
                mean_confidence=float(res.dense_confidence[res.dense_valid].mean()),
                params=p.asdict())
    json.dump(info, open(os.path.join(out_dir, name, "snap.json"), "w"), indent=1)
    print(f"{name}: {info['n_vertices']} vertices, {info['n_dense']} samples (x{f}), "
          f"edge at {info['convention_offset']:+.2f} voxels, mean |shift| {info['mean_abs_shift']:.2f}, "
          f"moved >= 1 voxel {info['frac_moved_1vox']:.0%}, flagged {info['frac_reversed_face']:.1%}, "
          f"at the search limit {info['frac_at_limit']:.1%}, "
          f"normals {'as the grid' if info['normal_sign'] > 0 else 'flipped'}"
          f"{'' if np.isnan(info['orientation_agree']) else ' (from the CT, %.0f%% agree)' % (100 * info['orientation_agree'])}, "
          f"{info['seconds']} s", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("patches", nargs="*")
    ap.add_argument("--tifxyz")
    ap.add_argument("--volume")
    ap.add_argument("--offset", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--level-scale", type=float, default=1.0)
    ap.add_argument("--target-offset", type=float, default=None)
    ap.add_argument("--normals", choices=("auto", "grid", "flipped"), default="auto")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a snapper parameter (see scroll_label_snap/core.py Params), e.g. t_max=6")
    ap.add_argument("--axis", type=float, nargs=2, metavar=("Y", "X"),
                    help="scroll axis position (volume y, x) for the orientation of a --tifxyz mesh")
    a = ap.parse_args()
    p = Params()
    for kv in a.param:
        k, v = kv.split("=", 1)
        if not hasattr(p, k):
            sys.exit(f"unknown parameter {k}")
        cur = getattr(p, k)
        p = replace(p, **{k: (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur or 0.0)(v)})
    if a.target_offset is not None:
        p = replace(p, convention_offset=a.target_offset)
    sign = {"auto": 0, "grid": 1, "flipped": -1}[a.normals]
    jobs = [(os.path.basename(os.path.normpath(d)),) + load_patch(d) + (sign or patch_orientation(d),)
            for d in a.patches]
    if a.tifxyz:
        if not a.volume:
            sys.exit("--tifxyz needs --volume")
        if not sign and a.axis:
            P, valid, _ = read_tifxyz(a.tifxyz)
            sign = axis_orientation(P, valid, tuple(a.axis))[0]
        jobs.append((os.path.basename(os.path.normpath(a.tifxyz)).replace(".tifxyz", ""), a.tifxyz, a.volume,
                     list(a.offset), a.level_scale, sign))
    if not jobs:
        sys.exit("nothing to do: give PATCH_DIRs or --tifxyz and --volume")
    os.makedirs(a.out_dir, exist_ok=True)
    infos = [run_one(a.out_dir, *job, p) for job in jobs]
    json.dump(infos, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
