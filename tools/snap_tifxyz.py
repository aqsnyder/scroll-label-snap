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
                      inside the papyrus). Default: keep the mesh's own typical offset. Each sample
                      still moves at most t_max voxels (--param), so a far target stops short:
                      see snap_limit.
  --normals auto|grid|flipped|ct
                      which way papyrus -> gap points at the recto face, relative to the grid's
                      own normal (cross product of the column and row directions). Default: from
                      the scroll geometry (the recto faces the scroll axis): from patch.json, or
                      the whole segment saved next to the patches, or --axis, or for --tifxyz a
                      circle fit to the mesh itself if it is long enough; a --tifxyz mesh that gives
                      none of these stops with a message, and a patch folder uses the CT with a
                      warning. ct: let the CT decide. That vote was wrong on 4 of 16 real patches,
                      and then every sample is pulled toward the wrong faces.
  --axis Y X          scroll axis position in the volume, for --tifxyz meshes
  --fine-grid V       also write snapped_fine.tifxyz, with vertices about V voxels apart (5 is what
                      has been tested; default 0 = don't write it)

Writes, per input:
  OUT_DIR/<name>/snapped.tifxyz        the input grid with its vertices moved. Same grid as the
                                       input, but about 20 voxels is too coarse to hold the snapped
                                       surface: it follows the dense one only in part
  OUT_DIR/<name>/snapped_fine.tifxyz   with --fine-grid: the same fit on a finer grid, 5 voxels (meta
                                       scale 0.2) for a 20-voxel input, which stays close to the dense
                                       surface. Its effect on reading ink has not been measured
  OUT_DIR/<name>/snapped_dense.tifxyz  the snapped surface at about 3-voxel spacing
      All carry extra channels: snap_shift (voxels moved along the normal) and
      snap_confidence (0-255). The dense and fine ones also have snap_flag (255 where the mesh sits
      on a face of the opposite polarity) and snap_limit (255 where the shift stopped at the edge of
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
from scroll_label_snap.core import Params, parse_params  # noqa: E402
from scroll_label_snap.mesh import (FIT_RULE, axis_fit_is_meaningful, axis_orientation, dense_at_fine, expand_paths,  # noqa: E402
                                    patch_orientation, refuse_label_only, snap_mesh, surface_area)
from scroll_label_snap.tifxyz import densified_scale, read_tifxyz, write_tifxyz  # noqa: E402


def load_patch(d):
    pj = json.load(open(os.path.join(d, "patch.json")))
    return (os.path.join(d, "segment.tifxyz"), os.path.join(d, "volume.tif"),
            pj["offset_zyx"], pj.get("scale_to_level", 1.0))


def run_one(out_dir, name, seg, vol, offset, level_scale, normal_sign, p, ct_ok=False, fine=0):
    t0 = time.time()
    P, valid, meta = read_tifxyz(seg)
    ct = tifffile.imread(vol)
    res = snap_mesh(ct, P, valid, p, offset_zyx=offset, level_scale=level_scale, normal_sign=normal_sign, fine=fine)
    f = res.factor
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
                 scale=densified_scale(meta, f), uuid=f"{uuid}_snapped_dense",
                 channels=dict(snap_shift=res.dense_shift.astype(np.float32),
                               snap_confidence=(res.dense_confidence * 255).astype(np.uint8),
                               snap_flag=(res.dense_flag * 255).astype(np.uint8),
                               snap_limit=(res.dense_limit * 255).astype(np.uint8)))
    if fine:
        g = res.fine_factor
        write_tifxyz(os.path.join(out_dir, name, "snapped_fine.tifxyz"), res.fine_P, res.fine_valid,
                     dict(src, area_vx2=surface_area(res.fine_P, res.fine_valid)),
                     scale=densified_scale(meta, g), uuid=f"{uuid}_snapped_fine",
                     channels=dict(snap_shift=res.fine_shift.astype(np.float32),
                                   snap_confidence=(dense_at_fine(res.dense_confidence, f, g) * 255).astype(np.uint8),
                                   snap_flag=(dense_at_fine(res.dense_flag, f, g) * 255).astype(np.uint8),
                                   snap_limit=(dense_at_fine(res.dense_limit, f, g) * 255).astype(np.uint8)))
    info = {k: v for k, v in res.info.items() if k != "params"}
    vs = res.dense_shift[res.dense_valid]
    info.update(name=name, seconds=round(time.time() - t0, 1),
                dense_shift_percentiles={str(q): float(np.percentile(vs, q)) for q in (5, 25, 50, 75, 95)} if len(vs) else {},
                mean_confidence=float(res.dense_confidence[res.dense_valid].mean()) if len(vs) else 0.0,
                fine_factor=res.fine_factor, params=p.asdict())
    json.dump(info, open(os.path.join(out_dir, name, "snap.json"), "w"), indent=1)
    print(f"{name}: {info['n_vertices']} vertices, {info['n_dense']} samples (x{f}), "
          f"edge at {info['convention_offset']:+.2f} voxels, mean |shift| {info['mean_abs_shift']:.2f}, "
          f"moved >= 1 voxel {info['frac_moved_1vox']:.0%}, flagged {info['frac_reversed_face']:.1%}, "
          f"at the search limit {info['frac_at_limit']:.1%}, "
          f"normals {'as the grid' if info['normal_sign'] > 0 else 'flipped'}"
          f"{'' if np.isnan(info['orientation_agree']) else ' (from the CT, %.0f%% agree)' % (100 * info['orientation_agree'])}, "
          f"{info['seconds']} s", flush=True)
    if normal_sign == 0 and not ct_ok:
        print(f"{name}: warning: nothing in the scroll's geometry settled the recto side, so the CT vote did; it was wrong "
              "on 4 of 16 real patches. Check it, or pass --normals grid or flipped", flush=True)
    if not info["n_movable"]:
        print(f"{name}: warning: no samples inside the volume crop, so the mesh is unchanged", flush=True)
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
    ap.add_argument("--normals", choices=("auto", "grid", "flipped", "ct"), default="auto")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a snapper parameter (see scroll_label_snap/core.py Params), e.g. t_max=6")
    ap.add_argument("--axis", type=float, nargs=2, metavar=("Y", "X"),
                    help="scroll axis position (volume y, x) for the orientation of a --tifxyz mesh")
    ap.add_argument("--fine-grid", type=float, default=0, metavar="V",
                    help="also write snapped_fine.tifxyz, with vertices about V voxels apart (default 0: don't)")
    a = ap.parse_args()
    overrides = parse_params(a.param)
    refuse_label_only(overrides)
    p = Params(**overrides)
    if a.target_offset is not None:
        if "convention_offset" in overrides or p.calibrate:
            sys.exit("--target-offset sets the offset, so it can't be combined with --param convention_offset or calibrate=true")
        p = replace(p, convention_offset=0.0 - a.target_offset)   # T is mesh minus edge; the offset is edge minus mesh
    sign = {"auto": 0, "ct": 0, "grid": 1, "flipped": -1}[a.normals]
    jobs = [(os.path.basename(os.path.normpath(d)),) + load_patch(d) + (sign or (0 if a.normals == "ct" else patch_orientation(d)),)
            for d in expand_paths(a.patches)]
    if a.tifxyz:
        if not a.volume:
            sys.exit("--tifxyz needs --volume")
        if a.normals == "auto":
            P, valid, _ = read_tifxyz(a.tifxyz)
            if a.axis:
                sign = axis_orientation(P, valid, tuple(a.axis))[0]
            else:
                sign, agree, axis = axis_orientation(P, valid)
                curved, radius, extent = axis_fit_is_meaningful(P, valid, axis)
                if agree < 0.9 or not curved:
                    sys.exit(f"this mesh's own curve does not settle which side is recto (agreement {agree:.0%}, fitted radius "
                             f"{radius:.0f} voxels for a mesh {extent:.0f} voxels across; {FIT_RULE}); pass --axis Y X, or --normals grid "
                             "or flipped, or --normals ct to let the CT vote decide")
        jobs.append((os.path.basename(os.path.normpath(a.tifxyz)).replace(".tifxyz", ""), a.tifxyz, a.volume,
                     list(a.offset), a.level_scale, sign))
    if not jobs:
        sys.exit("nothing to do: give PATCH_DIRs or --tifxyz and --volume")
    os.makedirs(a.out_dir, exist_ok=True)
    infos = [run_one(a.out_dir, *job, p, ct_ok=a.normals == "ct", fine=a.fine_grid) for job in jobs]
    json.dump(infos, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
