"""Snap a whole VC3D segment (tifxyz) onto the papyrus surface, tile by tile, reading the scan as
it goes.

Usage:
  python tools/snap_segment.py OUT_DIR --segment TIFXYZ --volume OME_ZARR [options]

  TIFXYZ    a tifxyz folder, or the https URL of one (e.g. in the Vesuvius Challenge open-data bucket)
  OME_ZARR  the scan the segment was traced on: a local path or an https URL (OME-Zarr, zarr v2)

Options:
  --tile 24 --overlap 4   tiles of 24 x 24 vertices, overlapping neighbours by 4 vertices on each
                          side and blended there, so there are no seams
  --workers 4             tiles processed in parallel (each needs about 1-2 GB of memory)
  --cache DIR             keep the downloaded scan chunks in DIR, so reruns and neighbouring tiles
                          don't download them again (recommended; a whole segment can need 10+ GB)
  --target-offset T       where the mesh should sit relative to the recto face (as in
                          snap_tifxyz.py). Default: one offset for the whole segment, the median
                          of the segment's own offsets in --sample-tiles tiles (12), so the result
                          sits at one consistent depth.
  --normals auto|grid|flipped   recto side; default from the segment's curvature
  --param NAME=VALUE      override a snapper parameter
  --limit N               only the first N tiles (for a quick try)
  --name NAME             name for the outputs (default: the segment folder's name)

Writes:
  OUT_DIR/<name>_snapped.tifxyz        the input grid, vertices moved (drop-in replacement)
  OUT_DIR/<name>_snapped_dense.tifxyz  the snapped surface at ~3-voxel spacing
      with channels snap_shift, snap_confidence and (dense) snap_flag / snap_limit or
      (input grid) snap_review, as from snap_tifxyz.py
  OUT_DIR/snap_segment.json            summary
  OUT_DIR/tiles/                       one file per finished tile; a rerun skips them
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params, estimate_convention  # noqa: E402
from scroll_label_snap.mesh import (axis_fit_is_meaningful, axis_orientation, densify,  # noqa: E402
                                    fit_vertex_shifts, mean_spacing, prepare_mesh, snap_mesh, vertex_normals)
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def open_volume(url, level, cache=None):
    """A resolution level of an OME-Zarr (or the array itself if it has no levels), local or
    over http(s); with `cache`, downloaded chunks are kept on disk."""
    import fsspec
    import zarr
    base = url.rstrip("/")
    opts = {}
    if cache and base.startswith(("http://", "https://")):
        base = "filecache::" + base
        opts = dict(filecache=dict(cache_storage=cache))
    for path in (f"{base}/{level}", base):
        try:
            return zarr.open_array(fsspec.get_mapper(path, **opts), mode="r")
        except Exception:
            continue
    raise FileNotFoundError(f"no zarr array at {url} (level {level})")


def tile_windows(valid, tile, overlap):
    """(r0, r1, c0, c1, core) windows: core tiles of tile - 2 * overlap vertices tile the grid,
    each window adds `overlap` vertices around its core."""
    H, W = valid.shape
    core = max(1, tile - 2 * overlap)
    out = []
    for r in range(0, H, core):
        for c in range(0, W, core):
            r0, r1 = max(0, r - overlap), min(H, r + core + overlap)
            c0, c1 = max(0, c - overlap), min(W, c + core + overlap)
            if valid[r:min(H, r + core), c:min(W, c + core)].sum() >= 4:
                out.append((r0, r1, c0, c1, (r, min(H, r + core), c, min(W, c + core))))
    return out


def read_crop(vol, P, valid, level_scale, margin):
    Q = P[valid].astype(np.float64) * level_scale
    lo = np.maximum(np.floor(Q.min(0)).astype(int) - margin, 0)
    hi = np.minimum(np.ceil(Q.max(0)).astype(int) + margin + 1, np.array(vol.shape[-3:]))
    return np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]), lo


_STATE = {}


def _init(volume, level, cache, seg_dir):
    _STATE["vol"] = open_volume(volume, level, cache)
    _STATE["seg"] = read_tifxyz(seg_dir)


def tile_convention(win, p, level_scale, margin, factor, sign):
    """The tile's own offset of the recto face from the mesh."""
    P, valid, _ = _STATE["seg"]
    r0, r1, c0, c1, _ = win
    Pw, vw = P[r0:r1, c0:c1], valid[r0:r1, c0:c1]
    ct, lo = read_crop(_STATE["vol"], Pw, vw, level_scale, margin)
    prep = prepare_mesh(ct, Pw, vw, p, lo, level_scale, factor=factor, normal_sign=sign)
    c_off, _ = estimate_convention(prep.ct_s, prep.Pd[prep.vd], prep.Nd[prep.vd], p)
    return float(c_off)


def run_tile(win, p, level_scale, margin, factor, sign, out_path):
    P, valid, _ = _STATE["seg"]
    r0, r1, c0, c1, _ = win
    t0 = time.time()
    Pw, vw = P[r0:r1, c0:c1], valid[r0:r1, c0:c1]
    ct, lo = read_crop(_STATE["vol"], Pw, vw, level_scale, margin)
    t1 = time.time()
    shape = ((r1 - r0 - 1) * factor + 1, (c1 - c0 - 1) * factor + 1)
    info = dict(window=[r0, r1, c0, c1], crop=list(ct.shape), read_s=round(t1 - t0, 1))
    try:
        res = snap_mesh(ct, Pw, vw, p, offset_zyx=lo, level_scale=level_scale, factor=factor, normal_sign=sign)
        out = dict(shift=res.dense_shift.astype(np.float32), conf=res.dense_confidence.astype(np.float32),
                   flag=res.dense_flag, limit=res.dense_limit, valid=res.dense_valid)
        info.update(mean_abs_shift=res.info["mean_abs_shift"], frac_at_limit=res.info["frac_at_limit"],
                    frac_flagged=res.info["frac_reversed_face"])
    except Exception as e:                      # e.g. a ragged edge tile with too few samples: leave it as it is
        out = dict(shift=np.zeros(shape, np.float32), conf=np.zeros(shape, np.float32), flag=np.zeros(shape, bool),
                   limit=np.zeros(shape, bool), valid=np.zeros(shape, bool))
        info.update(error=f"{type(e).__name__}: {e}")
    np.savez_compressed(out_path + ".tmp.npz", window=np.array([r0, r1, c0, c1]), **out)
    os.replace(out_path + ".tmp.npz", out_path)
    info["snap_s"] = round(time.time() - t1, 1)
    return info


def blend_weights(n, ov_lo, ov_hi, f):
    """1D weights over a tile window's dense samples: 1 in the core, ramping down across the
    overlap on each side that borders another tile."""
    w = np.ones(n, np.float32)
    k_lo, k_hi = ov_lo * f, ov_hi * f
    if k_lo:
        w[:k_lo + 1] = np.linspace(1.0 / (k_lo + 1), 1.0, k_lo + 1)
    if k_hi:
        w[n - k_hi - 1:] = np.linspace(1.0, 1.0 / (k_hi + 1), k_hi + 1)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--tile", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache")
    ap.add_argument("--target-offset", type=float)
    ap.add_argument("--sample-tiles", type=int, default=12)
    ap.add_argument("--normals", choices=("auto", "grid", "flipped"), default="auto")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", help="name for the outputs (default: the segment folder's name)")
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(os.path.join(a.out_dir, "tiles"), exist_ok=True)

    seg_dir = a.segment
    if a.segment.startswith(("http://", "https://")):
        from fetch_segment import download_tifxyz
        seg_dir = os.path.join(a.out_dir, "segment.tifxyz")
        download_tifxyz(a.segment, seg_dir)
    P, valid, meta = read_tifxyz(seg_dir)
    name = a.name or os.path.basename(os.path.normpath(a.segment.rstrip("/"))).replace(".tifxyz", "")
    level_scale = 1.0 / 2 ** a.level

    p = Params()
    for kv in a.param:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        p = replace(p, **{k: (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur or 0.0)(v)})

    N, ok = vertex_normals(P.astype(np.float32), valid)
    spacing = mean_spacing(P.astype(np.float32), ok) * level_scale
    factor = max(1, int(round(spacing / p.spacing)))
    if a.normals == "auto":
        sign, agree, axis = axis_orientation(P, valid)
        print(f"{name}: grid {P.shape[0]} x {P.shape[1]}, {valid.mean():.0%} valid, vertices {spacing:.1f} voxels apart; "
              f"scroll axis near y {axis[0]:.0f}, x {axis[1]:.0f}, recto {'along' if sign > 0 else 'against'} the grid "
              f"normal at {agree:.0%} of vertices", flush=True)
        curved, radius, extent = axis_fit_is_meaningful(P, valid, axis)
        if agree < 0.9 or not curved:
            sys.exit(f"the segment's curvature does not settle which side is recto (agreement {agree:.0%}, fitted radius "
                     f"{radius:.0f} voxels for a segment {extent:.0f} voxels across); pass --normals grid or flipped")
    else:
        sign = 1 if a.normals == "grid" else -1

    wins = tile_windows(ok, a.tile, a.overlap)
    if a.limit:
        wins = wins[:a.limit]
    print(f"{len(wins)} tiles of up to {a.tile} x {a.tile} vertices; dense factor {factor}", flush=True)
    ctx = dict(initializer=_init, initargs=(a.volume, a.level, a.cache, seg_dir))

    # one convention for the whole segment: the median of the segment's own offsets in sample tiles
    c_note = "given"
    if a.target_offset is not None:
        c_off = a.target_offset
    else:
        conv_path = os.path.join(a.out_dir, "convention.json")
        if os.path.exists(conv_path):
            conv = json.load(open(conv_path))
        else:
            rng = np.random.default_rng(0)
            pick = sorted(rng.choice(len(wins), min(a.sample_tiles, len(wins)), replace=False).tolist())
            with ProcessPoolExecutor(max_workers=a.workers, **ctx) as ex:
                offs = list(ex.map(tile_convention, [wins[k] for k in pick], *zip(*[(p, level_scale, a.margin, factor, sign)] * len(pick))))
            conv = dict(tiles=pick, offsets=offs, median=float(np.median(offs)))
            json.dump(conv, open(conv_path, "w"), indent=1)
        c_off = conv["median"]
        c_note = f"median of {len(conv['offsets'])} tiles (range {min(conv['offsets']):+.2f} to {max(conv['offsets']):+.2f})"
    p = replace(p, convention_offset=float(c_off))
    print(f"recto face at {c_off:+.2f} voxels from the mesh ({c_note})", flush=True)
    if a.target_offset is None and abs(c_off) >= p.convention_reach - 0.25:
        print("warning: the face sits at the edge of the search range; the recto side may be the other one "
              "(--normals), or the mesh is far off the surface", flush=True)

    # snap every tile
    todo = []
    for w in wins:
        path = os.path.join(a.out_dir, "tiles", f"r{w[0]:05d}_c{w[2]:05d}.npz")
        if not os.path.exists(path):
            todo.append((w, path))
    print(f"{len(wins) - len(todo)} tiles already done, {len(todo)} to go", flush=True)
    logs = []
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers, **ctx) as ex:
            futs = {ex.submit(run_tile, w, p, level_scale, a.margin, factor, sign, path): w for w, path in todo}
            for k, fu in enumerate(as_completed(futs), 1):
                info = fu.result()
                logs.append(info)
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t_start
                    print(f"  {k}/{len(todo)} tiles, {el / 60:.1f} min; last: crop {info['crop']}, read {info['read_s']} s, "
                          f"snap {info['snap_s']} s", flush=True)

    # blend the tiles into one dense shift field
    H, W = valid.shape
    Hd, Wd = (H - 1) * factor + 1, (W - 1) * factor + 1
    acc = np.zeros((Hd, Wd), np.float64)
    wsum = np.zeros((Hd, Wd), np.float64)
    conf = np.zeros((Hd, Wd), np.float64)
    flag = np.zeros((Hd, Wd), np.float64)
    limit = np.zeros((Hd, Wd), np.float64)
    for w in wins:
        path = os.path.join(a.out_dir, "tiles", f"r{w[0]:05d}_c{w[2]:05d}.npz")
        t = np.load(path)
        r0, r1, c0, c1 = t["window"]
        cr0, cr1, cc0, cc1 = w[4]
        wr = blend_weights(t["shift"].shape[0], cr0 - r0, r1 - cr1, factor)
        wc = blend_weights(t["shift"].shape[1], cc0 - c0, c1 - cc1, factor)
        wt = (wr[:, None] * wc[None, :]) * t["valid"]
        sl = (slice(r0 * factor, r0 * factor + t["shift"].shape[0]), slice(c0 * factor, c0 * factor + t["shift"].shape[1]))
        acc[sl] += wt * t["shift"]
        conf[sl] += wt * t["conf"]
        flag[sl] += wt * t["flag"]
        limit[sl] += wt * t["limit"]
        wsum[sl] += wt
    have = wsum > 0
    shift_d = np.where(have, acc / np.maximum(wsum, 1e-12), 0).astype(np.float32)
    conf_d = np.where(have, conf / np.maximum(wsum, 1e-12), 0).astype(np.float32)
    flag_d = have & (flag / np.maximum(wsum, 1e-12) >= 0.5)
    limit_d = have & (limit / np.maximum(wsum, 1e-12) >= 0.5)

    # outputs: dense surface and the input grid fitted to it (least squares)
    Nn = N * sign                                      # shifts are in volume voxels along these normals
    Pd, Nd, vd = densify(P.astype(np.float32), Nn, ok, factor)
    vd &= have
    dense = Pd + shift_d[..., None] * Nd
    dense[~vd] = -1
    v_shift = fit_vertex_shifts(ok, factor, vd, shift_d)
    Pout = P.astype(np.float32) + v_shift[..., None] * Nn
    Pout[~ok] = -1
    scale = float(np.atleast_1d(meta.get("scale", [1.0]))[0])
    src = dict(meta, snapped_from=name, snap_convention_offset=float(c_off))
    review_v = ndi.uniform_filter((flag_d | limit_d).astype(np.float32), size=factor)[::factor, ::factor] >= 0.5
    write_tifxyz(os.path.join(a.out_dir, f"{name}_snapped.tifxyz"), Pout, ok, src, uuid=f"{name}_snapped",
                 channels=dict(snap_shift=v_shift.astype(np.float32),
                               snap_confidence=(conf_d[::factor, ::factor] * 255).astype(np.uint8),
                               snap_review=(review_v * 255).astype(np.uint8)))
    write_tifxyz(os.path.join(a.out_dir, f"{name}_snapped_dense.tifxyz"), dense, vd, src, scale=round(scale * factor, 6),
                 uuid=f"{name}_snapped_dense",
                 channels=dict(snap_shift=shift_d, snap_confidence=(conf_d * 255).astype(np.uint8),
                               snap_flag=(flag_d * 255).astype(np.uint8), snap_limit=(limit_d * 255).astype(np.uint8)))
    s = shift_d[vd]
    summary = dict(segment=a.segment, volume=a.volume, level=a.level, name=name, grid=list(P.shape[:2]), factor=factor,
                   tiles=len(wins), convention_offset=float(c_off), convention_note=c_note, normal_sign=int(sign),
                   n_dense=int(vd.sum()), mean_abs_shift=float(np.abs(s).mean()), frac_moved_1vox=float(np.mean(np.abs(s) >= 1)),
                   frac_flagged=float(flag_d[vd].mean()), frac_at_limit=float(limit_d[vd].mean()),
                   frac_vertices_review=float(review_v[ok].mean()), minutes=round((time.time() - t_start) / 60, 1),
                   tiles_failed=[lg for lg in logs if "error" in lg],
                   params=p.asdict(), tile_logs=logs)
    json.dump(summary, open(os.path.join(a.out_dir, "snap_segment.json"), "w"), indent=1)
    print(f"done: {summary['n_dense']:,} samples, mean |shift| {summary['mean_abs_shift']:.2f}, moved >= 1 voxel "
          f"{summary['frac_moved_1vox']:.0%}, review {summary['frac_vertices_review']:.0%} of vertices, "
          f"{summary['minutes']} min -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
