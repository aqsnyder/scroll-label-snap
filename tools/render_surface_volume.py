"""Render a surface volume from a tifxyz mesh: the scan sampled on layers along the mesh's normals,
one pixel per voxel on the mesh's own grid, in the layout the ink models read: (layers, rows,
columns), uint8, 0 outside the mesh.

A mesh and its snapped version (snap_segment.py or snap_tifxyz.py, which keep the vertex grid)
render onto the same pixel grid, so ink predictions from the two line up pixel for pixel. So does
snapped_fine, from --fine-grid 5 (scale 0.2, 5 pixels per vertex, 4 vertices per input vertex; --rows
and --cols count its own vertices). snapped_dense does not: no whole number of pixels per vertex gives
20 per input vertex at 7 vertices each.

Usage:
  python tools/render_surface_volume.py OUT_ZARR --segment TIFXYZ --volume OME_ZARR [options]

Options:
  --layers 21 --step 1      layers 1 voxel apart, centred on the mesh (with 21 layers, layer 10
                            is the mesh itself)
  --center C                centre the layers C voxels along the normal from the mesh (default 0)
  --normals grid|flipped    normal direction; grid = cross(d/d column, d/d row) as in mesh.py
  --pixels-per-vertex K     default round(1 / the tifxyz scale): one pixel per voxel
  --rows R0 R1 --cols C0 C1 only this part of the vertex grid (default: all of it)
  --volume-level L          the scan's resolution level (default 0); positions are scaled to it
  --cache DIR --workers N   as for snap_segment.py

Rendering is tile by tile (--tile pixels a side) and resumable: finished tiles are listed in
OUT_ZARR.tiles_done (beside the zarr) and skipped on a rerun.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from snap_segment import open_volume, read_crop  # noqa: E402
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import densify, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402

_S = {}


def layer_offsets(layers, step, center):
    return center + (np.arange(layers, dtype=np.float64) - (layers - 1) / 2.0) * step


def tile_pixels(P, N, ok, k, i0, i1, j0, j1):
    """Positions, normals and validity for output pixels [i0, i1) x [j0, j1), from the vertex
    sub-grid that covers them (bilinear inside each cell, exactly as densify on the whole grid)."""
    H, W = ok.shape
    r0, c0 = i0 // k, j0 // k
    r1, c1 = min(H - 1, -(-(i1 - 1) // k)), min(W - 1, -(-(j1 - 1) // k))
    r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
    Pd, Nd, vd = densify(P[r0:r1 + 1, c0:c1 + 1], N[r0:r1 + 1, c0:c1 + 1], ok[r0:r1 + 1, c0:c1 + 1], k)
    a, b = i0 - r0 * k, j0 - c0 * k
    sl = (slice(a, a + (i1 - i0)), slice(b, b + (j1 - j0)))
    return Pd[sl], Nd[sl], vd[sl]


def _init(segment, volume, level, cache, normals_sign, out_zarr):
    import zarr
    P, valid, _ = read_tifxyz(segment)
    P = P.astype(np.float32)
    N, ok = vertex_normals(P, valid)
    _S.update(P=P, N=N * normals_sign, ok=ok, vol=open_volume(volume, level, cache),
              out=zarr.open_array(out_zarr, mode="r+"))


def render_tile(job, k, offsets, level_scale, rows0, cols0):
    i0, i1, j0, j1 = job
    t0 = time.time()
    Q, NQ, vq = tile_pixels(_S["P"], _S["N"], _S["ok"], k, i0 + rows0 * k, i1 + rows0 * k,
                            j0 + cols0 * k, j1 + cols0 * k)
    block = np.zeros((len(offsets), i1 - i0, j1 - j0), np.uint8)
    n_valid = int(vq.sum())
    if n_valid:
        reach = float(np.max(np.abs(offsets)))
        ct, lo = read_crop(_S["vol"], Q, vq, level_scale, int(np.ceil(reach * level_scale)) + 3)
        q = Q[vq] * level_scale - lo
        n = NQ[vq] * level_scale
        ct = ct.astype(np.float32)
        for li, d in enumerate(offsets):
            block[li][vq] = np.clip(np.rint(sample(ct, q + d * n)), 0, 255).astype(np.uint8)
    _S["out"][:, i0:i1, j0:j1] = block
    return job, n_valid, round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_zarr")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--volume-level", type=int, default=0)
    ap.add_argument("--layers", type=int, default=21)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--center", type=float, default=0.0)
    ap.add_argument("--normals", choices=("grid", "flipped"), default="grid")
    ap.add_argument("--pixels-per-vertex", type=int)
    ap.add_argument("--rows", type=int, nargs=2)
    ap.add_argument("--cols", type=int, nargs=2)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import zarr

    P, valid, meta = read_tifxyz(a.segment)
    scale = float(np.atleast_1d(meta.get("scale", [1.0]))[0])
    k = a.pixels_per_vertex or max(1, int(round(1.0 / scale)))
    R0, R1 = a.rows or (0, P.shape[0])
    C0, C1 = a.cols or (0, P.shape[1])
    H, W = (R1 - R0 - 1) * k + 1, (C1 - C0 - 1) * k + 1
    offsets = layer_offsets(a.layers, a.step, a.center)
    if a.tile % a.chunk:
        sys.exit("--tile must be a multiple of --chunk, so tiles never share a chunk")
    attrs = dict(segment=a.segment, volume=a.volume, volume_level=a.volume_level, layers=a.layers, step=a.step,
                 center=a.center, normals=a.normals, pixels_per_vertex=k, rows=[R0, R1], cols=[C0, C1],
                 layer_offsets=[round(float(o), 4) for o in offsets],
                 layout="(layer, row, column); layer offsets are voxels along the normal from the mesh")
    if os.path.exists(os.path.join(a.out_zarr, ".zarray")):
        z = zarr.open_array(a.out_zarr, mode="r+")
        if z.shape != (a.layers, H, W) or dict(z.attrs).get("segment") != a.segment:
            sys.exit(f"{a.out_zarr} exists with another shape or segment; remove it or pick another path")
    else:
        z = zarr.open_array(a.out_zarr, mode="w", shape=(a.layers, H, W), chunks=(a.layers, a.chunk, a.chunk),
                            dtype="u1", fill_value=0)
        z.attrs.update(attrs)
    done_path = a.out_zarr.rstrip("/\\") + ".tiles_done"
    done = set()
    if os.path.exists(done_path):
        done = {tuple(int(v) for v in ln.split()) for ln in open(done_path) if ln.strip()}
    # tiles that touch the mesh at all (from the vertex validity, cheaply)
    okv = vertex_normals(P.astype(np.float32), valid)[1][R0:R1, C0:C1]
    jobs = []
    for i0 in range(0, H, a.tile):
        for j0 in range(0, W, a.tile):
            i1, j1 = min(H, i0 + a.tile), min(W, j0 + a.tile)
            rs, re_ = i0 // k, min(R1 - R0, -(-i1 // k) + 1)
            cs, ce = j0 // k, min(C1 - C0, -(-j1 // k) + 1)
            if okv[rs:re_, cs:ce].any() and (i0, i1, j0, j1) not in done:
                jobs.append((i0, i1, j0, j1))
    level_scale = 1.0 / 2 ** a.volume_level
    print(f"{a.segment}: {P.shape[0]} x {P.shape[1]} vertices, {k} pixels per vertex -> {a.layers} x {H} x {W}; "
          f"layers at {offsets[0]:+.1f} .. {offsets[-1]:+.1f} voxels; {len(jobs)} tiles to render "
          f"({len(done)} already done)", flush=True)
    sign = 1 if a.normals == "grid" else -1
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.segment, a.volume, a.volume_level, a.cache, sign, a.out_zarr)) as ex:
        futs = [ex.submit(render_tile, j, k, offsets, level_scale, R0, C0) for j in jobs]
        with open(done_path, "a") as log:
            for n, fu in enumerate(as_completed(futs), 1):
                job, n_valid, sec = fu.result()
                log.write(" ".join(str(v) for v in job) + "\n")
                log.flush()
                if n % 25 == 0 or n == len(futs):
                    print(f"  {n}/{len(futs)} tiles, {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(dict(attrs, shape=[a.layers, H, W], minutes=round((time.time() - t0) / 60, 1)),
              open(a.out_zarr.rstrip("/\\") + ".json", "w"), indent=1)
    print(f"done -> {a.out_zarr}", flush=True)


if __name__ == "__main__":
    main()
