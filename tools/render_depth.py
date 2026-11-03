"""Render a surface volume of a target segment moved to another surface's depth (from
tools/depth_transfer.py): on the target's own pixel grid and along the target's own normals, so
only the depth differs between surfaces carried onto the same target, and the target's ink labels
line up with every render.

Usage:
  python tools/render_depth.py OUT_ZARR --target TIFXYZ --depth NPZ --volume OME_ZARR
         [--layers 21 --step 1 --center 0 --normals grid|flipped --volume-level 0 --cache DIR --workers N]

The layout is that of tools/render_surface_volume.py: (layers, rows, columns) uint8 over the depth
file's roi, 0 where the target or the depth is missing, layer offsets along the (oriented) normal
around the moved surface. Resumable the same way (OUT_ZARR.tiles_done).
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
from render_surface_volume import layer_offsets, tile_pixels  # noqa: E402
from snap_segment import open_volume, read_crop  # noqa: E402
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402

_S = {}


def _init(target, depth_npz, volume, level, cache, sign, out_zarr):
    import zarr
    P, valid, _ = read_tifxyz(target)
    P = P.astype(np.float32)
    N, ok = vertex_normals(P, valid)
    d = np.load(depth_npz)
    _S.update(P=P, N=N, ok=ok, depth=d["depth"], roi=d["roi"].tolist(), sign=sign,
              vol=open_volume(volume, level, cache), out=zarr.open_array(out_zarr, mode="r+"))


def render_tile(job, k, offsets, level_scale):
    i0, i1, j0, j1 = job                      # pixels of the roi
    r0, _, c0, _ = _S["roi"]
    t0 = time.time()
    Q, NQ, vq = tile_pixels(_S["P"], _S["N"], _S["ok"], k, i0 + r0, i1 + r0, j0 + c0, j1 + c0)
    dep = _S["depth"][i0:i1, j0:j1]
    vq = vq & np.isfinite(dep)
    block = np.zeros((len(offsets), i1 - i0, j1 - j0), np.uint8)
    n_valid = int(vq.sum())
    if n_valid:
        M = Q + np.where(vq, dep, 0)[..., None] * NQ          # the target moved to the surface's depth
        n_out = NQ * _S["sign"]
        reach = float(np.max(np.abs(offsets)))
        ct, lo = read_crop(_S["vol"], M, vq, level_scale, int(np.ceil(reach * level_scale)) + 3)
        q = M[vq] * level_scale - lo
        n = n_out[vq] * level_scale
        ct = ct.astype(np.float32)
        for li, d in enumerate(offsets):
            block[li][vq] = np.clip(np.rint(sample(ct, q + d * n)), 0, 255).astype(np.uint8)
    _S["out"][:, i0:i1, j0:j1] = block
    return job, n_valid, round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_zarr")
    ap.add_argument("--target", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--volume-level", type=int, default=0)
    ap.add_argument("--layers", type=int, default=21)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--center", type=float, default=0.0)
    ap.add_argument("--normals", choices=("grid", "flipped"), default="grid")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import zarr
    _, _, meta = read_tifxyz(a.target)
    k = max(1, int(round(1.0 / float(np.atleast_1d(meta.get("scale", [1.0]))[0]))))
    d = np.load(a.depth)
    r0, r1, c0, c1 = d["roi"].tolist()
    H, W = r1 - r0, c1 - c0
    if d["depth"].shape != (H, W):
        sys.exit(f"the depth file is {d['depth'].shape}, its roi {H} x {W}")
    have = np.isfinite(d["depth"])
    offsets = layer_offsets(a.layers, a.step, a.center)
    if a.tile % a.chunk:
        sys.exit("--tile must be a multiple of --chunk")
    attrs = dict(target=a.target, depth=a.depth, volume=a.volume, volume_level=a.volume_level, layers=a.layers,
                 step=a.step, center=a.center, normals=a.normals, pixels_per_vertex=k, roi=[r0, r1, c0, c1],
                 layer_offsets=[round(float(o), 4) for o in offsets],
                 layout="(layer, row, column) over the roi; layer offsets are voxels along the target's normal "
                        "from the target moved to the surface's depth")
    if os.path.exists(os.path.join(a.out_zarr, ".zarray")):
        z = zarr.open_array(a.out_zarr, mode="r+")
        if z.shape != (a.layers, H, W) or dict(z.attrs).get("depth") != a.depth:
            sys.exit(f"{a.out_zarr} exists with another shape or depth file; remove it or pick another path")
    else:
        z = zarr.open_array(a.out_zarr, mode="w", shape=(a.layers, H, W), chunks=(a.layers, a.chunk, a.chunk),
                            dtype="u1", fill_value=0)
        z.attrs.update(attrs)
    done_path = a.out_zarr.rstrip("/\\") + ".tiles_done"
    done = set()
    if os.path.exists(done_path):
        done = {tuple(int(v) for v in ln.split()) for ln in open(done_path) if ln.strip()}
    jobs = []
    for i0 in range(0, H, a.tile):
        for j0 in range(0, W, a.tile):
            i1, j1 = min(H, i0 + a.tile), min(W, j0 + a.tile)
            if have[i0:i1, j0:j1].any() and (i0, i1, j0, j1) not in done:
                jobs.append((i0, i1, j0, j1))
    level_scale = 1.0 / 2 ** a.volume_level
    print(f"{a.depth}: {H} x {W} pixels ({have.mean():.1%} with depth) -> {a.layers} layers at {offsets[0]:+.1f} .. "
          f"{offsets[-1]:+.1f}; {len(jobs)} tiles to render ({len(done)} already done)", flush=True)
    sign = 1 if a.normals == "grid" else -1
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.target, a.depth, a.volume, a.volume_level, a.cache, sign, a.out_zarr)) as ex:
        futs = [ex.submit(render_tile, j, k, offsets, level_scale) for j in jobs]
        with open(done_path, "a") as log:
            for n, fu in enumerate(as_completed(futs), 1):
                job, _, _ = fu.result()
                log.write(" ".join(str(v) for v in job) + "\n")
                log.flush()
                if n % 25 == 0 or n == len(futs):
                    print(f"  {n}/{len(futs)} tiles, {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(dict(attrs, shape=[a.layers, H, W], minutes=round((time.time() - t0) / 60, 1)),
              open(a.out_zarr.rstrip("/\\") + ".json", "w"), indent=1)
    print(f"done -> {a.out_zarr}", flush=True)


if __name__ == "__main__":
    main()
