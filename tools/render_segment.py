"""Render a whole segment before and after snapping: the CT just inside and just outside the
recto face, read through the original mesh and through the snapped one (the whole-segment
version of tools/render_mesh.py).

Usage:
  python tools/render_segment.py OUT_PREFIX SNAP_DIR --volume OME_ZARR [--cache DIR]
         [--depths -1.5 1.5] [--pixel 2] [--workers 4]

  SNAP_DIR is the OUT_DIR of tools/snap_segment.py (its snap_segment.json says which segment,
  which side is recto and where the face is).

Each mesh is resampled at --pixel voxels and the CT (lightly smoothed) is read at `depth` voxels
from where the mesh puts the recto face, tile by tile. Writes OUT_PREFIX_<mesh>_<depth>.png for
each mesh and depth, and OUT_PREFIX_compare.png (all of them, downsized).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snap_segment import open_volume, read_crop  # noqa: E402
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import densify, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402

_S = {}


def _init(volume, cache, level):
    _S["vol"] = open_volume(volume, level, cache)


def render_tile(Q, NQ, vq, depths, level_scale, margin=8):
    """CT at the given depths along the normals NQ for the points Q (volume coordinates)."""
    out = np.full((len(depths),) + vq.shape, np.nan, np.float32)
    if vq.sum() < 4:
        return out
    ct, lo = read_crop(_S["vol"], Q, vq, level_scale, margin + int(np.ceil(max(abs(d) for d in depths))) + 2)
    ct_s = ndi.gaussian_filter(ct.astype(np.float32), 0.5)
    q = Q[vq] * level_scale - lo
    n = NQ[vq]
    for k, d in enumerate(depths):
        out[k][vq] = sample(ct_s, q + d * level_scale * n)
    return out


def mesh_pixels(P, valid, sign, f):
    """The mesh resampled by an integer factor f: positions and oriented normals per pixel."""
    N, ok = vertex_normals(P, valid)
    return densify(P, N * sign, ok, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_prefix")
    ap.add_argument("snap_dir")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--depths", type=float, nargs="+", default=[-1.5, 1.5])
    ap.add_argument("--pixel", type=float, default=1.5, help="about this many voxels per output pixel")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tile", type=int, default=200, help="pixels per tile side")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.load(open(os.path.join(a.snap_dir, "snap_segment.json")))
    name, sign, c = s["name"], s["normal_sign"], s["convention_offset"]
    level_scale = 1.0 / 2 ** s.get("level", 0)
    seg = s["segment"]
    if seg.startswith(("http://", "https://")):
        seg = os.path.join(a.snap_dir, "segment.tifxyz")
    meshes = {"original": read_tifxyz(seg)[:2],
              "snapped": read_tifxyz(os.path.join(a.snap_dir, f"{name}_snapped_dense.tifxyz"))[:2]}
    depths = [c + d for d in a.depths]
    # both meshes on the same pixel grid: the snapped dense grid is the input grid upsampled by
    # `factor`, so upsample it by k and the input grid by factor * k
    factor = int(s["factor"])
    P0, v0 = meshes["original"]
    N0, ok0 = vertex_normals(P0.astype(np.float32), v0)
    sp = float(np.median(np.linalg.norm(np.diff(P0, axis=1)[ok0[:, 1:] & ok0[:, :-1]], axis=-1))) / factor
    k = max(1, int(round(sp / a.pixel)))
    factors = {"original": factor * k, "snapped": k}
    print(f"pixels of {sp / k:.2f} voxels", flush=True)
    images = {}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.volume, a.cache, s.get("level", 0))) as ex:
        for mname, (P, valid) in meshes.items():
            Pd, Nd, vd = mesh_pixels(P.astype(np.float32), valid, sign, factors[mname])
            H, W = vd.shape
            jobs = []
            for r in range(0, H, a.tile):
                for cc in range(0, W, a.tile):
                    sl = (slice(r, min(H, r + a.tile)), slice(cc, min(W, cc + a.tile)))
                    jobs.append((sl, ex.submit(render_tile, Pd[sl], Nd[sl], vd[sl], depths, level_scale)))
            img = np.full((len(depths), H, W), np.nan, np.float32)
            for k, (sl, fu) in enumerate(jobs):
                img[(slice(None),) + sl] = fu.result()
                if (k + 1) % 50 == 0:
                    print(f"  {mname}: {k + 1}/{len(jobs)} tiles", flush=True)
            images[mname] = img
            print(f"{mname}: {H} x {W} pixels", flush=True)
    allv = np.concatenate([im[np.isfinite(im)] for im in images.values()])
    lo, hi = np.percentile(allv, [1, 99])
    for mname, img in images.items():
        for k, d in enumerate(a.depths):
            x = np.clip((np.nan_to_num(img[k], nan=lo) - lo) / (hi - lo), 0, 1)
            plt.imsave(f"{a.out_prefix}_{mname}_{d:+.1f}.png", x, cmap="gray", vmin=0, vmax=1)
    fig, axes = plt.subplots(len(a.depths), 2, figsize=(14, 7 * len(a.depths) * images["original"].shape[1]
                                                        / max(1, images["original"].shape[2])), squeeze=False)
    for k, d in enumerate(a.depths):
        for j, mname in enumerate(("original", "snapped")):
            ax = axes[k, j]
            ax.imshow(images[mname][k], cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased")
            ax.set_title(f"{name}, {mname} mesh, {d:+.1f} voxels from its recto face", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{a.out_prefix}_compare.png", dpi=110)
    print("wrote", f"{a.out_prefix}_compare.png")


if __name__ == "__main__":
    main()
