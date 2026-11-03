"""Carry another surface's depth onto a labelled segment's pixel grid: for every pixel of the target
segment, how far along its normal the other surface lies. Rendering the target at that depth keeps
the target's own pixel grid, so its ink labels still line up, while the depth comes from the other
surface (for example a spiral-fit winding, before and after snapping).

Usage:
  python tools/depth_transfer.py OUT_NPZ --target TIFXYZ --surface TIFXYZ --roi R0 R1 C0 C1
         [--max-depth 10] [--max-gap 2] [--min-dot 0.8]

  --roi        the target's pixels to cover (half-open; pixel = 1/scale per vertex, as rendered)
  --max-depth  ignore the surface beyond this distance along the target's normal (voxels)
  --max-gap    the hit point on the normal must lie within this distance of the surface sample
  --min-dot    and the two surfaces must face within acos(min_dot) of each other

The surface is densified to about one sample per voxel. For each target pixel, its nearest surface
sample's tangent plane is intersected with the target's normal line: depth t = (q - p) . n_s / (n . n_s).
Writes OUT_NPZ with depth (float32, NaN where no surface was found), the gap, the roi, and stats.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from render_surface_volume import tile_pixels  # noqa: E402
from scroll_label_snap.mesh import densify, mean_spacing, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402


def surface_samples(P, valid, lo, hi, margin):
    """The surface's samples about one voxel apart, inside the box [lo - margin, hi + margin]."""
    P = P.astype(np.float32)
    N, ok = vertex_normals(P, valid)
    inside = ok & np.all((P >= lo - margin) & (P <= hi + margin), axis=-1)
    if not inside.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0
    rows, cols = np.nonzero(inside)
    r0, r1 = max(0, rows.min() - 1), min(P.shape[0], rows.max() + 2)
    c0, c1 = max(0, cols.min() - 1), min(P.shape[1], cols.max() + 2)
    k = max(1, int(round(mean_spacing(P, valid))))
    Pd, Nd, vd = densify(P[r0:r1, c0:c1], N[r0:r1, c0:c1], ok[r0:r1, c0:c1], k)
    keep = vd & np.all((Pd >= lo - margin) & (Pd <= hi + margin), axis=-1)
    return Pd[keep], Nd[keep], k


def transfer(Ps, Ns, vs, Qw, Nw, max_depth, max_gap, min_dot):
    """Depth along the target normals Ns of the surface given by samples Qw (normals Nw)."""
    from scipy.spatial import cKDTree
    depth = np.full(vs.shape, np.nan, np.float32)
    gap = np.full(vs.shape, np.nan, np.float32)
    if len(Qw) == 0 or not vs.any():
        return depth, gap
    tree = cKDTree(Qw)
    p, n = Ps[vs].astype(np.float64), Ns[vs].astype(np.float64)
    dist, idx = tree.query(p, distance_upper_bound=float(np.hypot(max_depth, max_gap)) + 1.0, workers=-1)
    found = np.isfinite(dist)
    t = np.full(len(p), np.nan)
    g = np.full(len(p), np.nan)
    q, m = Qw[idx[found]].astype(np.float64), Nw[idx[found]].astype(np.float64)
    pn, nn = p[found], n[found]
    d = np.einsum("ij,ij->i", nn, m)
    good = np.abs(d) >= min_dot
    tt = np.where(good, np.einsum("ij,ij->i", q - pn, m) / np.where(good, d, 1.0), np.nan)
    hit = pn + tt[:, None] * nn
    gg = np.linalg.norm(hit - q, axis=1)
    ok = good & (np.abs(tt) <= max_depth) & (gg <= max_gap)
    t[np.flatnonzero(found)[ok]] = tt[ok]
    g[np.flatnonzero(found)[ok]] = gg[ok]
    depth[vs], gap[vs] = t.astype(np.float32), g.astype(np.float32)
    return depth, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_npz")
    ap.add_argument("--target", required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--roi", type=int, nargs=4, required=True)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--max-gap", type=float, default=2.0)
    ap.add_argument("--min-dot", type=float, default=0.8)
    a = ap.parse_args()
    P, valid, meta = read_tifxyz(a.target)
    P = P.astype(np.float32)
    k = max(1, int(round(1.0 / float(np.atleast_1d(meta.get("scale", [1.0]))[0]))))
    N, ok = vertex_normals(P, valid)
    r0, r1, c0, c1 = a.roi
    Ps, Ns, vs = tile_pixels(P, N, ok, k, r0, r1, c0, c1)
    if not vs.any():
        sys.exit("the target has no valid pixels in this roi")
    lo, hi = Ps[vs].min(0), Ps[vs].max(0)
    Pw, vw, _ = read_tifxyz(a.surface)
    Qw, Nw, kw = surface_samples(Pw, vw, lo, hi, a.max_depth + a.max_gap + 2)
    depth, gap = transfer(Ps, Ns, vs, Qw, Nw, a.max_depth, a.max_gap, a.min_dot)
    have = np.isfinite(depth)
    stats = dict(target=a.target, surface=a.surface, roi=a.roi, pixels_per_vertex=k, surface_densify=kw,
                 surface_samples=int(len(Qw)), target_pixels=int(vs.sum()), with_depth=int(have.sum()),
                 depth_percentiles={str(q): float(np.percentile(depth[have], q)) for q in (1, 5, 25, 50, 75, 95, 99)}
                 if have.any() else None,
                 max_depth=a.max_depth, max_gap=a.max_gap, min_dot=a.min_dot)
    np.savez_compressed(a.out_npz, depth=depth, gap=gap, valid=vs, roi=np.array(a.roi), stats=json.dumps(stats))
    med = stats["depth_percentiles"]["50"] if have.any() else float("nan")
    sname = os.path.basename(os.path.normpath(a.surface))
    tname = os.path.basename(os.path.normpath(a.target))
    print(f"{sname} on {tname}: depth found for {have.sum():,} of {vs.sum():,} pixels "
          f"({have.sum() / max(vs.sum(), 1):.1%}), median {med:+.2f} voxels")


if __name__ == "__main__":
    main()
