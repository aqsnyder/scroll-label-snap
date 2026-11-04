"""Carry another surface's depth onto a labelled segment's pixel grid: for every pixel of the target
segment, how far along its normal the other surface lies. Rendering the target at that depth keeps
the target's own pixel grid, so its ink labels still line up, while the depth comes from the other
surface (for example a spiral-fit winding, before and after snapping).

Usage:
  python tools/depth_transfer.py OUT_NPZ --target TIFXYZ --surface TIFXYZ --roi R0 R1 C0 C1
         [--max-depth 10] [--max-gap 2] [--min-dot 0.8] [--refine 0] [--normal-sigma 0]

  --roi        the target's pixels to cover (half-open; pixel = 1/scale per vertex, as rendered)
  --max-depth  ignore the surface beyond this distance along the target's normal (voxels)
  --max-gap    the hit point on the normal must lie within this distance of the surface sample
  --min-dot    and the two surfaces must face within acos(min_dot) of each other
  --refine     times the sample nearest the current hit point replaces the sample nearest the pixel
               (0, the default, is what results/ink_test_0841_plan.md ran: the gap is measured to the
               sample nearest the pixel; use 3 for the corrected test)
  --normal-sigma  smooth the surface's normals over this many voxels (0 = none, the default; use 10
               with --refine 3)

The surface is densified to about one sample per voxel. For each target pixel, its nearest surface
sample's tangent plane is intersected with the target's normal line: depth t = (q - p) . n_s / (n . n_s).
The sample nearest the pixel is where the surface faces the pixel, which on a rough surface can be far
from where the normal line crosses it, by about depth x local slope. With --refine 0 such a pixel fails
the gap test and stays NaN, and tools/render_depth.py renders it as zeros, so a rough surface comes out
speckled with holes. With --refine the sample nearest the hit point is taken instead, and the gap and
normal tests are applied to it.
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


def smooth_normals(N, ok, sigma):
    """Normals averaged with a Gaussian of sigma grid steps over the valid vertices, renormalised."""
    from scipy import ndimage as ndi
    w = ndi.gaussian_filter(ok.astype(np.float32), sigma, mode="constant")
    S = np.stack([ndi.gaussian_filter(N[..., c] * ok, sigma, mode="constant") for c in range(3)], -1)
    S = S / np.maximum(w, 1e-6)[..., None]
    norm = np.linalg.norm(S, axis=-1)
    ok = ok & (norm > 0.5)
    S = S / np.maximum(norm, 1e-12)[..., None]
    S[~ok] = 0
    return S.astype(np.float32), ok


def surface_samples(P, valid, lo, hi, margin, normal_sigma=0.0):
    """The surface's samples about one voxel apart, inside the box [lo - margin, hi + margin]."""
    P = P.astype(np.float32)
    N, ok = vertex_normals(P, valid)
    if normal_sigma > 0:
        N, ok = smooth_normals(N, ok, normal_sigma / mean_spacing(P, valid))
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


def transfer(Ps, Ns, vs, Qw, Nw, max_depth, max_gap, min_dot, refine=0):
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
    if refine > 0:
        d = np.einsum("ij,ij->i", nn, m)
        d = np.where(np.abs(d) < 0.2, np.nan, d)            # nearly parallel: no hit on this plane
        tt = np.einsum("ij,ij->i", q - pn, m) / d
        alive = np.isfinite(tt) & (np.abs(tt) <= max_depth + max_gap + 1)
        for _ in range(refine):
            hit = pn[alive] + tt[alive][:, None] * nn[alive]
            _, j = tree.query(hit, workers=-1)
            qa, ma = Qw[j].astype(np.float64), Nw[j].astype(np.float64)
            da = np.einsum("ij,ij->i", nn[alive], ma)
            da = np.where(np.abs(da) < 0.2, np.nan, da)
            step = np.einsum("ij,ij->i", qa - hit, ma) / da
            tt[np.flatnonzero(alive)] += np.clip(np.nan_to_num(step), -3, 3)
        hit = pn + tt[:, None] * nn
        _, j = tree.query(np.nan_to_num(hit), workers=-1)
        q, m = Qw[j].astype(np.float64), Nw[j].astype(np.float64)
        gg = np.linalg.norm(hit - q, axis=1)
        with np.errstate(invalid="ignore"):
            ok = alive & (np.abs(tt) <= max_depth) & (gg <= max_gap) & (np.abs(np.einsum("ij,ij->i", nn, m)) >= min_dot)
        t[np.flatnonzero(found)[ok]] = tt[ok]
        g[np.flatnonzero(found)[ok]] = gg[ok]
        depth[vs], gap[vs] = t.astype(np.float32), g.astype(np.float32)
        return depth, gap
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
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--normal-sigma", type=float, default=0.0)
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
    Qw, Nw, kw = surface_samples(Pw, vw, lo, hi, a.max_depth + a.max_gap + 2, a.normal_sigma)
    depth, gap = transfer(Ps, Ns, vs, Qw, Nw, a.max_depth, a.max_gap, a.min_dot, a.refine)
    have = np.isfinite(depth)
    stats = dict(target=a.target, surface=a.surface, roi=a.roi, pixels_per_vertex=k, surface_densify=kw,
                 surface_samples=int(len(Qw)), target_pixels=int(vs.sum()), with_depth=int(have.sum()),
                 depth_percentiles={str(q): float(np.percentile(depth[have], q)) for q in (1, 5, 25, 50, 75, 95, 99)}
                 if have.any() else None,
                 max_depth=a.max_depth, max_gap=a.max_gap, min_dot=a.min_dot,
                 refine=a.refine, normal_sigma=a.normal_sigma)
    np.savez_compressed(a.out_npz, depth=depth, gap=gap, valid=vs, roi=np.array(a.roi), stats=json.dumps(stats))
    med = stats["depth_percentiles"]["50"] if have.any() else float("nan")
    sname = os.path.basename(os.path.normpath(a.surface))
    tname = os.path.basename(os.path.normpath(a.target))
    print(f"{sname} on {tname}: depth found for {have.sum():,} of {vs.sum():,} pixels "
          f"({have.sum() / max(vs.sum(), 1):.1%}), median {med:+.2f} voxels")


if __name__ == "__main__":
    main()
