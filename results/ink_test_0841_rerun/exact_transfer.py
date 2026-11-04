"""Exact depth transfer: for every target pixel, where its normal line really crosses the surface mesh.

Written for the PHerc. 0841 rerun as an independent check of tools/depth_transfer.py. It shares no code
with that tool's transfer() or surface_samples(): no nearest-sample lookups, no tangent planes, no
gap test. Only the target side is shared on purpose (tile_pixels, vertex_normals), because the pixel grid
and normals must be the very ones render_depth.py renders along.

Usage:
  python -B -P -X utf8 exact_transfer.py OUT_NPZ --target TIFXYZ --surface TIFXYZ --roi R0 R1 C0 C1
         [--max-depth 10] [--min-dot 0.8] [--fit-radius 20] [--cell 1.0] [--diag DIAG_NPZ] [--workers 12]

Writes OUT_NPZ with depth (float32, NaN where nothing is accepted), gap, valid, roi, stats: the format of
tools/depth_transfer.py, so render_depth.py reads it unchanged.

The surface.
  Every grid quad whose four vertices are valid is a bilinear patch (the surface tools/depth_transfer.py
  samples and that the tifxyz grid means). Each patch is cut into f x f cells with f = 2^S, S the smallest
  integer that makes the median vertex spacing / f at most --cell voxels, and each cell into two triangles
  along one diagonal. A triangle mesh of a twisted patch differs from the patch by at most
  |a - b - c + d| / (4 f^2) per cell (a, b, c, d the patch's corners); stats reports that bound
  (`bilinear_bound_*`), so "exact" means exact to that mesh and within this bound of the bilinear surface.

Finding every crossing, with no approximation.
  A bounding-box pyramid over the quads, continued inside each patch by implicit subdivision. A bilinear
  patch and each of its sub-patches lie in the convex hull of their four corners, so the box of the corners
  can never exclude a real crossing. Each pixel's segment p + t n, |t| <= --max-depth, is traced through
  the boxes breadth first, and the Moller-Trumbore test is run on every triangle in every box it enters.
  Barycentric tests are closed with a 1e-9 tolerance, so a line through a shared edge can't slip between
  two triangles. No pixel is skipped and there is no search limit; a crossing is missed only if a triangle is
  degenerate (|det| < 1e-12), in which case the line lies in that triangle's plane and its neighbours
  catch it. Crossings closer than 1e-3 voxels along the line count as one.

Which crossing. The nearest: smallest |t| within --max-depth. If that one fails the normal test the pixel gets
  no depth; a farther crossing is not tried, so the test can never move a pixel to another sheet.

The normal test (the plan's "normals within 37 degrees", --min-dot 0.8) and why this form.
  |n_target . n_sheet| >= min_dot, with n_sheet the normal of a least-squares plane fitted to the valid
  vertices within --fit-radius voxels along the grid (window of 2m + 1 vertices a side, m = round(radius /
  median spacing), at least 1), interpolated bilinearly to the crossing. PCA smallest eigenvector, not an
  average of normals, so it shares no method with the patch's smooth_normals.
  Why not the raw vertex normal: it is a central difference over 2 grid steps, so its scale follows the
  grid. On a 20-24 voxel grid (traced meshes, windings) it looks over +-20-24 voxels; on a 3-voxel
  dense grid over +-3, where it follows the roughness rather than the sheet. The test's job is to ask
  whether the sheet runs alongside the target; a test whose scale changes with the sampling would treat
  arms differently. 20 voxels is the traced meshes' vertex spacing (scale 0.05), the scale the raw test had on
  the surfaces the plan had in mind. It was fixed from that argument before this tool saw real data.
  The diagnostics file holds the raw-normal dot and the no-test depth too, so other choices can be read off.

--diag writes (all rows x cols of the roi): depth_any (nearest crossing, before the normal test), dot_fit,
dot_raw (the plan's raw vertex normal, bilinear at the crossing), ncross (distinct crossings within
--max-depth), sep (largest distance along the line from the nearest crossing to another crossing).
`gap` holds the distance from the crossing to the nearest corner of its quad. The crossing is on the mesh by
construction, so this isn't a test: it only says how far the nearest grid vertex is.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PIN = r"D:/vesuvius/ink/scroll-label-snap-627d058"
for _p in (PIN, os.path.join(PIN, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EPS_BARY = 1e-9
EPS_DET = 1e-12
SAME_CROSSING = 1e-3
CHUNK = 150_000


def set_below_normal():
    """Typed and read back: with ctypes' default types the pseudo-handle is truncated and the call fails silently."""
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_uint32
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.SetPriorityClass.restype = ctypes.c_int
    h = k.GetCurrentProcess()
    k.SetPriorityClass(h, 0x00004000)
    if k.GetPriorityClass(h) != 0x00004000:
        sys.exit("could not set BelowNormal priority; not running")


def _spacing(P, valid):
    d = []
    for a in (0, 1):
        lo = tuple(slice(0, -1) if b == a else slice(None) for b in range(2))
        hi = tuple(slice(1, None) if b == a else slice(None) for b in range(2))
        ok = valid[lo] & valid[hi]
        d.append(np.linalg.norm(P[hi] - P[lo], axis=-1)[ok])
    d = np.concatenate(d)
    return float(np.median(d)) if len(d) else 1.0


def _build_pyramid(P, qok):
    """Boxes per quad and 2 x 2 merges up to a single box. An empty box has lo > hi."""
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
    hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
    lo[~qok] = np.inf
    hi[~qok] = -np.inf
    pyr = [(lo, hi)]
    while pyr[-1][0].shape[0] > 1 or pyr[-1][0].shape[1] > 1:
        l0, h0 = pyr[-1]
        H, W = l0.shape[:2]
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        l1 = np.full((H2, W2, 3), np.inf)
        h1 = np.full((H2, W2, 3), -np.inf)
        for dr in (0, 1):
            for dc in (0, 1):
                sub_l, sub_h = l0[dr::2, dc::2], h0[dr::2, dc::2]
                sh = sub_l.shape[:2]
                l1[:sh[0], :sh[1]] = np.minimum(l1[:sh[0], :sh[1]], sub_l)
                h1[:sh[0], :sh[1]] = np.maximum(h1[:sh[0], :sh[1]], sub_h)
        pyr.append((l1, h1))
    return pyr


def _seg_box(p, inv, lo, hi, T):
    """Segment p + t n, |t| <= T, against boxes. inv = 1 / n with zeros replaced by 1e12."""
    t1 = (lo - p) * inv
    t2 = (hi - p) * inv
    tmin = np.minimum(t1, t2).max(1)
    tmax = np.maximum(t1, t2).min(1)
    return (tmin <= tmax) & (tmin <= T) & (tmax >= -T) & (lo[:, 0] <= hi[:, 0])


def _bilin(a, b, c, d, u, v):
    """Patch position at row fraction u, column fraction v."""
    return (a * (1 - v) + b * v) * (1 - u) + (c * (1 - v) + d * v) * u


def _triangles(p, n, v0, v1, v2, T):
    """Moller-Trumbore for each row. Returns (hit, t, bu, bv)."""
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(n, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > EPS_DET
    idet = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - v0
    bu = (s * h).sum(1) * idet
    q = np.cross(s, e1)
    bv = (n * q).sum(1) * idet
    t = (e2 * q).sum(1) * idet
    hit = ok & (bu >= -EPS_BARY) & (bv >= -EPS_BARY) & (bu + bv <= 1 + EPS_BARY) & (np.abs(t) <= T)
    return hit, t, bu, bv


def _trace_chunk(ids, p_all, n_all, inv_all, pyr, P, S, T, other_diagonal=False):
    """All crossings of the segments `ids`: (pixel id, t, row position, column position on the grid)."""
    p, n, inv = p_all[ids], n_all[ids], inv_all[ids]
    pix = np.arange(len(ids))
    r = np.zeros(len(ids), np.int64)
    c = np.zeros(len(ids), np.int64)
    lev = len(pyr) - 1
    while True:                                 # the pyramid over the quads
        lo, hi = pyr[lev]
        keep = _seg_box(p[pix], inv[pix], lo[r, c], hi[r, c], T)
        pix, r, c = pix[keep], r[keep], c[keep]
        if lev == 0 or len(pix) == 0:
            break
        lev -= 1
        H, W = pyr[lev][0].shape[:2]
        pix = np.repeat(pix, 4)
        r = np.repeat(2 * r, 4) + np.tile([0, 0, 1, 1], len(r))
        c = np.repeat(2 * c, 4) + np.tile([0, 1, 0, 1], len(c))
        inside = (r < H) & (c < W)
        pix, r, c = pix[inside], r[inside], c[inside]
    empty = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    if len(pix) == 0:
        return empty
    i = np.zeros(len(pix), np.int64)            # inside the patches, by implicit subdivision
    j = np.zeros(len(pix), np.int64)
    for s in range(S + 1):
        f = 2.0 ** s
        a, b, cc, d = P[r, c], P[r, c + 1], P[r + 1, c], P[r + 1, c + 1]
        u0, u1 = (i / f)[:, None], ((i + 1) / f)[:, None]
        v0, v1 = (j / f)[:, None], ((j + 1) / f)[:, None]
        c00 = _bilin(a, b, cc, d, u0, v0)
        c01 = _bilin(a, b, cc, d, u0, v1)
        c10 = _bilin(a, b, cc, d, u1, v0)
        c11 = _bilin(a, b, cc, d, u1, v1)
        lo = np.minimum(np.minimum(c00, c01), np.minimum(c10, c11))
        hi = np.maximum(np.maximum(c00, c01), np.maximum(c10, c11))
        keep = _seg_box(p[pix], inv[pix], lo, hi, T)
        if s == S:
            pix, r, c, i, j = pix[keep], r[keep], c[keep], i[keep], j[keep]
            c00, c01, c10, c11 = c00[keep], c01[keep], c10[keep], c11[keep]
            break
        pix, r, c, i, j = pix[keep], r[keep], c[keep], i[keep], j[keep]
        if len(pix) == 0:
            return empty
        pix = np.repeat(pix, 4)
        r = np.repeat(r, 4)
        c = np.repeat(c, 4)
        i = np.repeat(2 * i, 4) + np.tile([0, 0, 1, 1], len(i))
        j = np.repeat(2 * j, 4) + np.tile([0, 1, 0, 1], len(j))
    if len(pix) == 0:
        return empty
    pp, nn = p[pix], n[pix]
    f = 2.0 ** S
    out = []
    if not other_diagonal:                       # diagonal c00 - c11
        h1, t1, a1, b1 = _triangles(pp, nn, c00, c01, c11, T)   # local (row, col) = (b, a + b)
        out.append((h1, t1, b1, a1 + b1))
        h2, t2, a2, b2 = _triangles(pp, nn, c00, c11, c10, T)   # local (row, col) = (a + b, a)
        out.append((h2, t2, a2 + b2, a2))
    else:                                        # diagonal c01 - c10
        h1, t1, a1, b1 = _triangles(pp, nn, c00, c01, c10, T)   # local (row, col) = (b, a)
        out.append((h1, t1, b1, a1))
        h2, t2, a2, b2 = _triangles(pp, nn, c11, c10, c01, T)   # local (row, col) = (1 - b, 1 - a)
        out.append((h2, t2, 1 - b2, 1 - a2))
    ri, ti, ro, co = [], [], [], []
    for h, t, lr, lc in out:
        ri.append(ids[pix[h]])
        ti.append(t[h])
        ro.append(r[h] + (i[h] + lr[h]) / f)
        co.append(c[h] + (j[h] + lc[h]) / f)
    return np.concatenate(ri), np.concatenate(ti), np.concatenate(ro), np.concatenate(co)


def fit_normals(P, valid, m):
    """Unit normal of the least-squares plane through the valid vertices in a (2m + 1)^2 window,
    signed to agree with the grid's orientation. Zero where fewer than 3 vertices or a degenerate window."""
    from scipy import ndimage as ndi
    size = 2 * m + 1
    w = valid.astype(np.float64)
    Q = np.where(valid[..., None], P, 0.0)
    off = Q[valid].mean(0) if valid.any() else np.zeros(3)
    Q = np.where(valid[..., None], P - off, 0.0)
    box = lambda x: ndi.uniform_filter(x, size=size, mode="constant")
    cnt = box(w)
    mean = np.stack([box(Q[..., k]) for k in range(3)], -1)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    mom = {pr: box(Q[..., pr[0]] * Q[..., pr[1]]) for pr in pairs}
    out = np.zeros(P.shape, np.float64)
    use = valid & (cnt * size * size >= 3 - 1e-6)
    idx = np.nonzero(use)
    if len(idx[0]) == 0:
        return out
    cn = cnt[idx]
    mn = mean[idx] / cn[:, None]
    C = np.zeros((len(cn), 3, 3))
    for (a, b) in pairs:
        val = mom[(a, b)][idx] / cn - mn[:, a] * mn[:, b]
        C[:, a, b] = val
        C[:, b, a] = val
    _, vec = np.linalg.eigh(C)
    nrm = vec[:, :, 0]                          # smallest eigenvalue
    # orientation: agree with the grid's own cross product over the same window (sign only)
    dr = np.gradient(np.where(valid[..., None], P, np.nan), axis=0)
    dc = np.gradient(np.where(valid[..., None], P, np.nan), axis=1)
    gn = np.cross(dc[..., ::-1], dr[..., ::-1])[..., ::-1]
    gn = np.nan_to_num(gn)
    gn = np.stack([box(gn[..., k]) for k in range(3)], -1)[idx]
    flip = (nrm * gn).sum(1) < 0
    nrm[flip] *= -1
    out[idx] = nrm
    return out


def _at(F, ro, co):
    """Bilinear lookup of a per-vertex vector field at grid positions, renormalised."""
    r0 = np.minimum(np.floor(ro).astype(np.int64), F.shape[0] - 2)
    c0 = np.minimum(np.floor(co).astype(np.int64), F.shape[1] - 2)
    u, v = (ro - r0)[:, None], (co - c0)[:, None]
    x = _bilin(F[r0, c0], F[r0, c0 + 1], F[r0 + 1, c0], F[r0 + 1, c0 + 1], u, v)
    nm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nm, 1e-12)


def exact_core(Ps, Ns, vs, P, valid, max_depth=10.0, min_dot=0.8, fit_radius=20.0, cell=1.0, workers=12,
               other_diagonal=False, raw_normals=None, verbose=False):
    """Ps, Ns, vs: target pixels (rows, cols, 3), normals, validity. P, valid: surface grid (H, W, 3) z y x.
    raw_normals: (N, ok) of the surface as the plan defines them (vertex_normals), or None.
    Returns dict of rows x cols arrays and an info dict."""
    shape = vs.shape
    res = dict(depth=np.full(shape, np.nan, np.float32), depth_any=np.full(shape, np.nan, np.float32),
               gap=np.full(shape, np.nan, np.float32), dot_fit=np.full(shape, np.nan, np.float32),
               dot_raw=np.full(shape, np.nan, np.float32), ncross=np.zeros(shape, np.uint8),
               sep=np.full(shape, np.nan, np.float32))
    info = dict(max_depth=max_depth, min_dot=min_dot, fit_radius=fit_radius, cell=cell, other_diagonal=other_diagonal)
    P = P.astype(np.float64)
    qok0 = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    info.update(valid_quads=int(qok0.sum()))
    if not vs.any() or not qok0.any():
        return res, dict(info, triangles=0, subdivide=0, fit_window=0, spacing=None)
    spacing = _spacing(P, valid)
    p_all, n_all = Ps[vs].astype(np.float64), Ns[vs].astype(np.float64)
    bb_lo, bb_hi = p_all.min(0) - max_depth - 1, p_all.max(0) + max_depth + 1
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    qlo = np.minimum(np.minimum(a, b), np.minimum(c, d))
    qhi = np.maximum(np.maximum(a, b), np.maximum(c, d))
    near = qok0 & np.all((qhi >= bb_lo) & (qlo <= bb_hi), axis=-1)
    if not near.any():
        return res, dict(info, triangles=0, subdivide=0, fit_window=0, spacing=spacing)
    rows, cols = np.nonzero(near)
    m = max(1, int(round(fit_radius / spacing)))
    r0, r1 = rows.min(), rows.max() + 2
    c0, c1 = cols.min(), cols.max() + 2
    # normals need the window around the crop too
    R0, R1 = max(0, r0 - m), min(P.shape[0], r1 + m)
    C0, C1 = max(0, c0 - m), min(P.shape[1], c1 + m)
    Pw, vw = P[R0:R1, C0:C1], valid[R0:R1, C0:C1]
    Nfit = fit_normals(Pw, vw, m)
    Nraw = None
    if raw_normals is not None:
        Nraw = raw_normals[0][R0:R1, C0:C1].astype(np.float64)
    qok = np.zeros(qok0[R0:R1 - 1, C0:C1 - 1].shape, bool)
    qok[r0 - R0:r1 - 1 - R0, c0 - C0:c1 - 1 - C0] = near[r0:r1 - 1, c0:c1 - 1]
    S = 0
    while spacing / 2 ** S > cell:
        S += 1
    a, b, c, d = Pw[:-1, :-1], Pw[:-1, 1:], Pw[1:, :-1], Pw[1:, 1:]
    bound = np.linalg.norm(a - b - c + d, axis=-1)[qok] / (4 * 4 ** S)
    info.update(spacing=spacing, subdivide=2 ** S, fit_window=2 * m + 1, quads_used=int(qok.sum()),
                triangles=int(qok.sum()) * 2 * 4 ** S,
                bilinear_bound_p50=float(np.percentile(bound, 50)), bilinear_bound_p99=float(np.percentile(bound, 99)),
                bilinear_bound_max=float(bound.max()))
    pyr = _build_pyramid(Pw, qok)
    safe = np.where(np.abs(n_all) < 1e-12, 1e-12, n_all)
    inv_all = 1.0 / safe
    N = len(p_all)
    chunks = [np.arange(i, min(N, i + CHUNK)) for i in range(0, N, CHUNK)]
    t0 = time.time()
    done = [0]

    def run(ids):
        out = _trace_chunk(ids, p_all, n_all, inv_all, pyr, Pw, S, max_depth, other_diagonal)
        done[0] += 1
        if verbose and done[0] % 10 == 0:
            print(f"   {done[0]}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
        return out

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(run, chunks))
    else:
        parts = [run(ch) for ch in chunks]
    pid = np.concatenate([x[0] for x in parts])
    tt = np.concatenate([x[1] for x in parts])
    ro = np.concatenate([x[2] for x in parts])
    co = np.concatenate([x[3] for x in parts])
    info.update(crossing_records=int(len(pid)), seconds=round(time.time() - t0, 1))
    if len(pid) == 0:
        return res, info
    # nearest crossing per pixel, distinct crossings per pixel
    order = np.lexsort((tt, pid))
    pid_s, tt_s = pid[order], tt[order]
    newpix = np.r_[True, pid_s[1:] != pid_s[:-1]]
    newcross = newpix | np.r_[True, np.abs(np.diff(tt_s)) > SAME_CROSSING]
    grp = np.cumsum(newpix) - 1
    ncr = np.bincount(grp, weights=newcross.astype(np.float64)).astype(np.int64)
    order2 = np.lexsort((np.abs(tt), pid))
    pid2 = pid[order2]
    first = order2[np.r_[True, pid2[1:] != pid2[:-1]]]
    upix = pid[first]
    t_near = tt[first]
    tmin = np.full(N, np.inf)
    tmax = np.full(N, -np.inf)
    np.minimum.at(tmin, pid, tt)
    np.maximum.at(tmax, pid, tt)
    sepv = np.maximum(tmax[upix] - t_near, t_near - tmin[upix])
    rn, cn = ro[first], co[first]
    nn = n_all[upix]
    nfit = _at(Nfit, rn, cn)
    dot_fit = np.abs((nn * nfit).sum(1))
    dot_raw = np.abs((nn * _at(Nraw, rn, cn)).sum(1)) if Nraw is not None else np.full(len(upix), np.nan)
    r_i = np.minimum(np.round(rn).astype(np.int64), Pw.shape[0] - 1)
    c_i = np.minimum(np.round(cn).astype(np.int64), Pw.shape[1] - 1)
    hitp = p_all[upix] + t_near[:, None] * nn
    gapv = np.linalg.norm(hitp - Pw[r_i, c_i], axis=1)
    H = np.flatnonzero(vs.ravel())[upix]
    for key, val in (("depth_any", t_near), ("dot_fit", dot_fit), ("dot_raw", dot_raw), ("sep", sepv), ("gap", gapv)):
        res[key].ravel()[H] = val.astype(np.float32)
    res["ncross"].ravel()[H] = np.minimum(ncr, 255).astype(np.uint8)
    keep = dot_fit >= min_dot
    res["depth"].ravel()[H[keep]] = t_near[keep].astype(np.float32)
    res["gap"].ravel()[H[~keep]] = np.nan
    return res, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_npz")
    ap.add_argument("--target", required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--roi", type=int, nargs=4, required=True)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--min-dot", type=float, default=0.8)
    ap.add_argument("--fit-radius", type=float, default=20.0)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--other-diagonal", action="store_true")
    ap.add_argument("--diag")
    a = ap.parse_args()
    if a.workers > 12:
        sys.exit("at most 12 threads on this PC")
    set_below_normal()
    from render_surface_volume import tile_pixels
    from scroll_label_snap.mesh import vertex_normals
    from scroll_label_snap.tifxyz import read_tifxyz
    Pt, vt, meta = read_tifxyz(a.target)
    Pt = Pt.astype(np.float32)
    k = max(1, int(round(1.0 / float(np.atleast_1d(meta.get("scale", [1.0]))[0]))))
    Nt, okt = vertex_normals(Pt, vt)
    r0, r1, c0, c1 = a.roi
    Ps, Ns, vs = tile_pixels(Pt, Nt, okt, k, r0, r1, c0, c1)
    if not vs.any():
        sys.exit("the target has no valid pixels in this roi")
    Pw, vw, _ = read_tifxyz(a.surface)
    Pw32 = Pw.astype(np.float32)
    raw = vertex_normals(Pw32, vw)
    res, info = exact_core(Ps, Ns, vs, Pw, vw, a.max_depth, a.min_dot, a.fit_radius, a.cell, a.workers,
                           a.other_diagonal, raw, verbose=True)
    depth = res["depth"]
    have = np.isfinite(depth)
    stats = dict(target=a.target, surface=a.surface, roi=a.roi, pixels_per_vertex=k, method="exact line-triangle",
                 target_pixels=int(vs.sum()), with_depth=int(have.sum()), with_crossing=int(np.isfinite(res["depth_any"]).sum()),
                 depth_percentiles={str(q): float(np.percentile(depth[have], q)) for q in (1, 5, 25, 50, 75, 95, 99)}
                 if have.any() else None, max_gap=None, **info)
    np.savez_compressed(a.out_npz, depth=depth, gap=res["gap"], valid=vs, roi=np.array(a.roi), stats=json.dumps(stats))
    if a.diag:
        np.savez_compressed(a.diag, depth_any=res["depth_any"], dot_fit=res["dot_fit"], dot_raw=res["dot_raw"],
                            ncross=res["ncross"], sep=res["sep"])
    med = stats["depth_percentiles"]["50"] if have.any() else float("nan")
    print(f"{os.path.basename(os.path.normpath(a.surface))} on {os.path.basename(os.path.normpath(a.target))}: "
          f"depth for {have.sum():,} of {vs.sum():,} pixels ({have.sum() / max(vs.sum(), 1):.1%}), "
          f"crossings at {stats['with_crossing']:,}, median {med:+.2f} voxels, {info.get('seconds')} s", flush=True)


if __name__ == "__main__":
    main()
