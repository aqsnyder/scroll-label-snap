"""Snap a VC3D surface mesh (tifxyz grid of vertices) onto the papyrus surface.

The same scoring and graph cut as label snapping (core.solve_surface), applied to a mesh:

  1. Normals come from the grid itself: the cross product of the row and column directions,
     so they agree along the whole mesh. They are then flipped as a whole, if needed, to point
     from papyrus toward the gap at the face the mesh follows (judged from the CT).
  2. The mesh is densified bilinearly to about p.spacing voxels between samples, so it is
     scored and solved at the same resolution as labels. Neighbours are the 4 grid neighbours.
  3. Each dense sample gets a shift along its normal. The mesh keeps its own convention: where
     it sits relative to the papyrus edge is estimated from the mesh and kept, unless
     p.convention_offset says where it should sit instead.
  4. Output: the snapped dense surface, and the input grid with each vertex moved so the
     bilinear mesh best matches the snapped dense surface (least squares).

Positions are (z, y, x). The CT is a crop: crop coordinates = volume coordinates * level_scale
- offset_zyx (level_scale = 1 at full resolution, 0.5 for OME-Zarr level 1, ...).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .core import Params, sample, solve_surface, unary_costs
from .tifxyz import read_tifxyz


def _grid_diff(P, valid, axis):
    """Derivative of positions along one grid axis: the mean of the forward and backward
    differences that exist (central where both neighbours are valid, one-sided at holes)."""
    lo = tuple(slice(0, -1) if a == axis else slice(None) for a in range(2))
    hi = tuple(slice(1, None) if a == axis else slice(None) for a in range(2))
    step = np.zeros_like(P)
    step[lo] = P[hi] - P[lo]
    ok = np.zeros(valid.shape, bool)
    ok[lo] = valid[hi] & valid[lo]
    fwd, fok = step, ok
    bwd = np.zeros_like(P)
    bwd[hi] = step[lo]
    bok = np.zeros(valid.shape, bool)
    bok[hi] = ok[lo]
    cnt = fok.astype(np.float32) + bok
    d = (fwd * fok[..., None] + bwd * bok[..., None]) / np.maximum(cnt, 1)[..., None]
    return d, cnt > 0


def vertex_normals(P, valid):
    """Unit normals at the vertices: cross(d/d column, d/d row) of the positions, computed in
    (x, y, z) order. Returns (N in (z, y, x) order, ok mask). Vertices without a valid
    neighbour along both grid directions get no normal."""
    dc, okc = _grid_diff(P, valid, 1)
    dr, okr = _grid_diff(P, valid, 0)
    n_xyz = np.cross(dc[..., ::-1], dr[..., ::-1])
    norm = np.linalg.norm(n_xyz, axis=-1)
    ok = valid & okc & okr & (norm > 1e-6)
    N = (n_xyz / np.maximum(norm, 1e-12)[..., None])[..., ::-1].astype(np.float32)
    N[~ok] = 0
    return N, ok


def mean_spacing(P, valid):
    """Median distance between grid neighbours (voxels)."""
    d = []
    for a in (0, 1):
        lo = tuple(slice(0, -1) if b == a else slice(None) for b in range(2))
        hi = tuple(slice(1, None) if b == a else slice(None) for b in range(2))
        ok = valid[lo] & valid[hi]
        d.append(np.linalg.norm(P[hi] - P[lo], axis=-1)[ok])
    d = np.concatenate(d)
    return float(np.median(d)) if len(d) else 1.0


def dense_coords(shape, f):
    """Grid positions (row, col) of the dense samples of an H x W grid upsampled by f."""
    H, W = shape
    a = np.arange((H - 1) * f + 1, dtype=np.float64) / f
    b = np.arange((W - 1) * f + 1, dtype=np.float64) / f
    A, B = np.meshgrid(a, b, indexing="ij")
    return A, B


def densify(P, N, ok, f):
    """Bilinear upsampling of vertex positions and normals by an integer factor f. A dense
    sample is valid when every vertex with weight in its interpolation is valid; normals are
    renormalised (and the sample dropped where neighbouring normals cancel)."""
    A, B = dense_coords(ok.shape, f)
    c = np.stack([A.ravel(), B.ravel()])
    vd = ndi.map_coordinates(ok.astype(np.float32), c, order=1, mode="nearest") > 1 - 1e-4
    Pd = np.stack([ndi.map_coordinates(P[..., k], c, order=1, mode="nearest") for k in range(3)], -1)
    Nd = np.stack([ndi.map_coordinates(N[..., k], c, order=1, mode="nearest") for k in range(3)], -1)
    norm = np.linalg.norm(Nd, axis=-1)
    vd &= norm > 0.5
    Nd = Nd / np.maximum(norm, 1e-12)[:, None]
    shp = A.shape
    return (Pd.reshape(shp + (3,)).astype(np.float32), Nd.reshape(shp + (3,)).astype(np.float32),
            vd.reshape(shp))


def grid_pairs(valid):
    """4-neighbour pairs between valid samples of a grid, as indices into valid samples
    (row-major order), and a connected-component id per valid sample."""
    idx = -np.ones(valid.shape, np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    h = valid[:, :-1] & valid[:, 1:]
    v = valid[:-1, :] & valid[1:, :]
    i = np.concatenate([idx[:, :-1][h], idx[:-1, :][v]])
    j = np.concatenate([idx[:, 1:][h], idx[1:, :][v]])
    lab, _ = ndi.label(valid)
    return i, j, (lab[valid] - 1).astype(np.int64)


def orientation_vote(ct_s, pts, n, reach=4.0, sigma=2.0):
    """Which way the papyrus edge faces near the mesh. Per sample: minus the intensity change
    along n, averaged over +-reach voxels with weights falling off with distance from the mesh
    (positive when intensity drops along +n, i.e. n points from papyrus toward the gap).
    Returns (sign for the whole mesh, fraction of samples agreeing with it)."""
    tt = np.arange(-reach, reach + 1e-3, 0.5, dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + t * n) for t in tt], 1)
    deriv = np.diff(prof, axis=1)
    mid = (tt[1:] + tt[:-1]) / 2
    w = np.exp(-mid ** 2 / (2 * sigma ** 2))
    vote = -(deriv * w).sum(1)
    s = 1 if vote.sum() >= 0 else -1
    return s, float(np.mean(np.sign(vote) == s)) if len(vote) else 0.0


def axis_orientation(P, valid, axis_yx=None):
    """Which way papyrus -> gap points at the recto face, from the scroll's geometry: every
    wrap's recto faces the scroll centre, so there it points toward the scroll axis. The axis is
    taken parallel to z (the scan axis); unless given as (y, x), it is estimated by a circle fit
    to the vertices in (y, x), which works for a segment that covers a good part of a wrap.
    Returns (sign relative to the grid's own normal, fraction of vertices agreeing, (y, x) of the
    axis)."""
    N, ok = vertex_normals(P, valid)
    q, n = P[ok].astype(np.float64), N[ok].astype(np.float64)
    if axis_yx is None:
        Y, X = q[:, 1], q[:, 2]
        A = np.stack([X, Y, np.ones_like(X)], 1)
        (D, E, _), *_ = np.linalg.lstsq(A, -(X ** 2 + Y ** 2), rcond=None)
        axis_yx = (-E / 2, -D / 2)
    to_axis = np.stack([axis_yx[0] - q[:, 1], axis_yx[1] - q[:, 2]], 1)
    toward = (n[:, 1] * to_axis[:, 0] + n[:, 2] * to_axis[:, 1]) > 0
    frac = float(toward.mean()) if len(toward) else 0.5
    sign = 1 if frac >= 0.5 else -1
    return sign, max(frac, 1 - frac), (float(axis_yx[0]), float(axis_yx[1]))


_ORIENT = {}


def patch_orientation(patch_dir, verbose=True):
    """Papyrus -> gap sign at the recto face for a patch made by tools/fetch_segment.py: from
    patch.json if recorded there, else from the whole segment saved next to the patches (see
    axis_orientation). 0 (decide from the CT) if neither is available or the geometry is
    ambiguous (under 90% of vertices agree)."""
    with open(os.path.join(patch_dir, "patch.json")) as f:
        pj = json.load(f)
    if "recto_sign" in pj:
        return int(pj["recto_sign"])
    whole = os.path.join(os.path.dirname(os.path.normpath(patch_dir)), "segment.tifxyz")
    if not os.path.exists(os.path.join(whole, "x.tif")):
        return 0
    if whole not in _ORIENT:
        P, valid, _ = read_tifxyz(whole)
        sign, agree, axis = axis_orientation(P, valid)
        if verbose:
            print(f"orientation from the whole segment: scroll axis near y {axis[0]:.0f}, x {axis[1]:.0f}; "
                  f"recto normal {'= grid normal' if sign > 0 else '= flipped grid normal'} at {agree:.0%} of vertices",
                  flush=True)
        _ORIENT[whole] = sign if agree >= 0.9 else 0
    return _ORIENT[whole]


def fit_vertex_shifts(ok, f, vd, shift_d, lam=1e-3):
    """Per-vertex shifts whose bilinear interpolation onto the dense grid best matches the
    dense shifts (least squares over the valid dense samples; vertices without any get 0)."""
    H, W = ok.shape
    A, B = dense_coords(ok.shape, f)
    a, b, s = A[vd], B[vd], shift_d[vd]
    r0 = np.minimum(np.floor(a).astype(int), max(H - 2, 0))
    c0 = np.minimum(np.floor(b).astype(int), max(W - 2, 0))
    fr, fc = a - r0, b - c0
    rows, cols, vals = [], [], []
    m = np.arange(len(a))
    for dr, dc, wt in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                       (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
        rr, cc = np.minimum(r0 + dr, H - 1), np.minimum(c0 + dc, W - 1)
        keep = wt > 1e-9
        rows.append(m[keep])
        cols.append((rr * W + cc)[keep])
        vals.append(wt[keep])
    Bm = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(len(a), H * W))
    lhs = (Bm.T @ Bm + lam * sparse.identity(H * W)).tocsc()
    out = spsolve(lhs, Bm.T @ s).reshape(H, W)
    out[~ok] = 0
    return out.astype(np.float32)


@dataclass
class MeshPrep:
    """A mesh ready for scoring: dense samples in crop coordinates with oriented normals."""
    ct_s: np.ndarray
    P: np.ndarray            # vertices (H, W, 3), volume coordinates
    ok: np.ndarray           # vertices with a position and a normal
    N: np.ndarray            # vertex normals (H, W, 3), oriented papyrus -> gap
    factor: int              # dense samples per grid step
    Pd: np.ndarray           # dense samples (Hd, Wd, 3), crop coordinates
    Nd: np.ndarray           # their normals
    vd: np.ndarray           # valid dense samples (Hd, Wd)
    i: np.ndarray            # neighbour pairs (indices into valid dense samples)
    j: np.ndarray
    comp: np.ndarray
    normal_sign: int         # +1: papyrus -> gap agrees with the grid's own normal (cross product)
    orientation_agree: float
    spacing: float           # vertex spacing in crop voxels

    @property
    def prepared(self):
        return (self.ct_s, self.Pd[self.vd], self.Nd[self.vd], self.i, self.j, self.comp)


def prepare_mesh(ct, P, valid, p: Params | None = None, offset_zyx=(0, 0, 0), level_scale=1.0,
                 factor=None, normal_sign=0, ct_s=None):
    """Normals, densification, neighbours and orientation. normal_sign: 0 = decide from the CT,
    +1 / -1 = papyrus -> gap is along / against the grid's own normal."""
    p = p or Params()
    if ct_s is None:
        ct_s = ndi.gaussian_filter(ct.astype(np.float32), p.sigma_ct)
    Q = P.astype(np.float64) * level_scale - np.asarray(offset_zyx, np.float64)
    Q = Q.astype(np.float32)
    N, ok = vertex_normals(Q, valid)
    spacing = mean_spacing(Q, ok)
    f = int(factor) if factor else max(1, int(round(spacing / p.spacing)))
    Pd, Nd, vd = densify(Q, N, ok, f)
    agree = float("nan")                     # orientation given, not voted
    if normal_sign == 0:
        normal_sign, agree = orientation_vote(ct_s, Pd[vd], Nd[vd])
    N = N * normal_sign
    Nd = Nd * normal_sign
    i, j, comp = grid_pairs(vd)
    return MeshPrep(ct_s, P, ok, N, f, Pd, Nd, vd, i, j, comp, int(normal_sign), float(agree), spacing)


@dataclass
class MeshSnapResult:
    P: np.ndarray            # snapped vertices on the input grid (H, W, 3), volume coordinates
    valid: np.ndarray        # vertices that exist in the output
    vertex_shift: np.ndarray # shift of each vertex along its normal (volume voxels)
    dense_P: np.ndarray      # snapped dense surface (Hd, Wd, 3), volume coordinates
    dense_valid: np.ndarray
    dense_shift: np.ndarray  # shift along the normal per dense sample (volume voxels; 0 if invalid)
    dense_confidence: np.ndarray  # 0..1
    dense_flag: np.ndarray   # label-on-a-reversed-face flag, as for labels
    dense_limit: np.ndarray  # shift stopped at the edge of the search range: the face may be farther
    factor: int
    info: dict


def snap_mesh(ct, P, valid, p: Params | None = None, offset_zyx=(0, 0, 0), level_scale=1.0,
              factor=None, normal_sign=0, prep: MeshPrep | None = None):
    """Snap a tifxyz grid of vertices P (H, W, 3; volume coordinates, z y x) onto the papyrus
    surface in the CT crop `ct`."""
    p = p or Params()
    prep = prep or prepare_mesh(ct, P, valid, p, offset_zyx, level_scale, factor, normal_sign)
    shift, conf, movable, rev, info = solve_surface(prep.ct_s.shape, prep.prepared, p)
    vd = prep.vd
    shift_d = np.zeros(vd.shape, np.float32)
    shift_d[vd] = shift
    conf_d = np.zeros(vd.shape, np.float32)
    conf_d[vd] = conf
    flag_d = np.zeros(vd.shape, bool)
    flag_d[vd] = rev
    limit_d = np.zeros(vd.shape, bool)
    limit_d[vd] = np.abs(shift) >= p.t_max - 1e-6
    off = np.asarray(offset_zyx, np.float32)
    dense = (prep.Pd + shift_d[..., None] * prep.Nd + off) / level_scale
    v_shift = fit_vertex_shifts(prep.ok, prep.factor, vd, shift_d)
    Pout = P.astype(np.float32) + (v_shift[..., None] * prep.N) / level_scale
    Pout[~prep.ok] = -1
    dense[~vd] = -1
    info.update(factor=prep.factor, vertex_spacing=prep.spacing / level_scale,
                normal_sign=prep.normal_sign, orientation_agree=prep.orientation_agree,
                n_vertices=int(prep.ok.sum()), n_dense=int(vd.sum()),
                frac_at_limit=float(limit_d[vd].mean()) if vd.any() else 0.0,
                mean_abs_vertex_shift=float(np.abs(v_shift[prep.ok]).mean()) if prep.ok.any() else 0.0)
    return MeshSnapResult(Pout, prep.ok, v_shift / level_scale, dense.astype(np.float32), vd,
                          shift_d / level_scale, conf_d, flag_d, limit_d, prep.factor, info)


def trusted_dense(prep: MeshPrep, p: Params):
    """Dense samples where the mesh sits on a strong, unambiguous papyrus edge at its own
    convention (the criterion used for labels in tools/eval_snap.py). Returns a mask over the
    valid dense samples."""
    ts, E, mov, _, _ = unary_costs(prep.ct_s.shape, prep.prepared, p)
    peak = E.max(1)
    amax = np.argmax(E, 1)
    far = np.abs(ts[None, :] - ts[amax][:, None]) >= 2.0
    second = np.where(far, E, -np.inf).max(1)
    return mov & (np.abs(ts[amax]) <= 0.5) & (peak >= 0.8) & (second < 0.5 * peak)
