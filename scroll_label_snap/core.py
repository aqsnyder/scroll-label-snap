"""Snap approximate recto-surface labels onto the papyrus edge in a CT cube.

Inputs per cube: a CT volume (uint8) and a voxel label volume where 1 = recto surface,
2 = not annotated, 0 = annotated background.

Pipeline:
  1. surface_points  - sample the label-1 band on a coarse grid, estimate normals from the
                       Hessian of the smoothed band, and centre each sample on the band.
  2. neighbour_graph - link samples that lie on the same sheet (close together, parallel
                       normals, offset tangentially rather than along the normal).
  3. orient_normals  - make normals consistent along each sheet, then point them from
                       papyrus toward the gap using the CT.
  4. edge_costs      - score each candidate shift along the normal by CT edge strength.
  5. solve_shifts    - choose all shifts jointly and exactly with a graph cut (Ishikawa
                       construction): sum of edge costs + w * |shift_p - shift_q| between
                       neighbours, plus a hard limit on the shift difference.
  6. warp_labels     - move the label-1 band by the solved shift field.

A port of volume-cartographer's ObjAlphaCompRefinement (`alphacomp_shifts`) is included
as a baseline.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
import maxflow

SURF, UNK = 1, 2


@dataclass
class Params:
    sigma_band: float = 1.5       # smoothing of the label band for normals and centring
    sigma_ct: float = 1.0         # CT smoothing before measuring edges
    spacing: float = 3.0          # surface sample spacing (voxels)
    t_max: float = 4.0            # shift search range, each way (voxels)
    t_step: float = 0.5           # shift resolution (voxels)
    edge_halfwidth: float = 1.5   # edge strength = ct(t - h) - ct(t + h) along the normal
    smooth_w: float = 0.25        # L1 smoothness weight per neighbour pair, per voxel
    max_step: float = 1.0         # hard limit on |shift_p - shift_q| between neighbours
    prior_mu: float = 0.02        # pull toward the original label, per voxel of shift
    template_weight: float = 2.0  # weight of the learned-profile match (0 = step edge only)
    # Experimental (off by default; the published 20-cube evaluation used neither):
    calibrate: bool = False       # remove any cube-wide offset so the median correction is 0
    stay_bonus: float = 0.0       # score bonus for keeping the original position ("do no harm")
    neigh_radius: float = 3.6     # neighbour search radius (voxels)
    tangential_tol: float = 1.0   # max offset along the normal for same-sheet neighbours
    normal_dot: float = 0.8       # min |n_p . n_q| for same-sheet neighbours
    border: int = 8               # samples this close to the cube edge are not moved
    convention_offset: float | None = None  # label centre relative to the edge (None = estimate)
    convention_reach: float = 4.0  # how far from the label to look for the edge when estimating it
    batch_points: int = 60000     # max samples per graph-cut problem
    smooth_warp: bool = True      # blend displacements and resample linearly (False = v1 warp)
    inside_penalty: float = 2.0   # cost of moving a label outward (+n) to a spot that has
                                  # papyrus on both sides, i.e. into a sheet (0 = v1)
    inside_margin: float = 0.1    # 'papyrus' = brighter than the local mid level by this
                                  # fraction of the papyrus-gap contrast
    # Reversed-face labels sit on a clear face of the opposite polarity (gap on the -n side)
    # with no normal-polarity face in range. They are always flagged (SnapResult.reversed_face);
    # face_polarity=True also snaps them onto their own face instead of searching along +n.
    # Off by default: the blind reviews preferred the +n move about as often as not.
    face_polarity: bool = False
    reverse_min: float = 0.5      # own face at least this strong (normalised edge units)
    forward_max: float = 0.5      # and no candidate with a normal-polarity edge this strong

    def asdict(self):
        return asdict(self)


def sample(vol, pts, order=1):
    """Interpolate vol at float points (N, 3) given in (z, y, x) order."""
    return ndi.map_coordinates(vol, pts.T, order=order, mode="nearest")


# --------------------------------------------------------------------------- 1. points

def surface_points(lab, p: Params):
    """Band-centred samples of the label-1 surface with (unoriented) unit normals."""
    m = (lab == SURF).astype(np.float32)
    s = ndi.gaussian_filter(m, p.sigma_band)
    zz, yy, xx = np.nonzero(lab == SURF)
    if len(zz) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), s
    coords = np.stack([zz, yy, xx], 1)
    sval = s[zz, yy, xx]

    # keep the voxel nearest the band centre (largest smoothed value) in each grid cell
    cell = (coords // p.spacing).astype(np.int64)
    key = (cell[:, 0] * 4096 + cell[:, 1]) * 4096 + cell[:, 2]
    order = np.lexsort((-sval, key))
    ks = key[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    sel = order[first]
    pts = coords[sel].astype(np.float32)

    # normals: eigenvector of the most negative Hessian eigenvalue of the smoothed band
    orders = {(0, 0): (2, 0, 0), (1, 1): (0, 2, 0), (2, 2): (0, 0, 2),
              (0, 1): (1, 1, 0), (0, 2): (1, 0, 1), (1, 2): (0, 1, 1)}
    H = np.empty((len(sel), 3, 3), np.float32)
    z, y, x = coords[sel].T
    for (a, b), o in orders.items():
        v = ndi.gaussian_filter(m, p.sigma_band, order=o)[z, y, x]
        H[:, a, b] = v
        H[:, b, a] = v
    _, V = np.linalg.eigh(H)
    n = V[:, :, 0].astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    # centre each sample on the band along its normal
    tc = np.arange(-2.0, 2.001, 0.25, dtype=np.float32)
    sv = np.stack([sample(s, pts + t * n) for t in tc], 1)
    pts = pts + tc[np.argmax(sv, 1)][:, None] * n
    return pts.astype(np.float32), n, s


# --------------------------------------------------------------------------- 2. graph

def neighbour_graph(pts, n, p: Params):
    """Pairs (i, j) of samples that lie on the same sheet."""
    if len(pts) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = cKDTree(pts).query_pairs(p.neigh_radius, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pts[j] - pts[i]
    ok = ((np.abs(np.einsum("ij,ij->i", d, n[i])) < p.tangential_tol) &
          (np.abs(np.einsum("ij,ij->i", d, n[j])) < p.tangential_tol) &
          (np.abs(np.einsum("ij,ij->i", n[i], n[j])) > p.normal_dot))
    return i[ok], j[ok]


# --------------------------------------------------------------------------- 3. orient

def orient_normals(pts, n, i, j, ct_s):
    """Flip normals so they agree along each sheet and point from papyrus toward the gap.

    Returns (oriented normals, component id per sample)."""
    N = len(pts)
    A = sparse.coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(N, N)).tocsr()
    A = (A + A.T).tocsr()
    indptr, indices = A.indptr, A.indices
    sign = np.zeros(N, np.int8)
    comp = np.full(N, -1, np.int64)
    n_list = n.tolist()
    c = 0
    for root in range(N):
        if sign[root] != 0:
            continue
        sign[root] = 1
        comp[root] = c
        dq = deque([root])
        while dq:
            u = dq.popleft()
            nu = n_list[u]
            su = sign[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                if sign[v] == 0:
                    nv = n_list[v]
                    dot = nu[0] * nv[0] + nu[1] * nv[1] + nu[2] * nv[2]
                    sign[v] = su if dot >= 0 else -su
                    comp[v] = c
                    dq.append(v)
        c += 1
    n = n * sign[:, None]

    # per sheet: papyrus (bright) should be on the -n side. Judge each sample by the
    # strongest intensity step within 1.5 voxels of the label (the face the label is
    # on), not by sampling far away, which reaches through thin sheets.
    tt = np.arange(-1.5, 1.501, 0.5, dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + t * n) for t in tt], 1)
    deriv = np.diff(prof, axis=1)                    # intensity change toward +n
    k = np.argmax(np.abs(deriv), 1)
    vote = -deriv[np.arange(len(pts)), k]            # > 0 when intensity drops toward +n
    comp_vote = np.bincount(comp, weights=vote, minlength=c)
    flip = comp_vote[comp] < 0
    n[flip] *= -1

    # cross-sheet check: every wrap's recto faces the scroll centre, so nearby sheets
    # should point the same way. Flip any sheet that disagrees with the samples of the
    # other sheets around it (mostly small fragments with a weak vote of their own).
    if c > 1:
        sub = np.arange(0, N, 5)
        neighbours = cKDTree(pts).query_ball_point(pts[sub], r=12.0)
        agree_sum = np.zeros(c)
        agree_cnt = np.zeros(c)
        for a, lst in zip(sub, neighbours):
            lst = np.asarray(lst)
            lst = lst[comp[lst] != comp[a]]
            if len(lst) < 5:
                continue
            agree_sum[comp[a]] += np.mean(n[lst] @ n[a] > 0)
            agree_cnt[comp[a]] += 1
        ratio = agree_sum / np.maximum(agree_cnt, 1)
        bad = (agree_cnt > 0) & (ratio < 0.5)
        n[bad[comp]] *= -1
    return n.astype(np.float32), comp


# --------------------------------------------------------------------------- 4. costs

def edge_strength(ct_s, pts, n, ts, h, wide=3.5):
    """Papyrus-to-gap step at each candidate position t_k along the normal.

    Two measurements must agree (the smaller is used):
      narrow: ct(t - h) - ct(t + h)                           -> where the edge is
      wide:   mean ct over [t - wide, t - 0.5] minus mean over [t + 0.5, t + wide]
              -> is there papyrus behind and a real gap in front? A thin crack inside a
                 sheet has a sharp narrow step but papyrus again just beyond it.
    """
    offs = np.arange(-wide - h, wide + h + 1e-6, 0.5, dtype=np.float32)
    lo_t, hi_t = float(ts.min()), float(ts.max())
    grid = np.arange(lo_t + offs[0], hi_t + offs[-1] + 1e-6, 0.5, dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + g * n) for g in grid], 1)

    def at(x):
        """profile value at offset(s) x (0.5-voxel grid, linear interpolation)."""
        f = (np.asarray(x, np.float32) - grid[0]) / 0.5
        i0 = np.clip(np.floor(f).astype(int), 0, len(grid) - 2)
        w = np.clip(f - i0, 0, 1)
        return prof[:, i0] * (1 - w) + prof[:, i0 + 1] * w

    E = np.empty((len(pts), len(ts)), np.float32)
    back = np.arange(-wide, -0.5 + 1e-6, 0.5, dtype=np.float32)
    front = np.arange(0.5, wide + 1e-6, 0.5, dtype=np.float32)
    for k, t in enumerate(ts):
        narrow = at(t - h) - at(t + h)
        wide_step = at(t + back).mean(1) - at(t + front).mean(1)
        E[:, k] = np.minimum(narrow, wide_step)
    return E


def template_scores(ct_s, pts, n, ts, ref, half=4.0, step=0.5, chunk=50000, tmpl=None):
    """How well the CT profile around each candidate position matches this cube's own
    'typical label' profile (normalised cross-correlation, -1..1).

    The template is learned from the data: the mean CT profile across the samples in
    `ref` (a boolean mask of trusted samples), aligned at their label positions, then
    refined once by re-aligning each sample at its best match. It captures the labelling
    convention (where the band sits relative to papyrus and gap), typical sheet thickness
    and gap width, which a plain step detector does not."""
    n_u = int(round(2 * half / step)) + 1
    t0 = float(ts.min())
    n_g = int(round((float(ts.max()) - t0 + 2 * half) / step)) + 1
    grid = (t0 - half + step * np.arange(n_g)).astype(np.float32)
    k_t = np.rint((ts - t0) / step).astype(int)           # window start index per candidate
    k0 = int(np.argmin(np.abs(ts)))

    def windows(sel):
        P = np.stack([sample(ct_s, pts[sel] + g * n[sel]) for g in grid], 1)
        W = np.lib.stride_tricks.sliding_window_view(P, n_u, axis=1)[:, k_t, :]
        W = W - W.mean(-1, keepdims=True)
        return W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-6)

    if tmpl is None:
        ref_idx = np.flatnonzero(ref)
        if len(ref_idx) > 20000:
            ref_idx = np.random.default_rng(0).choice(ref_idx, 20000, replace=False)
        Wr = windows(ref_idx)
        tmpl = Wr[:, k0, :].mean(0)
        tmpl /= np.linalg.norm(tmpl) + 1e-6
        best = np.argmax(Wr @ tmpl, 1)
        best = np.clip(best - int(np.median(best)) + k0, 0, len(ts) - 1)   # keep it anchored
        tmpl = Wr[np.arange(len(Wr)), best, :].mean(0)
        tmpl /= np.linalg.norm(tmpl) + 1e-6

    S = np.empty((len(pts), len(ts)), np.float32)
    for a in range(0, len(pts), chunk):
        S[a:a + chunk] = windows(np.arange(a, min(a + chunk, len(pts)))) @ tmpl
    return S, tmpl


def inside_sheet(ct_s, pts, n, ts, c_off, margin=0.1, reach=2.5, thr=None):
    """True where candidate t puts the label inside papyrus: the mean CT over
    [t-reach, t-0.5] and over [t+0.5, t+reach] (around the expected edge, t + c_off) are
    both above the papyrus/gap mid level. At a real face one side is gap."""
    grid = np.arange(float(ts.min()) + c_off - reach, float(ts.max()) + c_off + reach + 1e-6, 0.5,
                     dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + g * n) for g in grid], 1)
    if thr is None:
        near = prof[:, np.abs(grid - c_off) <= 3.0]
        thr = float(np.median(near)) + margin * float(np.percentile(near, 90) - np.percentile(near, 10))
    offs = np.arange(0.5, reach + 1e-6, 0.5, dtype=np.float32)

    def mean_at(x):
        k = np.clip(np.rint((x - grid[0]) / 0.5).astype(int), 0, len(grid) - 1)
        return prof[:, k].mean(1)

    out = np.zeros((len(pts), len(ts)), bool)
    for k, t in enumerate(ts):
        e = t + c_off
        out[:, k] = (mean_at(e - offs) > thr) & (mean_at(e + offs) > thr)
    return out, thr


def estimate_convention(ct_s, pts, n, p: Params):
    """Where the labels put the band centre relative to the steepest edge, and the
    typical edge strength. Returns (offset, scale)."""
    tf = np.arange(-p.convention_reach, p.convention_reach + 0.001, 0.25, dtype=np.float32)
    E = edge_strength(ct_s, pts, n, tf, p.edge_halfwidth)
    peak = E.max(1)
    strong = peak > np.percentile(peak, 50)
    offset = float(np.median(tf[np.argmax(E[strong], 1)])) if strong.any() else 0.0
    scale = float(np.median(peak[strong])) if strong.any() else 1.0
    return offset, max(scale, 1e-3)


# --------------------------------------------------------------------------- 5. solve

def solve_shifts(D, i, j, w, K):
    """Exact minimiser of sum_p D[p, l_p] + w * sum_(p,q) |l_p - l_q|, with |l_p - l_q| <= K
    (K = None disables the hard limit), via the Ishikawa graph construction."""
    N, L = D.shape
    if N == 0:
        return np.zeros(0, int)
    Dn = (D - D.min(1, keepdims=True)).astype(np.float64)
    big = float(Dn.max() * N + w * len(i) * L + 1.0)
    g = maxflow.Graph[float]()
    g.add_nodes(N * (L - 1))
    ids = np.arange(N * (L - 1), dtype=np.int64).reshape(N, L - 1)

    src = np.zeros((N, L - 1))
    snk = np.zeros((N, L - 1))
    src[:, 0] = Dn[:, 0]
    snk[:, -1] = Dn[:, -1]
    g.add_grid_tedges(ids, src, snk)
    if L > 2:
        g.add_edges(ids[:, :-1].ravel(), ids[:, 1:].ravel(),
                    Dn[:, 1:-1].ravel(), np.full((L - 2) * N, big))
    if len(i):
        ii = ids[i].ravel()
        jj = ids[j].ravel()
        g.add_edges(ii, jj, np.full(len(ii), float(w)), np.full(len(ii), float(w)))
        if K is not None and K < L - 1:
            a = ids[i][:, K:].ravel()
            b = ids[j][:, :-K].ravel() if K > 0 else ids[j].ravel()
            g.add_edges(a, b, np.full(len(a), big), np.zeros(len(a)))
            a = ids[j][:, K:].ravel()
            b = ids[i][:, :-K].ravel() if K > 0 else ids[i].ravel()
            g.add_edges(a, b, np.full(len(a), big), np.zeros(len(a)))
    g.maxflow()
    in_sink = g.get_grid_segments(ids)
    return (~in_sink).sum(1)


def _batches(comp, size):
    """Group connected components into batches of at most `size` samples."""
    counts = np.bincount(comp)
    order = np.argsort(-counts)
    batches, cur, cur_n = [], [], 0
    for c in order:
        if cur and cur_n + counts[c] > size:
            batches.append(cur)
            cur, cur_n = [], 0
        cur.append(c)
        cur_n += counts[c]
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------- 6. warp

def warp_labels(lab, pts, n, shift, radius=6, chunk=1_000_000, comp=None, smooth=True,
                sigma=None, k=12):
    """Move the label-1 band so the surface near sample p shifts by shift[p] * n[p].
    Label-2 voxels are never changed.

    smooth=True (default): each voxel's displacement is a Gaussian-weighted blend of the
    displacements of nearby samples on the same sheet (sigma defaults to the sample
    spacing), and the band is resampled with linear interpolation. The moved band then has
    clean edges like the original. smooth=False reproduces v1: each voxel takes its nearest
    sample's displacement and the band is resampled to the nearest voxel, which leaves
    stair steps wherever neighbouring samples moved by different amounts."""
    band = lab == SURF
    out = lab.copy()
    if len(pts) == 0:
        return out
    near = ndi.binary_dilation(band, iterations=radius)
    xs = np.argwhere(near)
    tree = cKDTree(pts)
    new_band = np.zeros_like(band)
    hi = np.array(lab.shape) - 1
    vec = (np.asarray(shift, np.float32)[:, None] * n).astype(np.float32)
    if smooth:
        sigma = float(sigma) if sigma is not None else _sample_spacing(pts)
        bandf = band.astype(np.float32)
        comp = np.zeros(len(pts), np.int64) if comp is None else np.asarray(comp)
        kk = min(k, len(pts))
    for a in range(0, len(xs), chunk):
        x = xs[a:a + chunk]
        if not smooth:
            _, idx = tree.query(x, k=1)
            src = np.clip(np.rint(x - vec[idx]).astype(np.int64), 0, hi)
            new_band[x[:, 0], x[:, 1], x[:, 2]] = band[src[:, 0], src[:, 1], src[:, 2]]
            continue
        d, idx = tree.query(x, k=kk, workers=_kd_workers())
        if kk == 1:
            d, idx = d[:, None], idx[:, None]
        w = np.exp(-0.5 * (d / sigma) ** 2)
        w *= comp[idx] == comp[idx[:, :1]]                 # same sheet as the nearest sample
        disp = np.einsum("ik,ikc->ic", w, vec[idx]) / w.sum(1, keepdims=True)
        src = np.clip(x - disp, 0, hi).T
        vals = ndi.map_coordinates(bandf, src, order=1, mode="nearest")
        new_band[x[:, 0], x[:, 1], x[:, 2]] = vals >= 0.5
    annotated = lab != UNK
    out[band & annotated] = 0
    out[new_band & annotated] = SURF
    return out


def _kd_workers():
    """Threads for k-d tree queries: all cores by default; SNAP_THREADS=1 when the caller
    already runs one process per core (tools/snap_folder.py, tools/eval_folder.py)."""
    try:
        return int(os.environ.get("SNAP_THREADS", "-1"))
    except ValueError:
        return -1


def _sample_spacing(pts):
    """Typical distance between neighbouring samples."""
    if len(pts) < 2:
        return 1.0
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 5000), replace=False)]
    d, _ = cKDTree(pts).query(sub, k=2)
    return float(np.median(d[:, 1]))


def shift_volume(lab_shape, mask, pts, values, fill=0):
    """Paint per-sample values onto the voxels of `mask` (nearest sample)."""
    vol = np.full(lab_shape, fill, dtype=np.asarray(values).dtype)
    if len(pts) == 0:
        return vol
    xs = np.argwhere(mask)
    _, idx = cKDTree(pts).query(xs, k=1)
    vol[xs[:, 0], xs[:, 1], xs[:, 2]] = np.asarray(values)[idx]
    return vol


# --------------------------------------------------------------------------- driver

@dataclass
class SnapResult:
    labels: np.ndarray           # snapped label volume
    pts: np.ndarray              # surface samples (z, y, x)
    normals: np.ndarray          # oriented normals (papyrus -> gap)
    shift: np.ndarray            # chosen shift per sample (voxels along the normal)
    confidence: np.ndarray       # 0..1 per sample
    movable: np.ndarray          # samples that were optimised (not at the border)
    info: dict
    reversed_face: np.ndarray = None   # label on a face of the opposite polarity: flag for review


def prepare(img, lab, p: Params):
    """Steps 1-3: samples, neighbours, oriented normals, smoothed CT."""
    ct_s = ndi.gaussian_filter(img.astype(np.float32), p.sigma_ct)
    pts, n, _ = surface_points(lab, p)
    i, j = neighbour_graph(pts, n, p)
    n, comp = orient_normals(pts, n, i, j, ct_s)
    return ct_s, pts, n, i, j, comp


def unary_costs(lab_shape, prepared, p: Params):
    """Edge-based cost of each candidate shift for each sample, plus bookkeeping."""
    ct_s, pts, n, i, j, comp = prepared
    ts = np.arange(-p.t_max, p.t_max + 1e-6, p.t_step, dtype=np.float32)
    lo = p.border
    hi = np.array(lab_shape) - 1 - p.border
    movable = np.all((pts >= lo) & (pts <= hi), axis=1)
    c_off, scale = estimate_convention(ct_s, pts[movable], n[movable], p)
    if p.convention_offset is not None:
        c_off = p.convention_offset
    E = edge_strength(ct_s, pts, n, ts + c_off, p.edge_halfwidth) / scale
    return ts, E, movable, c_off, scale


def reversed_faces(ct_s, pts, n, E, c_off, scale, p: Params):
    """Samples whose label sits on a clear face of the opposite polarity (papyrus on the +n
    side, gap on the -n side) while no candidate along the normal shows a face of the
    expected polarity. There the label is probably on the far face of its sheet, or the
    normal is flipped; searching along +n would only push it into the papyrus."""
    near = np.arange(-1.5, 1.501, 0.5, dtype=np.float32)
    e_rev = edge_strength(ct_s, pts, -n, near + c_off, p.edge_halfwidth).max(1) / scale
    return (e_rev >= p.reverse_min) & (E.max(1) < p.forward_max)


def unary_scores(lab_shape, prepared, p: Params, extras=None):
    """Score of each candidate shift (higher = more like a correctly placed label):
    normalised step-edge strength + template_weight * learned-profile match.

    With p.face_polarity, samples found by reversed_faces() are scored along -n instead, so
    they settle on the face they are on. All scores stay indexed by the shift along +n.
    If `extras` is a dict, the reversed-face mask is stored in it as 'reversed'."""
    ts, E, movable, c_off, scale = unary_costs(lab_shape, prepared, p)
    ct_s, pts, n = prepared[0], prepared[1], prepared[2]
    S = np.zeros_like(E)
    ref, tmpl = None, None
    if p.template_weight > 0 and movable.sum() > 100:
        peak = E.max(1)
        ref = movable & (peak >= np.percentile(peak[movable], 50))
        S, tmpl = template_scores(ct_s, pts, n, ts, ref)
    rev = np.zeros(len(pts), bool)
    if len(pts):
        rev = reversed_faces(ct_s, pts, n, E, c_off, scale, p) & movable
        if p.face_polarity and rev.any():
            # ts is symmetric, so a shift t along -n is candidate -t along +n: reverse the axis
            E[rev] = (edge_strength(ct_s, pts[rev], -n[rev], ts + c_off, p.edge_halfwidth) / scale)[:, ::-1]
            if tmpl is not None:
                S[rev] = template_scores(ct_s, pts[rev], -n[rev], ts, None, tmpl=tmpl)[0][:, ::-1]
    if extras is not None:
        extras["reversed"] = rev
    U = E + p.template_weight * S

    # Self-calibration: assume the annotators are right on average, so the median
    # preferred correction over the cube should be zero. If the scores prefer a
    # consistent offset (e.g. phase-contrast halos or blur shift the apparent edge in
    # this scan), evaluate every candidate that much further along instead.
    if p.calibrate and movable.sum() > 100:
        d_med = float(np.median(ts[np.argmax(U[movable], 1)]))
        if abs(d_med) >= p.t_step:
            E = edge_strength(ct_s, pts, n, ts + c_off + d_med, p.edge_halfwidth) / scale
            if ref is not None:
                S, _ = template_scores(ct_s, pts, n, ts + d_med, ref)
            U = E + p.template_weight * S
            c_off += d_med
    return ts, U, E, S, movable, c_off, scale


def solve_all(D, i, j, comp, p: Params):
    """Run the graph cut per batch of sheets; returns the chosen level per sample."""
    w_level = p.smooth_w * p.t_step
    K = None if p.max_step is None else int(round(p.max_step / p.t_step))
    labels = np.zeros(len(D), int)
    for batch in _batches(comp, p.batch_points):
        sel = np.isin(comp, batch)
        idx = np.flatnonzero(sel)
        remap = -np.ones(len(D), np.int64)
        remap[idx] = np.arange(len(idx))
        e_sel = sel[i] & sel[j]
        labels[idx] = solve_shifts(D[idx], remap[i[e_sel]], remap[j[e_sel]], w_level, K)
    return labels


def snap(img, lab, p: Params | None = None, prepared=None):
    p = p or Params()
    prepared = prepared if prepared is not None else prepare(img, lab, p)
    ct_s, pts, n, i, j, comp = prepared
    shift, conf, movable, rev, info = solve_surface(lab.shape, prepared, p)
    out = warp_labels(lab, pts, n, shift, comp=comp, smooth=p.smooth_warp)
    return SnapResult(out, pts, n, shift, conf, movable, info, rev)


def solve_surface(shape, prepared, p: Params):
    """Steps 4-5 for any oriented surface samples (from labels or from a mesh): score the
    candidate shifts and choose them all at once. `shape` is the CT volume's shape; samples
    within p.border of its edge stay put. Returns (shift, confidence, movable, reversed, info)."""
    ct_s, pts, n, i, j, comp = prepared
    extras = {}
    ts, U, E, S, movable, c_off, scale = unary_scores(shape, prepared, p, extras)
    rev = extras["reversed"]
    D = -U + p.prior_mu * np.abs(ts)[None, :]
    k0 = int(np.argmin(np.abs(ts)))
    D[:, k0] -= p.stay_bonus
    if p.inside_penalty > 0:
        # +n points from papyrus toward the gap, so moving a label outward should never
        # end with papyrus on both sides of it. When it would, the normal is probably
        # flipped there or the label sits on the far face of a sheet; don't push it in.
        inside, thr = inside_sheet(ct_s, pts, n, ts, c_off, p.inside_margin)
        inside[:, ts <= 0] = False
        if p.face_polarity and rev.any():
            # for reversed-face samples "outward" is -n: same test in their own frame
            inside_r, _ = inside_sheet(ct_s, pts[rev], -n[rev], ts, c_off, p.inside_margin, thr=thr)
            inside_r[:, ts <= 0] = False
            inside[rev] = inside_r[:, ::-1]
        D = D + p.inside_penalty * inside
    # border samples: forced to stay put
    D[~movable] = 1e3
    D[~movable, k0] = 0.0

    labels = solve_all(D, i, j, comp, p)
    shift = ts[labels]
    rows = np.arange(len(pts))
    # confidence: how label-like the CT profile is at the chosen position (template
    # match), falling back to edge strength when the template is off
    conf = np.clip(S[rows, labels] if p.template_weight > 0 else E[rows, labels], 0, 1)
    info = dict(n_samples=int(len(pts)), n_pairs=int(len(i)), n_sheets=int(comp.max() + 1) if len(comp) else 0,
                convention_offset=float(c_off), edge_scale=float(scale),
                mean_abs_shift=float(np.mean(np.abs(shift[movable]))) if movable.any() else 0.0,
                frac_moved_1vox=float(np.mean(np.abs(shift[movable]) >= 1.0)) if movable.any() else 0.0,
                frac_reversed_face=float(np.mean(rev[movable])) if movable.any() else 0.0,
                params=p.asdict())
    return shift, conf, movable, rev, info


# --------------------------------------------------------------------------- baseline

def alphacomp_shifts(img, pts, n, start=-2.0, stop=30.0, step=2.0,
                     low=118 / 255, high=165 / 255, border_off=1.0, sigma=1.4, vol=None):
    """Port of volume-cartographer ObjAlphaCompRefinement: march into the material,
    turn blurred intensity into opacity with fixed thresholds, and move to the expected
    first-hit depth plus `border_off`. VC's normal points into the sheet, which is -n
    here, so the returned shift is along +n (papyrus -> gap)."""
    if vol is None:                     # the blurred volume can be passed in when calling repeatedly
        vol = ndi.gaussian_filter(img.astype(np.float32) / 255.0, sigma)
    transparent = np.ones(len(pts), np.float32)
    integ = np.zeros(len(pts), np.float32)
    off = start
    while off * np.sign(step) <= stop * np.sign(step) + 1e-6:
        v = sample(vol, pts - off * n)
        o = np.clip((v - low) / (high - low), 0, 1)
        joint = transparent * o
        integ += joint * off
        transparent -= joint
        off += step
    depth = integ / (1 - transparent + 1e-5)
    return -(depth + border_off)
