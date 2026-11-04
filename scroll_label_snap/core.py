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
import sys
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
    calibrate: bool = False       # remove any cube-wide offset so the median correction is 0 (this
                                  # also cancels a given convention_offset, so the tools refuse the pair)
    stay_bonus: float = 0.0       # score bonus for keeping the original position ("do no harm")
    neigh_radius: float = 3.6     # neighbour search radius (voxels)
    tangential_tol: float = 1.0   # max offset along the normal for same-sheet neighbours
    normal_dot: float = 0.8       # min |n_p . n_q| for same-sheet neighbours
    border: int = 8               # samples this close to the cube edge are not moved
    convention_offset: float | None = None  # the edge relative to the label centre, along +n (None = estimate)
    template_follows_offset: bool = True  # read the learned profile at convention_offset too, not only the step edge
    convention_reach: float = 4.0  # how far from the label to look for the edge when estimating it
    batch_points: int = 60000     # no longer used: each sheet is solved on its own (kept so saved params compare)
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
    # Move the band only by flipping simple points, so it keeps the input's components, tunnels
    # and cavities and separate sheets never touch. Off by default: changes the published output.
    # A mesh has no band to walk, so it doesn't use this.
    keep_topology: bool = False
    # Orient label samples by one axis per cube instead of sheet against neighbouring sheet (see
    # orient_normals). Off by default: changes the published output, and is being tested on
    # fresh cubes (results/orient_test_plan.md). Meshes are not oriented this way, so they don't use it.
    orient_cube: bool = False

    def asdict(self):
        return asdict(self)


def parse_params(pairs):
    """--param NAME=VALUE strings as overrides for Params, each converted to its field's type. A bad one stops
    the tool with a message, before any work starts."""
    fields = Params().asdict()
    out = {}
    for kv in pairs:
        k, eq, v = kv.partition("=")
        if not eq:
            sys.exit(f"--param {kv}: expected NAME=VALUE")
        if k not in fields:
            sys.exit(f"unknown parameter {k}")
        if isinstance(fields[k], bool):
            if v.lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
                sys.exit(f"--param {kv}: {k} is true or false (1, true, yes, on or 0, false, no, off)")
            out[k] = v.lower() in ("1", "true", "yes", "on")
        else:
            kind = type(fields[k] or 0.0)
            try:
                out[k] = kind(v)
            except ValueError:
                sys.exit(f"--param {kv}: {k} is {'a whole number' if kind is int else 'a number'}")
    if out.get("calibrate") and out.get("convention_offset") is not None:
        sys.exit("calibrate=true replaces a given convention_offset with the median correction, so the two can't be combined")
    return out


def sample(vol, pts, order=1):
    """Interpolate vol at float points (N, 3) given in (z, y, x) order."""
    return ndi.map_coordinates(vol, pts.T, order=order, mode="nearest")


# --------------------------------------------------------------------------- 1. points

def _filter_x_at(vol, z, y, x, sigma, order):
    """gaussian_filter1d(vol, sigma, axis=2, order=order)[z, y, x], without filtering the rest of
    the volume: the same filter runs on just the line segments around those voxels."""
    r = int(4.0 * float(sigma) + 0.5)                      # scipy's default radius (truncate=4)
    q = (x[:, None] + np.arange(-r, r + 1)) % (2 * vol.shape[2])
    q = np.where(q >= vol.shape[2], 2 * vol.shape[2] - 1 - q, q)     # mode="reflect"
    return ndi.gaussian_filter1d(vol[z[:, None], y[:, None], q], sigma, axis=1, order=order)[:, r]


def surface_points(lab, p: Params):
    """Band-centred samples of the label-1 surface with (unoriented) unit normals."""
    sb = p.sigma_band
    band = lab == SURF
    m = band.astype(np.float32)
    gz = ndi.gaussian_filter1d(m, sb, axis=0)              # smoothing passes, axis by axis
    gzy = ndi.gaussian_filter1d(gz, sb, axis=1)            # as ndi.gaussian_filter applies them
    s = ndi.gaussian_filter1d(gzy, sb, axis=2)
    zz, yy, xx = np.nonzero(band)
    if len(zz) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), s
    sval = s[zz, yy, xx]

    # keep the voxel nearest the band centre (largest smoothed value) in each grid cell
    key = ((zz // p.spacing).astype(np.int64) * 4096 + (yy // p.spacing).astype(np.int64)) * 4096
    key += (xx // p.spacing).astype(np.int64)
    order = np.lexsort((-sval, key))
    ks = key[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    sel = order[first]
    z, y, x = zz[sel], yy[sel], xx[sel]
    pts = np.stack([z, y, x], 1).astype(np.float32)
    del zz, yy, xx, sval, key, order, ks, first

    # normals: eigenvector of the most negative Hessian eigenvalue of the smoothed band. The six
    # second-derivative filters share their first passes, and the last pass runs only at the samples.
    H = np.empty((len(sel), 3, 3), np.float32)

    def put(a, b, vol, order):
        v = _filter_x_at(vol, z, y, x, sb, order)
        H[:, a, b] = v
        H[:, b, a] = v

    put(2, 2, gzy, 2)                                                # orders (0, 0, 2)
    del gzy
    put(1, 1, ndi.gaussian_filter1d(gz, sb, axis=1, order=2), 0)     # (0, 2, 0)
    put(1, 2, ndi.gaussian_filter1d(gz, sb, axis=1, order=1), 1)     # (0, 1, 1)
    del gz
    d = ndi.gaussian_filter1d(m, sb, axis=0, order=1)
    put(0, 1, ndi.gaussian_filter1d(d, sb, axis=1, order=1), 0)      # (1, 1, 0)
    put(0, 2, ndi.gaussian_filter1d(d, sb, axis=1), 1)               # (1, 0, 1)
    d = ndi.gaussian_filter1d(m, sb, axis=0, order=2)
    put(0, 0, ndi.gaussian_filter1d(d, sb, axis=1), 0)               # (2, 0, 0)
    del d, m
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

def orient_normals(pts, n, i, j, ct_s, orient_cube=False):
    """Flip normals so they agree along each sheet and point from papyrus toward the gap.
    The last step is across sheets: against neighbouring sheets, or with orient_cube=True by one
    axis for the whole cube.

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

    # orient_cube: every wrap's recto faces the scroll centre, so away from the centre all the
    # sheets of a cube lie along one axis and point the same way. One sheet's vote can be nearly
    # even or made by a small part of it, so each sheet takes the side of the axis it lies on, and
    # the votes of all sheets together only choose which way the axis points. Sheets across the
    # axis (mean along it under 0.3) keep their own vote.
    if c > 1 and orient_cube:
        w, v = np.linalg.eigh(n.astype(np.float64).T @ n.astype(np.float64))
        # A guard for cubes at the scroll centre, where the normals have no single axis and one sign
        # for the whole cube would be wrong: below this share of the normals' squared projection on
        # the axis, every sheet keeps its own vote. 0.7 was put between phantoms (0.675 with the
        # axis 120 voxels from the cube centre) and the smallest share on the development cubes
        # (0.797). It was not tuned, and no real cube near the centre has tested it.
        if w[-1] >= 0.7 * w.sum():
            along = np.bincount(comp, weights=n @ v[:, -1], minlength=c) / np.bincount(comp, minlength=c)
            if np.sign(along) @ np.abs(comp_vote) < 0:
                along = -along
            n[(along <= -0.3)[comp]] *= -1

    # cross-sheet check: every wrap's recto faces the scroll centre, so nearby sheets
    # should point the same way. Flip any sheet that disagrees with the samples of the
    # other sheets around it (mostly small fragments with a weak vote of their own).
    elif c > 1:
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

def _rays(ct_s, pts, n):
    """profile(grid) = np.stack([sample(ct_s, pts + g * n) for g in grid], 1), each offset sampled once:
    the scoring steps ask for heavily overlapping offsets (159 columns, on 38 distinct offsets, on sample_00007 and sample_00015).
    A subset of the samples is profile(grid)[rows], and looking along -n is profile(-grid)."""
    cols = {}

    def profile(grid):
        for g in grid:
            if float(g) not in cols:
                cols[float(g)] = sample(ct_s, pts + g * n)
        return np.stack([cols[float(g)] for g in grid], 1)
    return profile


def edge_strength(ct_s, pts, n, ts, h, wide=3.5, rays=None):
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
    prof = (rays or _rays(ct_s, pts, n))(grid)

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


def template_scores(ct_s, pts, n, ts, ref, half=4.0, step=0.5, chunk=50000, tmpl=None, rays=None):
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
    prof = (rays or _rays(ct_s, pts, n))(grid)

    def windows(sel):
        W = np.lib.stride_tricks.sliding_window_view(prof[sel], n_u, axis=1)[:, k_t, :]
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


def inside_sheet(ct_s, pts, n, ts, c_off, margin=0.1, reach=2.5, thr=None, rays=None):
    """True where candidate t puts the label inside papyrus: the mean CT over
    [t-reach, t-0.5] and over [t+0.5, t+reach] (around the expected edge, t + c_off) are
    both above the papyrus/gap mid level. At a real face one side is gap."""
    if len(pts) == 0:
        return np.zeros((0, len(ts)), bool), thr
    grid = np.arange(float(ts.min()) + c_off - reach, float(ts.max()) + c_off + reach + 1e-6, 0.5,
                     dtype=np.float32)
    prof = (rays or _rays(ct_s, pts, n))(grid)
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


def estimate_convention(ct_s, pts, n, p: Params, rays=None):
    """Where the steepest edge sits relative to the band centre, along +n, and the
    typical edge strength. Returns (offset, scale); (0, 1) if there are no samples to go on."""
    if len(pts) == 0:
        return 0.0, 1.0
    tf = np.arange(-p.convention_reach, p.convention_reach + 0.001, 0.25, dtype=np.float32)
    E = edge_strength(ct_s, pts, n, tf, p.edge_halfwidth, rays=rays)
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
    n_edges = (L - 2) * N + len(i) * (L - 1)
    if K is not None and K < L - 1:
        n_edges += 2 * len(i) * (L - 1 - K)
    g = maxflow.Graph[float](N * (L - 1), n_edges)          # no regrowing (and copying) of the arc array
    g.add_nodes(N * (L - 1))
    ids = np.arange(N * (L - 1), dtype=np.int64).reshape(N, L - 1)

    # Node l of a sample is on the source side when its level is above l, so cutting a chain at
    # level l must cost Dn[l]. Dn splits into a V-shaped envelope f (the running minimum from
    # either end) and a remainder. f goes on the terminal arcs and the remainder on the chain arcs:
    # the cut function is the same as with all of Dn on the chain arcs, but where neighbours agree
    # no flow is needed at all, so the same cut (the sink side is the nodes that can reach t) is found
    # with much shorter paths. That holds while the flow arithmetic is exact, as it is for float32 costs
    # and the default w = 1/8; with a w like 0.05, tied optima can come out one level apart.
    f = np.maximum(np.minimum.accumulate(Dn, axis=1), np.minimum.accumulate(Dn[:, ::-1], axis=1)[:, ::-1])
    df = f[:, 1:] - f[:, :-1]
    g.add_grid_tedges(ids, np.maximum(-df, 0), np.maximum(df, 0))
    if L > 2:
        g.add_edges(ids[:, :-1].ravel(), ids[:, 1:].ravel(),
                    (Dn - f)[:, 1:-1].ravel(), np.full((L - 2) * N, big))
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


# --------------------------------------------------------------------------- 6. warp

def warp_labels(lab, pts, n, shift, radius=6, chunk=200_000, comp=None, smooth=True,
                sigma=None, k=12, keep_topology=False, stages=3):
    """Move the label-1 band so the surface near sample p shifts by shift[p] * n[p].
    Label-2 voxels are never changed.

    smooth=True (default): each voxel's displacement is a Gaussian-weighted blend of the
    displacements of nearby samples on the same sheet (sigma defaults to the sample
    spacing), and the band is resampled with linear interpolation. The moved band then has
    clean edges like the original. smooth=False reproduces v1: each voxel takes its nearest
    sample's displacement and the band is resampled to the nearest voxel, which leaves
    stair steps wherever neighbouring samples moved by different amounts.

    keep_topology=True: the warped band is only a target. The input band walks to it through
    `stages` intermediate targets (fractions of the shift, so each overlaps the band it starts
    from), flipping only simple points, so components, tunnels and cavities stay as in the input
    at both connectivity pairs and separate sheets never touch. Label-2 voxels count as background."""
    band = lab == SURF
    out = lab.copy()
    if len(pts) == 0:
        return out
    annotated = lab != UNK                  # voxels labelled 2 keep their label, whatever the warp says
    near = ndi.binary_dilation(band, iterations=radius) & annotated
    tree = cKDTree(pts)
    fractions = [(s + 1) / stages for s in range(stages)] if keep_topology else [1.0]
    targets = np.zeros((len(fractions),) + band.shape, bool)
    hi = np.array(lab.shape) - 1
    vec = (np.asarray(shift, np.float32)[:, None] * n).astype(np.float32)
    if smooth:
        # a thick flat band can give coincident samples, and a spacing near 0 would erase it
        sigma = float(sigma) if sigma is not None else max(_sample_spacing(pts), 0.5)
        comp = np.zeros(len(pts), np.int64) if comp is None else np.asarray(comp)
        kk = min(k, len(pts))
    for z in range(0, near.shape[0], 16):      # a slab at a time: the voxel list is never held in full
        xs = np.argwhere(near[z:z + 16])
        xs[:, 0] += z
        for a in range(0, len(xs), chunk):
            x = xs[a:a + chunk]
            if not smooth:
                _, idx = tree.query(x, k=1)
                for target, f in zip(targets, fractions):
                    src = np.clip(np.rint(x - f * vec[idx]).astype(np.int64), 0, hi)
                    target[x[:, 0], x[:, 1], x[:, 2]] = band[src[:, 0], src[:, 1], src[:, 2]]
                continue
            d, idx = tree.query(x, k=kk, workers=_kd_workers())
            if kk == 1:
                d, idx = d[:, None], idx[:, None]
            w = np.exp(-0.5 * (d / sigma) ** 2)
            w *= comp[idx] == comp[idx[:, :1]]                 # same sheet as the nearest sample
            disp = np.einsum("ik,ikc->ic", w, vec[idx]) / w.sum(1, keepdims=True)
            for target, f in zip(targets, fractions):
                src = np.clip(x - f * disp, 0, hi).T
                vals = ndi.map_coordinates(band, src, order=1, mode="nearest", output=np.float32)
                target[x[:, 0], x[:, 1], x[:, 2]] = vals >= 0.5
    new_band = walk_to_targets(band, targets) if keep_topology else targets[0]
    out[band & annotated] = 0
    out[new_band & annotated] = SURF
    return out


# A voxel is simple when flipping it changes no component, tunnel or cavity. That depends only on
# its 26 neighbours, written as a 27-bit code (bit 9z + 3y + x for offsets z, y, x in 0..2; the
# centre, bit 13, is always 0). The tests below run on arrays of codes, by flooding bit masks.
_BITS = [(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)]
_ALL = (1 << 27) - 1
_NEIGHBOURS = _ALL ^ (1 << 13)
_FACES = sum(1 << b for b, o in enumerate(_BITS) if np.abs(o).sum() == 1)
_N18 = sum(1 << b for b, o in enumerate(_BITS) if 1 <= np.abs(o).sum() <= 2)
_X_UP = sum(1 << b for b, o in enumerate(_BITS) if o[2] < 1)     # voxels that can move +x, and so on
_X_DOWN = sum(1 << b for b, o in enumerate(_BITS) if o[2] > -1)
_Y_UP = sum(1 << b for b, o in enumerate(_BITS) if o[1] < 1)
_Y_DOWN = sum(1 << b for b, o in enumerate(_BITS) if o[1] > -1)
_simple_table = None                # per code: 0 not evaluated yet, 1 not simple, 2 simple


def _grow(a, conn):
    """Dilate each neighbourhood mask by one voxel, with 6- or 26-connectivity."""
    x = ((a & _X_UP) << 1) | ((a & _X_DOWN) >> 1)
    if conn == 6:
        return a | x | ((a & _Y_UP) << 3) | ((a & _Y_DOWN) >> 3) | ((a << 9) & _ALL) | (a >> 9)
    a = a | x
    a = a | ((a & _Y_UP) << 3) | ((a & _Y_DOWN) >> 3)
    return a | ((a << 9) & _ALL) | (a >> 9)


def _one_component(m, conn):
    """True where mask m has exactly one component around the centre: over all of N26 for
    26-connectivity, over N18 and counting only components that touch a face for 6-connectivity."""
    if conn == 6:
        m = m & _N18
    anchor = m & _FACES if conn == 6 else m
    reach = anchor & -anchor                        # the lowest set bit
    while True:
        grown = _grow(reach, conn) & m
        if np.array_equal(grown, reach):
            return (anchor != 0) & (reach & anchor == anchor)
        reach = grown


def simple_codes(codes, pairs=((6, 26), (26, 6))):
    """Which neighbourhood codes are simple under all the given (foreground, background)
    connectivity pairs (Bertrand & Malandain 1994): one foreground and one background component."""
    codes = np.asarray(codes, np.int64)
    ok = np.ones(len(codes), bool)
    for fg, bg in pairs:
        ok &= _one_component(codes, fg) & _one_component(~codes & _NEIGHBOURS, bg)
    return ok


def _simple_lookup(codes):
    """simple_codes under both pairs, remembering each distinct code; few occur in practice."""
    global _simple_table
    if _simple_table is None:
        _simple_table = np.zeros(_ALL + 1, np.uint8)     # pages stay untouched until used
    state = _simple_table[codes]
    if not state.all():
        new = np.unique(codes[state == 0])
        _simple_table[new] = 1 + simple_codes(new)
        state = _simple_table[codes]
    return state == 2


def flip_simple_points(X, T):
    """Flip voxels of X (boolean, padded by at least 1 voxel of False) to match T while a simple
    voxel remains, in place. Voxels of one 2x2x2 parity class are never neighbours, so each class
    can flip at once without changing any test in that class. Returns the number of flips."""
    Xf = X.ravel()                                       # a view
    sy, sx = X.shape[1] * X.shape[2], X.shape[2]
    steps = [(b, z * sy + y * sx + x) for b, (z, y, x) in enumerate(_BITS) if b != 13]
    todo = np.flatnonzero(Xf != T.ravel())
    parity = ((todo // sy) & 1) * 4 + (((todo % sy) // sx) & 1) * 2 + ((todo % sx) & 1)
    classes = [todo[parity == c] for c in range(8)]
    flips, last = 0, -1
    while flips != last:
        last = flips
        for c in range(8):
            if len(classes[c]) == 0:
                continue
            codes = np.zeros(len(classes[c]), np.int32)
            for b, d in steps:
                codes |= Xf[classes[c] + d].astype(np.int32) << b
            ok = _simple_lookup(codes)
            Xf[classes[c][ok]] ^= True
            classes[c] = classes[c][~ok]
            flips += int(ok.sum())
    return flips


def walk_to_targets(band, targets):
    """Move a boolean band through a sequence of target bands by simple points only."""
    X = np.pad(band, 1)
    for T in targets:
        flip_simple_points(X, np.pad(T, 1))
    return X[1:-1, 1:-1, 1:-1]


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
    return shift_volumes(lab_shape, mask, pts, [values], fill)[0]


def shift_volumes(lab_shape, mask, pts, value_arrays, fill=0):
    """shift_volume for several arrays of values, with one nearest-sample search."""
    vols = [np.full(lab_shape, fill, dtype=np.asarray(v).dtype) for v in value_arrays]
    if len(pts) == 0:
        return vols
    xs = np.argwhere(mask)
    _, idx = cKDTree(pts).query(xs, k=1)
    for vol, v in zip(vols, value_arrays):
        vol[xs[:, 0], xs[:, 1], xs[:, 2]] = np.asarray(v)[idx]
    return vols


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
    n, comp = orient_normals(pts, n, i, j, ct_s, orient_cube=p.orient_cube)
    return ct_s, pts, n, i, j, comp


def unary_costs(lab_shape, prepared, p: Params, rays=None, extras=None):
    """Edge-based cost of each candidate shift for each sample, plus bookkeeping.
    If `extras` is a dict, the samples' own offset (before p.convention_offset) is stored in it as 'own_offset'."""
    ct_s, pts, n, i, j, comp = prepared
    rays = rays or _rays(ct_s, pts, n)
    ts = np.arange(-p.t_max, p.t_max + 1e-6, p.t_step, dtype=np.float32)
    lo = p.border
    hi = np.array(lab_shape) - 1 - p.border
    movable = np.all((pts >= lo) & (pts <= hi), axis=1)
    c_off, scale = estimate_convention(ct_s, pts[movable], n[movable], p, rays=lambda g: rays(g)[movable])
    if extras is not None:
        extras["own_offset"] = c_off
    if p.convention_offset is not None:
        c_off = p.convention_offset
    E = edge_strength(ct_s, pts, n, ts + c_off, p.edge_halfwidth, rays=rays) / scale
    return ts, E, movable, c_off, scale


def reversed_faces(ct_s, pts, n, E, c_off, scale, p: Params, rays=None):
    """Samples whose label sits on a clear face of the opposite polarity (papyrus on the +n
    side, gap on the -n side) while no candidate along the normal shows a face of the
    expected polarity. There the label is probably on the far face of its sheet, or the
    normal is flipped; searching along +n would only push it into the papyrus."""
    near = np.arange(-1.5, 1.501, 0.5, dtype=np.float32)
    e_rev = edge_strength(ct_s, pts, -n, near + c_off, p.edge_halfwidth,
                          rays=(lambda g: rays(-g)) if rays else None).max(1) / scale
    return (e_rev >= p.reverse_min) & (E.max(1) < p.forward_max)


def unary_scores(lab_shape, prepared, p: Params, extras=None):
    """Score of each candidate shift (higher = more like a correctly placed label):
    normalised step-edge strength + template_weight * learned-profile match.

    With p.face_polarity, samples found by reversed_faces() are scored along -n instead, so
    they settle on the face they are on. All scores stay indexed by the shift along +n.
    If `extras` is a dict, the reversed-face mask is stored in it as 'reversed'.

    The template is the samples' own typical profile, so it puts the edge at their own offset.
    With p.convention_offset, every candidate's window is read that much further along, so that
    both terms put the edge at the requested offset (unless p.template_follows_offset is off)."""
    extras = {} if extras is None else extras
    ct_s, pts, n = prepared[0], prepared[1], prepared[2]
    rays = _rays(ct_s, pts, n)
    ts, E, movable, c_off, scale = unary_costs(lab_shape, prepared, p, rays=rays, extras=extras)
    d_own = c_off - extras["own_offset"] if p.template_follows_offset else 0.0
    S = np.zeros_like(E)
    ref, tmpl = None, None
    if p.template_weight > 0 and movable.sum() > 100:
        peak = E.max(1)
        ref = movable & (peak >= np.percentile(peak[movable], 50))
        S, tmpl = template_scores(ct_s, pts, n, ts, ref, rays=rays)
        if d_own != 0:
            S, _ = template_scores(ct_s, pts, n, ts + d_own, None, tmpl=tmpl, rays=rays)
    rev = np.zeros(len(pts), bool)
    if len(pts):
        rev = reversed_faces(ct_s, pts, n, E, c_off, scale, p, rays=rays) & movable
        if p.face_polarity and rev.any():
            # ts is symmetric, so a shift t along -n is candidate -t along +n: reverse the axis
            E[rev] = (edge_strength(ct_s, pts[rev], -n[rev], ts + c_off, p.edge_halfwidth) / scale)[:, ::-1]
            if tmpl is not None:
                S[rev] = template_scores(ct_s, pts[rev], -n[rev], ts + d_own, None, tmpl=tmpl)[0][:, ::-1]
    extras["reversed"] = rev
    extras["rays"] = rays
    U = E + p.template_weight * S

    # Self-calibration: assume the annotators are right on average, so the median
    # preferred correction over the cube should be zero. If the scores prefer a
    # consistent offset (e.g. phase-contrast halos or blur shift the apparent edge in
    # this scan), evaluate every candidate that much further along instead.
    if p.calibrate and movable.sum() > 100:
        d_med = float(np.median(ts[np.argmax(U[movable], 1)]))
        if abs(d_med) >= p.t_step:
            E = edge_strength(ct_s, pts, n, ts + c_off + d_med, p.edge_halfwidth, rays=rays) / scale
            if ref is not None:
                S, _ = template_scores(ct_s, pts, n, ts + d_own + d_med, ref, rays=rays)
            U = E + p.template_weight * S
            c_off += d_med
    return ts, U, E, S, movable, c_off, scale


def solve_all(D, i, j, comp, p: Params):
    """Run the graph cut sheet by sheet; returns the chosen level per sample."""
    w_level = p.smooth_w * p.t_step
    K = None if p.max_step is None else int(round(p.max_step / p.t_step))
    labels = np.zeros(len(D), int)
    if np.any(comp[i] != comp[j]):
        raise ValueError("neighbour pairs must stay within a sheet")
    # one graph per sheet: components don't interact, and the largest sheet sets the peak memory
    by_comp = np.argsort(comp, kind="stable")
    starts = np.concatenate(([0], np.cumsum(np.bincount(comp))))
    pos = np.empty(len(D), np.int64)
    pos[by_comp] = np.arange(len(D))
    e_order = np.argsort(comp[i], kind="stable")
    e_starts = np.concatenate(([0], np.cumsum(np.bincount(comp[i], minlength=len(starts) - 1))))
    for c in range(len(starts) - 1):
        idx = by_comp[starts[c]:starts[c + 1]]
        e = e_order[e_starts[c]:e_starts[c + 1]]
        labels[idx] = solve_shifts(D[idx], pos[i[e]] - starts[c], pos[j[e]] - starts[c], w_level, K)
    return labels


def snap(img, lab, p: Params | None = None, prepared=None):
    p = p or Params()
    if img.shape != lab.shape:
        raise ValueError(f"the CT has shape {img.shape} but the labels have shape {lab.shape}")
    prepared = prepared if prepared is not None else prepare(img, lab, p)
    ct_s, pts, n, i, j, comp = prepared
    shift, conf, movable, rev, info = solve_surface(lab.shape, prepared, p)
    del ct_s, prepared             # the warp doesn't need the smoothed CT (unless the caller keeps it)
    out = warp_labels(lab, pts, n, shift, comp=comp, smooth=p.smooth_warp, keep_topology=p.keep_topology)
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
        inside, thr = inside_sheet(ct_s, pts, n, ts, c_off, p.inside_margin, rays=extras["rays"])
        inside[:, ts <= 0] = False
        if p.face_polarity and rev.any():
            # for reversed-face samples "outward" is -n: same test in their own frame
            inside_r, _ = inside_sheet(ct_s, pts[rev], -n[rev], ts, c_off, p.inside_margin, thr=thr)
            inside_r[:, ts <= 0] = False
            inside[rev] = inside_r[:, ::-1]
        D = D + p.inside_penalty * inside
    del U, extras                  # the sampled profiles are no longer needed
    # border samples: forced to stay put
    D[~movable] = 1e3
    D[~movable, k0] = 0.0

    labels = solve_all(D, i, j, comp, p)
    shift = ts[labels]
    rows = np.arange(len(pts))
    # confidence: how label-like the CT profile is at the chosen position (template
    # match), falling back to edge strength when the template is off
    conf = np.clip(S[rows, labels] if p.template_weight > 0 else E[rows, labels], 0, 1)
    info = dict(n_samples=int(len(pts)), n_movable=int(movable.sum()), n_pairs=int(len(i)),
                n_sheets=int(comp.max() + 1) if len(comp) else 0, convention_offset=float(c_off), edge_scale=float(scale),
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
