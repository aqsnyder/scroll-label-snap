"""The fast paths in core.py must give exactly what the plain code they replaced gives, byte for byte. The plain versions
are kept here. The solver is checked against its specification: on costs where the flow arithmetic is exact, it returns the
pointwise-largest minimiser, whichever network or grouping of sheets is used."""
import itertools
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap import core  # noqa: E402
from scroll_label_snap.core import Params, sample, solve_all, solve_shifts  # noqa: E402


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def sheets(shape=(48, 48, 48), seed=0):
    """A noisy CT of two wavy sheets, its recto labels, and an unannotated corner."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    f = y - 12 - 2 * np.sin(x / 6) - 1.5 * np.cos(z / 7)
    inside = (f > 0) & (f < 6) | (f > 18) & (f < 24)
    ct = ndi.gaussian_filter(np.where(inside, 170.0, 60.0), 1.0) + np.random.default_rng(seed).normal(0, 8, shape)
    lab = ((np.abs(f - 6) < 1) | (np.abs(f - 24) < 1)).astype(np.uint8)
    lab[:8, :, :8] = 2
    return np.clip(ct, 0, 255).astype(np.uint8), lab


def plain_surface_points(lab, p):
    """Six whole-volume second-derivative filters."""
    m = (lab == core.SURF).astype(np.float32)
    s = ndi.gaussian_filter(m, p.sigma_band)
    zz, yy, xx = np.nonzero(lab == core.SURF)
    coords = np.stack([zz, yy, xx], 1)
    sval = s[zz, yy, xx]
    cell = (coords // p.spacing).astype(np.int64)
    key = (cell[:, 0] * 4096 + cell[:, 1]) * 4096 + cell[:, 2]
    order = np.lexsort((-sval, key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    pts = coords[sel].astype(np.float32)
    H = np.empty((len(sel), 3, 3), np.float32)
    z, y, x = coords[sel].T
    for (a, b), o in {(0, 0): (2, 0, 0), (1, 1): (0, 2, 0), (2, 2): (0, 0, 2),
                      (0, 1): (1, 1, 0), (0, 2): (1, 0, 1), (1, 2): (0, 1, 1)}.items():
        H[:, a, b] = H[:, b, a] = ndi.gaussian_filter(m, p.sigma_band, order=o)[z, y, x]
    n = np.linalg.eigh(H)[1][:, :, 0].astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    tc = np.arange(-2.0, 2.001, 0.25, dtype=np.float32)
    sv = np.stack([sample(s, pts + t * n) for t in tc], 1)
    return (pts + tc[np.argmax(sv, 1)][:, None] * n).astype(np.float32), n, s


def plain_warp(lab, pts, n, shift, radius, comp, smooth, sigma, k):
    """The whole voxel list at once, a float32 band and float32 displacements."""
    band = lab == core.SURF
    xs = np.argwhere(ndi.binary_dilation(band, iterations=radius))
    tree = cKDTree(pts)
    new_band = np.zeros_like(band)
    hi = np.array(lab.shape) - 1
    vec = (np.asarray(shift, np.float32)[:, None] * n).astype(np.float32)
    if smooth:
        d, idx = tree.query(xs, k=k)
        w = np.exp(-0.5 * (d / sigma) ** 2)
        w *= comp[idx] == comp[idx[:, :1]]
        disp = np.einsum("ik,ikc->ic", w, vec[idx]) / w.sum(1, keepdims=True)
        vals = ndi.map_coordinates(band.astype(np.float32), np.clip(xs - disp, 0, hi).T, order=1, mode="nearest")
        new_band[xs[:, 0], xs[:, 1], xs[:, 2]] = vals >= 0.5
    else:
        _, idx = tree.query(xs, k=1)
        src = np.clip(np.rint(xs - vec[idx]).astype(np.int64), 0, hi)
        new_band[xs[:, 0], xs[:, 1], xs[:, 2]] = band[src[:, 0], src[:, 1], src[:, 2]]
    out = lab.copy()
    out[band & (lab != core.UNK)] = 0
    out[new_band & (lab != core.UNK)] = core.SURF
    return out


def test_filter_at_points_matches_the_whole_volume():
    rng = np.random.default_rng(0)
    for nx in (1, 2, 5, 40):                     # narrower than the kernel: reflected more than once
        vol = rng.normal(size=(6, 7, nx)).astype(np.float32)
        z, y, x = (a.ravel() for a in np.mgrid[:6, :7, :nx])
        for sigma in (0.7, 1.5, 4.0):
            for order in (0, 1, 2):
                whole = ndi.gaussian_filter1d(vol, sigma, axis=2, order=order)[z, y, x]
                assert same(core._filter_x_at(vol, z, y, x, sigma, order), whole), (nx, sigma, order)


def test_surface_points_matches_plain_filters():
    _, lab = sheets()
    plane = np.zeros((40, 40, 40), np.uint8)
    plane[:, 19:21, :] = 1                       # every voxel of a cell ties
    for v in (lab, plane):
        for p in (Params(), Params(sigma_band=1.0, spacing=2.5)):
            for a, b in zip(core.surface_points(v, p), plain_surface_points(v, p)):
                assert same(a, b)


def test_ray_cache_matches_direct_sampling():
    ct, lab = sheets()
    ct_s = ndi.gaussian_filter(ct.astype(np.float32), 1.0)
    pts, n, _ = core.surface_points(lab, Params())
    grid = np.arange(-5.25, 5.3, 0.5, dtype=np.float32)
    rays = core._rays(ct_s, pts, n)
    assert same(rays(grid), np.stack([sample(ct_s, pts + g * n) for g in grid], 1))
    assert same(rays(grid[3:9]), np.stack([sample(ct_s, pts + g * n) for g in grid[3:9]], 1))
    assert same(rays(-grid), np.stack([sample(ct_s, pts + g * -n) for g in grid], 1))
    keep = np.arange(len(pts)) % 3 == 0
    assert same(rays(grid)[keep], np.stack([sample(ct_s, pts[keep] + g * n[keep]) for g in grid], 1))


def test_warp_matches_plain_loop():
    ct, lab = sheets()
    p = Params()
    ct_s, pts, n, i, j, comp = core.prepare(ct, lab, p)
    shift = np.random.default_rng(1).choice(np.arange(-4, 4.01, 0.5), len(pts)).astype(np.float32)
    sigma = core._sample_spacing(pts)
    for smooth in (True, False):
        for k in (12, 2):
            got = core.warp_labels(lab, pts, n, shift, comp=comp, smooth=smooth, k=k)
            assert same(got, plain_warp(lab, pts, n, shift, 6, comp, smooth, sigma, k)), (smooth, k)
    # a flat band pushed half a voxel: the resampled edge is exactly 0.5, the threshold
    plane = np.zeros((40, 40, 40), np.uint8)
    plane[:, 19:21, :] = 1
    pts = np.argwhere(plane == 1)[::3].astype(np.float32)
    n = np.tile(np.array([0, 1, 0], np.float32), (len(pts), 1))
    shift = np.full(len(pts), 0.5, np.float32)
    comp = np.zeros(len(pts), np.int64)
    got = core.warp_labels(plane, pts, n, shift, comp=comp)
    assert same(got, plain_warp(plane, pts, n, shift, 6, comp, True, core._sample_spacing(pts), 12))
    assert (got == 1).sum() > (plane == 1).sum()


def test_ties_give_the_largest_minimiser():
    rng = np.random.default_rng(2)
    for trial in range(300):
        N, L = int(rng.integers(1, 6)), int(rng.integers(2, 5))
        D = rng.integers(-3, 4, size=(N, L)) * 0.125             # dyadic costs: sums and ties are exact
        pairs = [(a, b) for a in range(N) for b in range(a + 1, N) if rng.random() < 0.6]
        i = np.array([a for a, _ in pairs], np.int64)
        j = np.array([b for _, b in pairs], np.int64)
        w = float(rng.choice([0.0, 0.125, 0.25]))
        K = [None, 0, 1, 2][trial % 4]
        best, top = np.inf, None
        for lab in itertools.product(range(L), repeat=N):
            lab = np.array(lab)
            if K is not None and len(i) and np.abs(lab[i] - lab[j]).max() > K:
                continue
            e = D[np.arange(N), lab].sum() + w * np.abs(lab[i] - lab[j]).sum()
            if e < best:
                best, top = e, lab
            elif e == best:
                top = np.maximum(top, lab)
        assert np.array_equal(solve_shifts(D, i, j, w, K), top), trial


def test_solve_all_does_not_depend_on_how_sheets_are_grouped():
    rng = np.random.default_rng(3)
    for trial in range(20):
        sizes = rng.integers(1, 30, size=int(rng.integers(1, 12)))
        comp = np.repeat(np.arange(len(sizes)), sizes)[rng.permutation(int(sizes.sum()))]
        pairs = [(a, b) for c in range(len(sizes)) for a, b in zip(np.flatnonzero(comp == c)[:-1], np.flatnonzero(comp == c)[1:])]
        i = np.array([a for a, _ in pairs], np.int64)
        j = np.array([b for _, b in pairs], np.int64)
        D = rng.integers(0, 4, size=(len(comp), 9)).astype(np.float32)
        one_graph = solve_shifts(D, i, j, Params().smooth_w * Params().t_step, 2)
        assert np.array_equal(solve_all(D, i, j, comp, Params()), one_graph), trial
