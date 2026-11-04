"""tools/depth_transfer.py on synthetic surfaces with known depth: the plan's behaviour by default, and
what --refine and --normal-sigma change on a rough surface."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

tifffile = pytest.importorskip("tifffile")

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, ".."))
import depth_transfer as dt  # noqa: E402
from scroll_label_snap.tifxyz import write_tifxyz  # noqa: E402

SIZE = 240
SPACING = 3


def rough_height(rms, seed=0, corr=4.0):
    """A height field in voxels over a 1-voxel grid, smooth over `corr` voxels, with this rms."""
    h = ndi.gaussian_filter(np.random.default_rng(seed).normal(size=(SIZE, SIZE)), corr, mode="wrap")
    return h / h.std() * rms


def surface(height, z0, spacing=SPACING):
    """The sheet z = z0 + height sampled every `spacing` voxels, as (z, y, x) vertices."""
    h = height[::spacing, ::spacing]
    r, c = np.mgrid[0:SIZE:spacing, 0:SIZE:spacing].astype(np.float32)
    return np.stack([z0 + h.astype(np.float32), r, c], -1), np.ones(h.shape, bool)


def target_pixels(z0=100.0, margin=40):
    """Flat target pixels with normals along z, away from the surface's edge."""
    r, c = np.mgrid[margin:SIZE - margin, margin:SIZE - margin].astype(np.float32)
    P = np.stack([np.full_like(r, z0), r, c], -1)
    N = np.zeros_like(P)
    N[..., 0] = 1
    return P, N, np.ones(r.shape, bool), (r.astype(int), c.astype(int))


def carry(height, depth0, refine, sigma, min_dot=0.8):
    Pw, vw = surface(height, 100.0 + depth0)
    P, N, v, rc = target_pixels()
    lo, hi = P[v].min(0), P[v].max(0)
    Q, M, _ = dt.surface_samples(Pw, vw, lo, hi, 14, sigma)
    d, g = dt.transfer(P, N, v, Q, M, 10.0, 2.0, min_dot, refine)
    return d, g, depth0 + height[rc]


def as_run(Ps, Ns, vs, Qw, Nw, max_depth, max_gap, min_dot):
    """The pinned tool's test (627d058): the gap to the sample nearest the pixel."""
    from scipy.spatial import cKDTree
    depth = np.full(vs.shape, np.nan, np.float32)
    p, n = Ps[vs].astype(np.float64), Ns[vs].astype(np.float64)
    dist, idx = cKDTree(Qw).query(p, distance_upper_bound=float(np.hypot(max_depth, max_gap)) + 1.0)
    found = np.isfinite(dist)
    t = np.full(len(p), np.nan)
    q, m, pn, nn = Qw[idx[found]].astype(np.float64), Nw[idx[found]].astype(np.float64), p[found], n[found]
    d = np.einsum("ij,ij->i", nn, m)
    good = np.abs(d) >= min_dot
    tt = np.where(good, np.einsum("ij,ij->i", q - pn, m) / np.where(good, d, 1.0), np.nan)
    gg = np.linalg.norm(pn + tt[:, None] * nn - q, axis=1)
    ok = good & (np.abs(tt) <= max_depth) & (gg <= max_gap)
    t[np.flatnonzero(found)[ok]] = tt[ok]
    depth[vs] = t.astype(np.float32)
    return depth


def test_default_is_the_pinned_test():
    Pw, vw = surface(rough_height(1.5), 106.0)
    P, N, v, _ = target_pixels()
    Q, M, _ = dt.surface_samples(Pw, vw, P[v].min(0), P[v].max(0), 14)
    d, _ = dt.transfer(P, N, v, Q, M, 10.0, 2.0, 0.8)
    assert np.array_equal(d, as_run(P, N, v, Q, M, 10.0, 2.0, 0.8), equal_nan=True)
    assert np.isnan(d).mean() > 0.05    # this surface is rough enough to show the holes


def test_both_agree_on_a_flat_surface():
    for refine, sigma in ((0, 0.0), (3, 10.0)):
        d, _, truth = carry(np.zeros((SIZE, SIZE)), 6.0, refine, sigma)
        assert np.isfinite(d).all()
        assert np.abs(d - truth).max() < 1e-3


@pytest.mark.parametrize("depth0", [4.0, 7.0])
def test_refine_removes_false_holes_without_wrong_depths(depth0):
    height = rough_height(1.5)
    before, _, truth = carry(height, depth0, 0, 0.0)
    after, gap, _ = carry(height, depth0, 3, 10.0)
    reachable = np.abs(truth) < 9.5     # peaks past --max-depth are rightly left out
    assert np.isnan(before)[reachable].mean() > 0.05
    assert np.isnan(after)[reachable].mean() < 0.005
    assert (gap[np.isfinite(gap)] <= 2.0).all()
    err = np.abs(after - truth)[np.isfinite(after)]
    assert (err > 1).mean() < 0.01
    assert (err > 2).mean() < 0.002


def test_refine_alone_also_removes_most_of_them():
    height = rough_height(1.5)
    before = np.isnan(carry(height, 7.0, 0, 0.0)[0]).mean()
    after = np.isnan(carry(height, 7.0, 3, 0.0)[0]).mean()
    assert after < before / 3


def test_limits_still_hold_with_refine():
    d, _, _ = carry(np.zeros((SIZE, SIZE)), 10.5, 3, 10.0)
    assert np.isnan(d).all()    # beyond --max-depth, though near enough for the search to find it
    # a sheet steeper than the normal test everywhere: 45 degrees against min_dot 0.8
    r, _ = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    d, _, _ = carry(r - SIZE / 2, 0.0, 3, 10.0)
    assert np.isnan(d).all()


def test_gap_limit_holds_with_refine():
    """Past the surface's edge the normal line misses it: within 2 voxels of a sample, or no depth."""
    Pw, vw = surface(np.zeros((SIZE, SIZE)), 106.0)
    vw[:, Pw[0, :, 2] > 120] = False
    P, N, v, (_, c) = target_pixels()
    Q, M, _ = dt.surface_samples(Pw, vw, P[v].min(0), P[v].max(0), 14)
    d, g = dt.transfer(P, N, v, Q, M, 10.0, 2.0, 0.8, 3)
    edge = Q[:, 2].max()
    assert np.isfinite(d[c <= edge]).all()
    assert np.isnan(d[c > edge + 2.5]).all()
    assert (g[np.isfinite(g)] <= 2.0).all()


def test_normal_sigma_keeps_pixels_on_steep_facets():
    """Raw normals on a rough 3-voxel surface judge its facets, not its sheet, against --min-dot."""
    height = rough_height(1.5, corr=2.0)
    raw = np.isnan(carry(height, 5.0, 3, 0.0)[0]).mean()
    smooth = np.isnan(carry(height, 5.0, 3, 10.0)[0]).mean()
    assert raw > 0.01
    assert smooth < raw / 3


def test_smooth_normals_average_within_the_mask():
    N = np.zeros((40, 40, 3), np.float32)
    N[..., 0] = 1
    N[::2, :, 1] = 1
    N[1::2, :, 1] = -1
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    ok = np.ones((40, 40), bool)
    ok[:, 30:] = False
    S, okS = dt.smooth_normals(N, ok, 3.0)
    assert not okS[:, 30:].any()
    assert (S[~okS] == 0).all()
    assert np.allclose(np.linalg.norm(S[okS], axis=-1), 1, atol=1e-5)
    assert np.abs(S[10:30, 5:25, 1]).max() < 0.05
    C = np.zeros((40, 40, 3), np.float32)
    C[::2, :, 1] = 1
    C[1::2, :, 1] = -1
    assert not dt.smooth_normals(C, np.ones((40, 40), bool), 3.0)[1][10:30, 10:30].any()    # they cancel


def test_refine_stays_on_the_nearer_sheet():
    P, N, v, _ = target_pixels()
    lo, hi = P[v].min(0), P[v].max(0)
    near, vn = surface(rough_height(1.0, 1), 104.0)
    far, vf = surface(rough_height(1.0, 2), 112.0)
    Qa, Ma, _ = dt.surface_samples(near, vn, lo, hi, 14, 10.0)
    Qb, Mb, _ = dt.surface_samples(far, vf, lo, hi, 14, 10.0)
    d, _ = dt.transfer(P, N, v, np.concatenate([Qa, Qb]), np.concatenate([Ma, Mb]), 10.0, 2.0, 0.8, 3)
    assert np.isfinite(d).mean() > 0.99
    assert (d[np.isfinite(d)] < 8).mean() > 0.99


def test_normal_sigma_is_in_voxels(monkeypatch):
    seen = []
    real = dt.smooth_normals
    monkeypatch.setattr(dt, "smooth_normals", lambda N, ok, sigma: seen.append(sigma) or real(N, ok, sigma))
    P, N, v, _ = target_pixels()
    for spacing in (3, 6):
        Pw, vw = surface(np.zeros((SIZE, SIZE)), 106.0, spacing)
        dt.surface_samples(Pw, vw, P[v].min(0), P[v].max(0), 14, 12.0)
    assert seen == pytest.approx([12.0 / 3, 12.0 / 6], rel=1e-3)


def test_command_line_records_the_settings(tmp_path):
    height = rough_height(1.5)
    Pw, vw = surface(height, 106.0)
    r, c = np.mgrid[0:SIZE:20, 0:SIZE:20].astype(np.float32)
    Pt = np.stack([np.full_like(r, 100.0), r, c], -1)
    write_tifxyz(str(tmp_path / "surface.tifxyz"), Pw, vw, scale=1 / SPACING)
    write_tifxyz(str(tmp_path / "target.tifxyz"), Pt, np.ones(r.shape, bool), scale=0.05)
    results = {}
    for name, extra in (("default", []), ("fixed", ["--refine", "3", "--normal-sigma", "10"]),
                        ("refine", ["--refine", "3"])):
        out = tmp_path / f"{name}.npz"
        subprocess.run([sys.executable, os.path.join(TOOLS, "depth_transfer.py"), str(out),
                        "--target", str(tmp_path / "target.tifxyz"), "--surface", str(tmp_path / "surface.tifxyz"),
                        "--roi", "40", "200", "40", "200"] + extra, check=True, capture_output=True, text=True)
        z = np.load(out)
        results[name] = (z["depth"], json.loads(str(z["stats"])))
    assert results["default"][1]["refine"] == 0 and results["default"][1]["normal_sigma"] == 0
    assert results["fixed"][1]["refine"] == 3 and results["fixed"][1]["normal_sigma"] == 10
    assert np.isnan(results["default"][0]).mean() > 0.05
    assert np.isnan(results["fixed"][0]).mean() < 0.005
    assert results["refine"][1]["normal_sigma"] == 0
    assert not np.array_equal(results["refine"][0], results["fixed"][0], equal_nan=True)    # the flag reaches the normals
