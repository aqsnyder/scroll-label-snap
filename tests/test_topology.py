"""warp_labels(keep_topology=True) must keep the input's components, tunnels and cavities at both
connectivity pairs, and its simple-point test must match component counting. snap_cube.py and snap_folder.py
can set the option with --param."""
import json
import os
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scroll_label_snap.core import Params, flip_simple_points, prepare, simple_codes, snap, warp_labels  # noqa: E402
from regression_cases import label_case  # noqa: E402
from test_robustness import mini_set, run  # noqa: E402

S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)
OFFSETS = [(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)]
WINDOWS = [[(1, 1, 1)], [(1, 1, 2), (1, 2, 1), (2, 1, 1)], [(1, 2, 2), (2, 1, 2), (2, 2, 1)], [(2, 2, 2)]]


def betti(X, conn):
    """(components, tunnels, cavities) with the outside as background. conn=6: foreground 6-connected,
    voxels are the vertices of a cell complex. conn=26: foreground 26-connected, voxels are closed cubes."""
    Xp = np.pad(np.asarray(X, bool), 1)
    fg, bg = (S6, S26) if conn == 6 else (S26, S6)
    b0 = ndi.label(Xp, structure=fg)[1]
    b2 = ndi.label(np.pad(~Xp, 1, constant_values=True), structure=bg)[1] - 1
    chi = 0
    for k, shapes in enumerate(WINDOWS):
        for s in shapes:
            w = np.lib.stride_tricks.sliding_window_view(Xp, s)
            count = w.all((3, 4, 5)).sum() if conn == 6 else w.any((3, 4, 5)).sum()
            chi += (-1) ** k * int(count)
    if conn == 26:
        chi = -chi
    return b0, b0 + b2 - chi, b2


def test_betti_on_known_shapes():
    z = np.zeros((14, 14, 14), bool)
    ball, torus, two = z.copy(), z.copy(), z.copy()
    ball[3:10, 3:10, 3:10] = True
    shell = ball.copy()
    shell[5:8, 5:8, 5:8] = False
    torus[3:10, 3:10, 5:7] = True
    torus[5:8, 5:8, 5:7] = False
    two[2:4, 2:12, 2:12] = True
    two[6:8, 2:12, 2:12] = True
    for conn in (6, 26):
        assert betti(ball, conn) == (1, 0, 0)
        assert betti(shell, conn) == (1, 0, 1)
        assert betti(torus, conn) == (1, 1, 0)
        assert betti(two, conn) == (2, 0, 0)


def components(cells, adjacent):
    """Connected components of a list of offsets, by plain search."""
    left, comps = set(cells), []
    while left:
        stack = [left.pop()]
        comp = set(stack)
        while stack:
            u = stack.pop()
            for v in [c for c in left if adjacent(u, c)]:
                left.discard(v)
                comp.add(v)
                stack.append(v)
        comps.append(comp)
    return comps


def adj6(a, b):
    return sum(abs(p - q) for p, q in zip(a, b)) == 1


def adj26(a, b):
    return max(abs(p - q) for p, q in zip(a, b)) == 1


def t6(cells):
    near = [c for c in cells if sum(abs(v) for v in c) <= 2]
    return sum(any(sum(abs(v) for v in c) == 1 for c in comp) for comp in components(near, adj6))


def test_simple_codes_match_component_counting():
    rng = np.random.default_rng(0)
    codes = (rng.random((4000, 27)) < rng.uniform(0.05, 0.95, (4000, 1))).astype(np.int64)
    codes[:, 13] = 0
    codes = (codes << np.arange(27)).sum(1)
    a = simple_codes(codes, pairs=((6, 26),))
    b = simple_codes(codes, pairs=((26, 6),))
    assert np.array_equal(simple_codes(codes), a & b)
    assert 0.02 < a.mean() < 0.98 and 0.02 < b.mean() < 0.98     # both answers occur
    for c, sa, sb in zip(codes.tolist(), a, b):
        fg = [o for k, o in enumerate(OFFSETS) if k != 13 and (c >> k) & 1]
        bg = [o for k, o in enumerate(OFFSETS) if k != 13 and not (c >> k) & 1]
        assert sa == (t6(fg) == 1 and len(components(bg, adj26)) == 1), c
        assert sb == (len(components(fg, adj26)) == 1 and t6(bg) == 1), c


def noisy_plates(rng, shape=(36, 36, 36)):
    """Two close wavy plates and a noisy target, so many flips are refused."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    w = 1.5 * np.sin(x / 7 + rng.uniform(0, 6)) + 1.5 * np.cos(z / 9 + rng.uniform(0, 6))
    X = (np.abs(y - 11 - w) <= 1.2) | (np.abs(y - 20 - w) <= 1.2)
    T = (np.abs(y - 13 - w) <= 1.2) | (np.abs(y - 18 - w) <= 1.2)
    return np.pad(X, 2), np.pad(T ^ (rng.random(shape) < 0.02), 2)


def test_flips_keep_every_betti_number():
    for seed in range(6):
        X, T = noisy_plates(np.random.default_rng(seed))
        before = betti(X, 6), betti(X, 26)
        assert flip_simple_points(X, T) > 0
        assert (betti(X, 6), betti(X, 26)) == before


def test_flipping_neighbours_together_would_break_it():
    """Control: the same flips without the parity classes do change the topology."""
    X, T = noisy_plates(np.random.default_rng(0))
    before = betti(X, 6), betti(X, 26)
    Xf, Tf = X.ravel(), T.ravel()
    sy, sx = X.shape[1] * X.shape[2], X.shape[2]
    steps = [(k, o[0] * sy + o[1] * sx + o[2]) for k, o in enumerate(OFFSETS) if k != 13]
    todo = np.flatnonzero(Xf != Tf)
    codes = np.zeros(len(todo), np.int64)
    for k, d in steps:
        codes |= Xf[todo + d].astype(np.int64) << k
    Xf[todo[simple_codes(codes)]] ^= True
    assert (betti(X, 6), betti(X, 26)) != before


def two_close_sheets(n=64):
    """Two plates 6 voxels apart whose shifts run into each other in the middle of the cube. Returns the label
    volume (one corner not annotated) and samples, normals, shifts and sheet ids for warp_labels."""
    lab = np.zeros((n, n, n), np.uint8)
    lab[:, 22:25, :] = 1
    lab[:, 31:34, :] = 1
    lab[:10, :, :10] = 2
    grid = np.arange(1.0, n, 3.0)
    Z, X = np.meshgrid(grid, grid, indexing="ij")
    bump = 3.9 * np.exp(-(((Z - n / 2) ** 2 + (X - n / 2) ** 2) / (2 * 12.0 ** 2)))
    lower = np.stack([Z, np.full_like(Z, 23.0), X], -1).reshape(-1, 3)
    upper = np.stack([Z, np.full_like(Z, 32.0), X], -1).reshape(-1, 3)
    pts = np.concatenate([lower, upper]).astype(np.float32)
    normals = np.tile(np.array([0, 1, 0], np.float32), (len(pts), 1))
    shift = np.concatenate([bump.ravel(), -bump.ravel()]).astype(np.float32)
    comp = np.repeat([0, 1], len(lower))
    return lab, pts, normals, shift, comp


@pytest.mark.parametrize("smooth", [True, False])
def test_two_close_sheets_keep_their_topology(smooth):
    lab, pts, normals, shift, comp = two_close_sheets()
    plain = warp_labels(lab, pts, normals, shift, comp=comp, smooth=smooth)
    kept = warp_labels(lab, pts, normals, shift, comp=comp, smooth=smooth, keep_topology=True)
    band, band_plain, band_kept = lab == 1, plain == 1, kept == 1
    # the phantom really does join the sheets when moved voxel by voxel
    assert ndi.label(band_plain, structure=S26)[1] == 1
    assert ndi.label(band, structure=S26)[1] == 2
    for conn in (6, 26):
        assert betti(band_kept, conn) == betti(band, conn)
    # label 2 untouched, and the band still goes where it was sent away from the contested middle
    assert np.array_equal(kept == 2, lab == 2)
    zz, xx = np.mgrid[:64, :64]
    far = (zz - 32) ** 2 + (xx - 32) ** 2 > 30 ** 2
    assert (band_kept == band_plain).all(1)[far & ~((zz < 10) & (xx < 10))].mean() > 0.95
    # both sheets survive in the middle, at least a voxel apart. Their targets overlap there, so
    # they share the 4 voxels of the joined target less the voxel between them
    column = band_kept[32, :, 32]
    assert ndi.label(column)[1] == 2
    assert column.sum() >= 3


def test_snap_option_keeps_topology_and_default_output():
    ct, lab = label_case()
    prepared = prepare(ct, lab, Params())
    r = snap(ct, lab, Params(keep_topology=True), prepared=prepared)
    for conn in (6, 26):
        assert betti(r.labels == 1, conn) == betti(lab == 1, conn)
    assert np.array_equal(r.labels == 2, lab == 2)
    plain = snap(ct, lab, Params(), prepared=prepared)
    assert np.array_equal(plain.shift, r.shift)                  # the option only changes the voxel step
    assert np.array_equal(plain.labels, warp_labels(lab, r.pts, r.normals, r.shift, comp=prepared[5]))


def test_tools_set_the_option_with_param(tmp_path):
    names = mini_set(tmp_path / "in", 1)
    img, lab = tmp_path / "in" / "images" / names[0], tmp_path / "in" / "labels" / names[0]
    r = run("snap_cube.py", tmp_path / "cube", img, lab, "--param", "keep_topology=true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / "cube" / (names[0][:-4] + ".json")))["params"]["keep_topology"] is True
    band = tifffile.imread(lab) == 1
    kept = tifffile.imread(tmp_path / "cube" / "labels" / names[0])
    for conn in (6, 26):
        assert betti(kept == 1, conn) == betti(band, conn)

    r = run("snap_folder.py", tmp_path / "in", tmp_path / "folder", "--workers", 1, "--param", "keep_topology=true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert np.array_equal(tifffile.imread(tmp_path / "folder" / "labels" / names[0]), kept)
    assert json.load(open(tmp_path / "folder" / "params.json")) == dict(keep_topology=True)
    # cubes made with other settings are not mixed in, either way round
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "folder", "--workers", 1)
    assert r.returncode != 0 and "other --param settings" in r.stderr
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "plain", "--workers", 1)
    assert r.returncode == 0 and not (tmp_path / "plain" / "params.json").exists()
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "plain", "--workers", 1, "--param", "keep_topology=true")
    assert r.returncode != 0 and "other --param settings" in r.stderr

    for tool, args in (("snap_cube.py", (tmp_path / "cube", img, lab)), ("snap_folder.py", (tmp_path / "in", tmp_path / "x"))):
        r = run(tool, *args, "--param", "no_such_thing=1")
        assert r.returncode != 0 and "unknown parameter" in r.stderr
