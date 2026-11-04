"""orient_normals(orient_cube=True) on small cases where the recto side is known. The default cross-sheet step
gets three of these cases wrong, which is why the option exists; the default must keep doing so until the
option has passed its test (results/orient_test_plan.md)."""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regression_cases import label_case, wavy_sheets  # noqa: E402
from scroll_label_snap.core import Params, neighbour_graph, orient_normals, prepare, surface_points  # noqa: E402
from test_robustness import mini_set, run  # noqa: E402


def flat_sheet(z, lo, hi, step=3):
    """Samples of a flat sheet at height z, every `step` voxels."""
    a, b = np.meshgrid(np.arange(lo, hi, step), np.arange(lo, hi, step), indexing="ij")
    return np.stack([np.full(a.size, float(z)), a.ravel(), b.ravel()], 1).astype(np.float32)


def orient_points(pts, ct, orient_cube=True):
    n = np.tile([1.0, 0, 0], (len(pts), 1)).astype(np.float32)
    i, j = neighbour_graph(pts, n, Params())
    return orient_normals(pts, n, i, j, ct, orient_cube=orient_cube)[0]


def big_sheet_and_flake():
    """1,521 samples with a unanimous CT vote, and next to them a 9-sample flake labelled on its far face."""
    ct = np.full((96, 200, 200), 60, np.float32)
    ct[30:40] = 170.0
    ct[48:54, 95:110, 95:110] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    big, flake = flat_sheet(40, 40, 157), flat_sheet(47.5, 96, 105)
    return np.vstack([big, flake]), len(big), ct


def facing_sheets():
    """Two equal sheets labelled on the faces that look at each other: the CT votes disagree, and flipping
    both at once would leave them disagreeing."""
    ct = np.full((64, 64, 64), 60, np.float32)
    ct[20:26] = 170.0
    ct[36:42] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    return np.vstack([flat_sheet(26, 9, 55), flat_sheet(35.5, 9, 55)]), ct


def wavy_phantom():
    """Wavy sheets labelled on their +y faces, so every normal should point to +y, plus a thin flake in one gap
    labelled on its -y face, whose CT vote points the other way."""
    shape = (128, 128, 128)
    ct, faces, (z, y, x) = wavy_sheets(shape, n_sheets=4, base=24.0)
    lab = np.zeros(shape, np.uint8)
    for f in faces:
        lab[(y >= f - 1.0) & (y < f + 1.0)] = 1
    patch = (z >= 50) & (z < 74) & (x >= 50) & (x < 74)
    ct[patch & (y >= faces[1] + 5) & (y < faces[1] + 8)] = 170
    lab[patch & (y >= faces[1] + 4) & (y < faces[1] + 6)] = 1
    p = Params()
    ct_s = ndi.gaussian_filter(ct.astype(np.float32), p.sigma_ct)
    pts, n, _ = surface_points(lab, p)
    i, j = neighbour_graph(pts, n, p)
    return pts, n, i, j, ct_s


def test_fragment_does_not_reverse_a_large_sheet():
    pts, n_big, ct = big_sheet_and_flake()
    n = orient_points(pts, ct)
    assert (n[:n_big, 0] > 0).all()
    assert (n[n_big:, 0] > 0).all()


def test_sheets_that_disagree_end_up_agreeing():
    pts, ct = facing_sheets()
    n = orient_points(pts, ct)
    assert abs(n[:, 0].sum()) == len(n)


def test_phantom_with_known_recto():
    pts, n, i, j, ct_s = wavy_phantom()
    out, _ = orient_normals(pts, n, i, j, ct_s, orient_cube=True)
    assert np.mean(out[:, 1] > 0) > 0.99


def test_normals_with_no_single_axis_keep_their_votes():
    """Near the scroll centre normals point every way: four plates labelled on the faces toward a common centre."""
    ct = np.full((60, 160, 160), 60, np.float32)
    ct[:, 20:30] = ct[:, 130:140] = 170.0
    ct[:, :, 20:30] = ct[:, :, 130:140] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    a, b = np.meshgrid(np.arange(10.0, 50, 3), np.arange(40.0, 120, 3), indexing="ij")
    a, b = a.ravel(), b.ravel()
    plates = [(a, np.full(a.size, 30.0), b), (a, np.full(a.size, 130.0), b), (a, b, np.full(a.size, 30.0)), (a, b, np.full(a.size, 130.0))]
    pts = np.vstack([np.stack(q, 1) for q in plates]).astype(np.float32)
    raw = np.vstack([np.tile(v, (a.size, 1)) for v in ([0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0])]).astype(np.float32)
    i, j = neighbour_graph(pts, raw, Params())
    n, _ = orient_normals(pts, raw, i, j, ct, orient_cube=True)
    assert (np.einsum("ij,ij->i", n, np.array([0, 80.0, 80.0], np.float32) - pts * [0, 1, 1]) > 0).all()


def test_the_default_step_is_unchanged_on_those_cases():
    pts, n_big, ct = big_sheet_and_flake()
    n = orient_points(pts, ct, orient_cube=False)
    assert (n[:n_big, 0] < 0).all() and (n[n_big:, 0] > 0).all()         # the flake reverses the large sheet
    pts, ct = facing_sheets()
    n = orient_points(pts, ct, orient_cube=False)
    assert abs(n[:, 0].sum()) == 0                                       # both flip and still disagree
    pts, n, i, j, ct_s = wavy_phantom()
    out, _ = orient_normals(pts, n, i, j, ct_s)
    assert np.mean(out[:, 1] > 0) < 0.99
    assert np.array_equal(out, orient_normals(pts, n, i, j, ct_s, orient_cube=False)[0])


def test_params_turn_the_option_on():
    ct, lab = label_case()
    assert Params().orient_cube is False
    a = prepare(ct, lab, Params())
    b = prepare(ct, lab, Params(orient_cube=True))
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[5], b[5])     # same samples and sheets
    pts, n, i, j, ct_s = a[1], surface_points(lab, Params())[1], a[3], a[4], a[0]
    assert np.array_equal(a[2], orient_normals(pts, n, i, j, ct_s)[0])
    assert np.array_equal(b[2], orient_normals(pts, n, i, j, ct_s, orient_cube=True)[0])


def test_tools_set_the_option_with_param(tmp_path):
    names = mini_set(tmp_path / "in", 1)
    r = run("snap_cube.py", tmp_path / "cube", tmp_path / "in" / "images" / names[0], tmp_path / "in" / "labels" / names[0],
            "--param", "orient_cube=true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / "cube" / (names[0][:-4] + ".json")))["params"]["orient_cube"] is True
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "folder", "--workers", 1, "--param", "orient_cube=1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / "folder" / "params.json")) == dict(orient_cube=True)
