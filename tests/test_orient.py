"""orient_normals(orient_cube=True) on small cases where the recto side is known. The default cross-sheet step
gets three of these cases wrong, which is why the option exists; the default must keep doing so until the
option has passed its test (results/orient_test_plan.md)."""
import itertools
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


def test_recto_is_found_under_every_mirror_and_axis_order():
    """The axis from the eigenvectors can point either way: only the votes say which is the recto side."""
    pts, n, i, j, ct_s = wavy_phantom()
    for perm in ((0, 1, 2), (1, 0, 2), (0, 2, 1)):
        for flips in itertools.product((False, True), repeat=3):
            c, q, m = ct_s.transpose(perm), pts[:, perm].copy(), n[:, perm].copy()
            recto = np.zeros(3)
            recto[perm.index(1)] = 1.0
            for axis, flip in enumerate(flips):
                if flip:
                    c = np.flip(c, axis)
                    q[:, axis] = c.shape[axis] - 1 - q[:, axis]
                    m[:, axis] *= -1
                    recto[axis] *= -1
            out, _ = orient_normals(q, m, i, j, c.copy(), orient_cube=True)
            assert np.mean(out @ recto > 0) > 0.99, (perm, flips)


def test_a_large_sheet_outvotes_more_small_ones():
    """One large sheet and three flakes labelled on their far faces: the flakes are more sheets, but fewer samples."""
    ct = np.full((96, 200, 200), 60, np.float32)
    ct[30:40] = 170.0
    starts = (45, 96, 145)
    for a in starts:
        ct[48:54, a - 1:a + 14, a - 1:a + 14] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    big = flat_sheet(40, 40, 157)
    pts = np.vstack([big] + [flat_sheet(47.5, a, a + 9) for a in starts])
    n = orient_points(pts, ct)
    assert (n[:, 0] > 0).all()


def test_a_sheet_across_the_axis_keeps_its_vote():
    """A plate leaning 11.5 degrees off being perpendicular to the cube's axis: its mean along the axis is -0.2, not far
    enough to be on the other side of it. Its vote is known from the CT."""
    lean = np.array([-0.2, 0.0, np.sqrt(1 - 0.04)])          # z, y, x: mostly along x, a little against +z
    z, y, x = np.mgrid[:96, :200, :200].astype(np.float32)
    origin = np.array([50.0, 150.0, 100.0])
    depth = (z - origin[0]) * lean[0] + (y - origin[1]) * lean[1] + (x - origin[2]) * lean[2]
    ct = np.full((96, 200, 200), 60, np.float32)
    ct[30:40] = 170.0
    ct[(depth > -8) & (depth < 0) & (y > 125) & (y < 175) & (z > 45) & (z < 90)] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    big = flat_sheet(40, 10, 120)
    across = np.array([np.sqrt(1 - 0.04), 0.0, 0.2])         # in the plate, at right angles to its normal
    a, b = np.meshgrid(np.arange(-15, 16, 3.0), np.arange(-15, 16, 3.0), indexing="ij")
    plate = (origin[None] + a.ravel()[:, None] * across + b.ravel()[:, None] * np.array([0, 1.0, 0])).astype(np.float32)
    plate[:, 0] += 18                                        # all of it away from the large sheet
    plate[:, 2] += 18 * 0.2 / np.sqrt(1 - 0.04)              # and still on the plate's face
    pts = np.vstack([big, plate])
    raw = np.vstack([np.tile([1.0, 0, 0], (len(big), 1)), np.tile(lean, (len(plate), 1))]).astype(np.float32)
    i, j = neighbour_graph(pts, raw, Params())
    n, comp = orient_normals(pts, raw, i, j, ct, orient_cube=True)
    assert comp.max() == 1 and (n[: len(big), 0] > 0).all()
    assert (n[len(big):] @ lean.astype(np.float32) > 0).all()


def plates_with_a_share(w_a, w_c, w_b):
    """Plates A and C with normals along y, and B with normals along x, of those widths: the normals' share on the y axis is
    (w_a + w_c) / (w_a + w_c + w_b). A is labelled on its +y face and C on its -y face, so C votes against A, and A is larger."""
    ct = np.full((60, 200, 200), 60, np.float32)
    ct[:, 20:30] = ct[:, 150:160] = 170.0
    ct[:, 40:118, 20:30] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    zs = np.arange(8.0, 50, 3)

    def plate(fixed_axis, level, start, width):
        z, b = np.meshgrid(zs, start + 3.0 * np.arange(width), indexing="ij")
        q = np.zeros((z.size, 3), np.float32)
        q[:, 0], q[:, fixed_axis] = z.ravel(), level
        q[:, 3 - fixed_axis] = b.ravel()
        return q
    pts = np.vstack([plate(1, 30, 35, w_a), plate(1, 149.5, 35, w_c), plate(2, 30, 40, w_b)])
    sizes = [w_a * len(zs), w_c * len(zs), w_b * len(zs)]
    raw = np.zeros((len(pts), 3), np.float32)
    raw[: sizes[0] + sizes[1], 1] = 1.0
    raw[sizes[0] + sizes[1]:, 2] = 1.0
    return pts, raw, ct, sizes


def test_guard_lies_between_its_two_phantoms():
    # 0.675 of the normals' squared projection on the axis: under the guard, so every sheet keeps its vote
    pts, raw, ct, (a, c, b) = plates_with_a_share(28, 26, 26)
    assert (28 + 26) / (28 + 26 + 26) == 0.675
    i, j = neighbour_graph(pts, raw, Params())
    n, _ = orient_normals(pts, raw, i, j, ct, orient_cube=True)
    assert (n[:a, 1] > 0).all() and (n[a:a + c, 1] < 0).all() and (n[a + c:, 2] > 0).all()
    # 0.797: over it, so the sheet that disagrees takes the axis's side
    pts, raw, ct, (a, c, b) = plates_with_a_share(52, 50, 26)
    assert abs((52 + 50) / (52 + 50 + 26) - 0.797) < 0.001
    i, j = neighbour_graph(pts, raw, Params())
    n, _ = orient_normals(pts, raw, i, j, ct, orient_cube=True)
    assert (n[:a, 1] > 0).all() and (n[a:a + c, 1] > 0).all() and (n[a + c:, 2] > 0).all()


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
