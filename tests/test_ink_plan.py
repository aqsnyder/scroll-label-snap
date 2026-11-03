"""tools/ink_plan.py: the fixed decisions of the pre-registered ink test. The patch rules must match
villa's flat inference, and the renderer-check arithmetic must turn a measured layer alignment into
the render settings that undo it."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import ink_plan as ip  # noqa: E402

TOOL = os.path.join(HERE, "..", "tools", "ink_plan.py")


def test_scheduling_copies():
    assert ip.sliding_positions(100, 128, 64) == [0]
    assert ip.sliding_positions(384, 128, 64) == [0, 64, 128, 192, 256]
    assert ip.sliding_positions(400, 128, 64) == [0, 64, 128, 192, 256, 272]
    assert list(ip.center_crop(21, 17)) == list(range(2, 19))
    assert list(ip.center_crop(64, 17)) == list(range(24, 41))   # upper centre for even excess
    assert list(ip.center_crop(65, 17)) == list(range(24, 41))
    assert list(ip.center_crop(15, 17)) == list(range(15))


def brute_count(occ, roi=None):
    H, W = occ.shape
    n = 0
    for y0 in ip.sliding_positions(H, 128, 64):
        for x0 in ip.sliding_positions(W, 128, 64):
            y1, x1 = y0 + min(128, H - y0), x0 + min(128, W - x0)
            if roi is not None and (y0 >= roi[1] or y1 <= roi[0] or x0 >= roi[3] or x1 <= roi[2]):
                continue
            n += bool(occ[y0:y1, x0:x1].any())
    return n


def test_count_patches_matches_brute_force():
    rng = np.random.default_rng(1)
    occ = np.zeros((700, 530), bool)
    occ[100:400, 50:300] = True
    occ[rng.integers(0, 700, 30), rng.integers(0, 530, 30)] = True
    for roi in (None, [200, 457, 0, 530], [0, 64, 0, 64], [650, 700, 500, 530]):
        assert ip.count_patches(occ, roi=roi) == brute_count(occ, roi)
    assert ip.count_patches(np.zeros((300, 300), bool)) == 0


def check(L_ref, order, off, corr=0.95, L=21):
    return dict(ours_shape=[L, 10, 10], reference_shape=[L_ref, 10, 10], inplane_shift_yx=[1, -2],
                best_layer_alignment=dict(order=order, ref_layer_of_our_layer_0=off, mean_corr=corr))


def realign(L_ref, order, off, normals, center, L=21):
    """Where our new layers land in the reference after rendering with (normals, center): the
    alignment the check would then find (assuming the reference layers are 1 voxel apart)."""
    # positions along the ORIGINAL normals of new layer i, and the old layer index found there
    s = 1 if normals == "grid" else -1
    pos = [s * (center + (i - (L - 1) / 2)) for i in range(L)]
    old = [p + (L - 1) / 2 for p in pos]
    ref = [off + order * o for o in old]
    new_order = "same" if ref[1] - ref[0] > 0 else "reversed"
    return new_order, ref[0], ref


@pytest.mark.parametrize("L_ref,order,off", [(21, "same", 0), (65, "same", 22), (65, "reversed", 42),
                                             (21, "same", 2), (21, "reversed", 18), (64, "same", 20),
                                             (40, "reversed", 25)])
def test_render_decision_undoes_the_alignment(L_ref, order, off):
    d = ip.render_decision(check(L_ref, order, off))
    o = 1 if order == "same" else -1
    if d["verdict"] == "pass":
        assert order == "same" and abs(d["its_offset_along_our_normals"]) <= 0.5
        return
    assert d["verdict"] == "re-render"
    new_order, ref0, ref = realign(L_ref, o, off, d["normals"], d["center"])
    assert new_order == "same"
    # after re-rendering, our layer 10 (the middle of the 17 the model reads) is the reference's
    assert abs(ref[10] - L_ref // 2) < 1e-9
    d2 = ip.render_decision(check(L_ref, new_order, int(round(ref0))), d["normals"], d["center"])
    assert d2["verdict"] == "pass"


def test_render_decision_needs_correlation():
    assert ip.render_decision(check(21, "same", 0, corr=0.89))["verdict"] == "fail"
    assert ip.render_decision(check(21, "same", 0, corr=0.9))["verdict"] == "pass"
    assert ip.render_decision(check(21, "reversed", 20, corr=0.5))["verdict"] == "re-render"


def test_crop_window_and_middle_rows():
    occ = np.zeros((3000, 5000), bool)
    occ[1000:2600, 200:4000] = True
    roi, centre = ip.crop_window(occ, 1024)
    assert centre == [1800, 2100] and roi == [1288, 2312, 1588, 2612]
    occ2 = np.zeros((900, 900), bool)
    occ2[850:, 850:] = True
    assert ip.crop_window(occ2, 1024)[0] == [0, 900, 0, 900]
    occ3 = np.zeros((3000, 3000), bool)
    occ3[2900:, :100] = True
    assert ip.crop_window(occ3, 1024)[0] == [1976, 3000, 0, 1024]
    assert ip.middle_rows_roi(dict(pixels_per_vertex=20, rows=[0, 385]), (7681, 9741), [96, 288]) == \
        [1920, 5761, 0, 9741]


zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")


def test_cli_crop_and_choose(tmp_path):
    vol = np.zeros((21, 1500, 1300), np.uint8)
    vol[:, 200:1400, 100:1200] = 90
    z = zarr.open_array(str(tmp_path / "r.zarr"), mode="w", shape=vol.shape, chunks=(21, 128, 128), dtype="u1")
    z[:] = vol
    z.attrs.update(pixels_per_vertex=20, rows=[0, 76])
    subprocess.run([sys.executable, TOOL, "crop", str(tmp_path / "r.zarr"), str(tmp_path / "crop.json"),
                    "--middle-rows", "20", "50"], check=True)
    crop = json.load(open(tmp_path / "crop.json"))
    r0, r1, c0, c1 = crop["roi"]
    assert (r1 - r0, c1 - c0) == (1024, 1024) and crop["crop_valid_fraction"] > 0.9
    occ = (vol[2:19] > 0).any(0)
    assert crop["patches_whole"] == brute_count(occ)
    assert crop["patches_crop"] == brute_count(occ, crop["roi"])
    assert crop["middle_roi"] == [400, 1001, 0, 1300]
    rng = np.random.default_rng(0)
    fwd = rng.integers(60, 150, (1024, 1024)).astype(np.uint8)
    rev = fwd.copy()
    rev[:150, :150] = 200                        # a brighter top 2% in the reverse map
    for name, im, sec in (("f", fwd, 30.0), ("r", rev, 34.0)):
        tifffile.imwrite(tmp_path / f"{name}.tif", im)
        json.dump(dict(seconds=sec, status="done"), open(tmp_path / f"{name}.json", "w"))
    subprocess.run([sys.executable, TOOL, "choose", str(tmp_path / "crop.json"), str(tmp_path / "d.json"),
                    "--forward", str(tmp_path / "f.tif"), str(tmp_path / "f.json"),
                    "--reverse", str(tmp_path / "r.tif"), str(tmp_path / "r.json"),
                    "--project", str(tmp_path / "r.zarr"), str(tmp_path / "r.zarr"),
                    "--budget-hours", "8"], check=True)
    d = json.load(open(tmp_path / "d.json"))
    assert d["direction"] == "reverse" and d["scope"] == "whole"
    assert d["patches_whole"] == 2 * crop["patches_whole"]
    assert abs(d["projected_hours_whole"] - 32.0 / crop["patches_crop"] * d["patches_whole"] / 3600) < 0.01
    subprocess.run([sys.executable, TOOL, "choose", str(tmp_path / "crop.json"), str(tmp_path / "d2.json"),
                    "--forward", str(tmp_path / "f.tif"), str(tmp_path / "f.json"),
                    "--reverse", str(tmp_path / "f.tif"), str(tmp_path / "r.json"),
                    "--project", str(tmp_path / "r.zarr"), "--budget-hours", "0.001"], check=True)
    d2 = json.load(open(tmp_path / "d2.json"))
    assert d2["direction"] == "forward" and d2["scope"] in ("middle rows", "stop")
