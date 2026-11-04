"""The fine output grid (mesh.fine_surface, snapped_fine.tifxyz, only with --fine-grid): the fit, what the
tools write, and that it changes nothing they wrote before. Run with: python -m pytest tests/test_fine_grid.py"""
import filecmp
import json
import os
import re
import sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
import eval_mesh  # noqa: E402
from eval_mesh import evaluate, output_on_dense  # noqa: E402
from regression_cases import mesh_case  # noqa: E402
from render_surface_volume import tile_pixels  # noqa: E402
from snap_tifxyz import run_one  # noqa: E402
from test_robustness import run  # noqa: E402
from scroll_label_snap.core import Params  # noqa: E402
from scroll_label_snap.mesh import (dense_at_fine, densify, fine_factor, fine_surface, fit_vertex_shifts,  # noqa: E402
                                    snap_mesh, surface_area, vertex_normals)
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def wavy_field(A, B):
    return 1.5 * np.sin(A / 1.3) * np.cos(B / 0.9)


def gap_to_dense(res, Q, v, g):
    N, ok = vertex_normals(Q, v)
    Pd, _, vd = densify(Q, N, ok, res.factor, g=g)
    return np.linalg.norm(Pd - res.dense_P, axis=-1)[vd & res.dense_valid]


def test_factor_is_capped_by_the_dense_samples():
    assert fine_factor(20.0, 7) == 4                 # 20-voxel grid, 7 dense samples per step: 5-voxel grid
    assert fine_factor(20.0, 7, voxels=10) == 2
    assert fine_factor(20.0, 3) == 3                 # no finer than the dense samples
    assert fine_factor(4.0, 1) == 1


def test_fit_recovers_a_field_the_fine_grid_can_hold():
    ok = np.ones((4, 5), bool)
    f, g = 7, 4
    Hg, Wg = 3 * g + 1, 4 * g + 1
    truth = wavy_field(*np.mgrid[:Hg, :Wg].astype(np.float64)).astype(np.float32)
    N = np.zeros((4, 5, 3), np.float32)
    N[..., 1] = 1
    Z, X = np.meshgrid(np.arange(4.0) * 20, np.arange(5.0) * 20, indexing="ij")
    P = np.stack([Z, np.zeros_like(Z), X], -1)
    # the dense shifts are exactly the bilinear reading of the truth
    td, _, vd = densify(np.repeat(truth[..., None], 3, -1), np.repeat(np.ones((Hg, Wg, 1), np.float32), 3, -1), np.ones((Hg, Wg), bool), f, g=g)
    got = fit_vertex_shifts(np.ones((Hg, Wg), bool), f, vd, td[..., 0], g=g)
    assert np.abs(got - truth).max() < 1e-2
    assert got.shape == (Hg, Wg)
    Pg, vg, s = fine_surface(P.astype(np.float32), N, ok, f, vd, td[..., 0], g)
    assert vg.all() and Pg.shape == (Hg, Wg, 3) and np.abs(s - truth).max() < 1e-2


def test_factor_one_is_the_input_grid_fit():
    ct, P, v = mesh_case()
    res = snap_mesh(ct, P, v, Params(), fine=20.0)
    assert res.fine_factor == 1
    assert np.array_equal(res.fine_valid, res.valid)
    assert np.abs(res.fine_P[res.valid] - res.P[res.valid]).max() < 1e-2


def test_fine_grid_keeps_what_the_input_grid_loses():
    ct, P, v = mesh_case()
    res = snap_mesh(ct, P, v, Params(), fine=5.0)
    assert res.fine_factor == 4 and res.fine_P.shape[:2] == ((P.shape[0] - 1) * 4 + 1, (P.shape[1] - 1) * 4 + 1)
    coarse = gap_to_dense(res, res.P, res.valid, 1)
    fine = gap_to_dense(res, res.fine_P, res.fine_valid, 4)
    assert np.percentile(fine, 95) < 0.5 * np.percentile(coarse, 95)
    assert np.percentile(fine, 95) < 0.6


def test_fine_grid_on_a_coarser_scan_level():
    ct, P, v = mesh_case()
    res = snap_mesh(ct[::2, ::2, ::2], P, v, Params(), level_scale=0.5, fine=5.0)
    assert res.factor == 3 and res.fine_factor == 2         # 10 crop voxels between vertices
    assert np.percentile(gap_to_dense(res, res.fine_P, res.fine_valid, 2), 95) < 1.5


def test_holes_stay_holes():
    ct, P, v = mesh_case()
    v = v.copy()
    v[2, 2] = False
    P = P.copy()
    P[2, 2] = -1
    res = snap_mesh(ct, P, v, Params(), fine=5.0)
    g = res.fine_factor
    assert not res.fine_valid[(2 - 1) * g + 1:(2 + 1) * g, (2 - 1) * g + 1:(2 + 1) * g].any()
    assert (res.fine_P[~res.fine_valid] == -1).all() and np.isfinite(res.fine_P).all()
    assert res.fine_valid[0, 0] and res.fine_valid[-1, -1]


def test_dense_at_fine_reads_the_nearest_sample():
    f, g = 7, 4
    dense = np.arange(((3 * f + 1) * (4 * f + 1))).reshape(3 * f + 1, 4 * f + 1)
    out = dense_at_fine(dense, f, g)
    assert out.shape == (3 * g + 1, 4 * g + 1)
    assert out[0, 0] == dense[0, 0] and out[-1, -1] == dense[-1, -1]
    assert out[4, 8] == dense[7, 14] and out[1, 0] == dense[2, 0]     # 7 / 4 = 1.75 rounds to 2


def test_surface_area():
    Z, X = np.meshgrid(np.arange(3.0) * 20, np.arange(4.0) * 20, indexing="ij")
    P = np.stack([Z, np.zeros_like(Z), X], -1).astype(np.float32)
    v = np.ones(Z.shape, bool)
    assert abs(surface_area(P, v) - 40 * 60) < 1e-6
    v[0, 0] = False
    assert abs(surface_area(P, v) - (40 * 60 - 400)) < 1e-6


def write_case(tmp_path):
    ct, P, v = mesh_case()
    seg = str(tmp_path / "seg.tifxyz")
    write_tifxyz(seg, P, v, {"scale": [0.05, 0.05], "area_vx2": 1.0}, uuid="synth")
    vol = str(tmp_path / "ct.tif")
    tifffile.imwrite(vol, ct)
    return seg, vol, P


def same_folder(a, b):
    names = sorted(os.listdir(a))
    assert names == sorted(os.listdir(b))
    for n in names:
        assert filecmp.cmp(os.path.join(a, n), os.path.join(b, n), shallow=False), n


def test_snap_tifxyz_adds_the_fine_grid_and_leaves_the_rest(tmp_path):
    seg, vol, P = write_case(tmp_path)
    p = Params()
    run_one(str(tmp_path / "on"), "m", seg, vol, [0, 0, 0], 1.0, 0, p, fine=5.0)
    run_one(str(tmp_path / "off"), "m", seg, vol, [0, 0, 0], 1.0, 0, p)                # the default writes no fine grid
    on, off = tmp_path / "on" / "m", tmp_path / "off" / "m"
    assert not (off / "snapped_fine.tifxyz").exists()
    assert json.load(open(off / "snap.json"))["fine_factor"] == 0
    same_folder(on / "snapped.tifxyz", off / "snapped.tifxyz")
    same_folder(on / "snapped_dense.tifxyz", off / "snapped_dense.tifxyz")
    Q, v, meta = read_tifxyz(str(on / "snapped_fine.tifxyz"))
    assert Q.shape[:2] == ((P.shape[0] - 1) * 4 + 1, (P.shape[1] - 1) * 4 + 1)
    assert meta["scale"] == [0.2, 0.2] and meta["uuid"] == "synth_snapped_fine" and meta["snapped_from"] == "synth"
    assert abs(meta["area_vx2"] - surface_area(Q, v)) < 1e-3
    for name, dtype in (("snap_shift", np.float32), ("snap_confidence", np.uint8), ("snap_flag", np.uint8), ("snap_limit", np.uint8)):
        img = tifffile.imread(str(on / "snapped_fine.tifxyz" / f"{name}.tif"))
        assert img.shape == v.shape and img.dtype == dtype, name
    assert json.load(open(on / "snap.json"))["fine_factor"] == 4


def test_fine_grid_channels_hold_what_their_names_say(tmp_path):
    seg, vol, P = write_case(tmp_path)
    p = Params(t_max=2.0)                                   # a short search, so some samples stop at its limit
    run_one(str(tmp_path / "w"), "m", seg, vol, [0, 0, 0], 1.0, 0, p, fine=5.0)
    res = snap_mesh(tifffile.imread(vol), P, np.ones(P.shape[:2], bool), p, fine=5.0)
    f, g = res.factor, res.fine_factor
    channel = lambda name: tifffile.imread(str(tmp_path / "w" / "m" / "snapped_fine.tifxyz" / f"{name}.tif"))  # noqa: E731
    assert np.abs(res.fine_shift).max() > 0.5 and np.array_equal(channel("snap_shift"), res.fine_shift.astype(np.float32))
    conf = (dense_at_fine(res.dense_confidence, f, g) * 255).astype(np.uint8)
    assert conf.any() and np.array_equal(channel("snap_confidence"), conf)
    limit = (dense_at_fine(res.dense_limit, f, g) * 255).astype(np.uint8)
    assert limit.any() and np.array_equal(channel("snap_limit"), limit)
    assert np.array_equal(channel("snap_flag"), (dense_at_fine(res.dense_flag, f, g) * 255).astype(np.uint8))
    assert not channel("snap_flag").any()                   # no reversed faces here, whatever the limit map holds


def test_fine_grid_values(tmp_path):
    seg, vol, _ = write_case(tmp_path)
    common = ("--tifxyz", seg, "--volume", vol, "--normals", "flipped")
    for bad in ("-5", "inf", "nan", "abc"):
        r = run("snap_tifxyz.py", tmp_path / "bad", *common, f"--fine-grid={bad}")
        assert r.returncode != 0 and "--fine-grid" in r.stderr and "Traceback" not in r.stderr and not (tmp_path / "bad").exists(), bad
        r = run("snap_segment.py", tmp_path / "bad", "--segment", seg, "--volume", vol, "--normals", "flipped", f"--fine-grid={bad}")
        assert r.returncode != 0 and "--fine-grid" in r.stderr and "Traceback" not in r.stderr and not (tmp_path / "bad").exists(), bad
    # what can't be had is reported: the input's spacing at most, the dense samples' at least
    for asked, low, high in (("100", 19.0, 23.0), ("0.5", 2.7, 3.3)):
        r = run("snap_tifxyz.py", tmp_path / asked, *common, f"--fine-grid={asked}")
        apart = re.search(r"gives vertices ([\d.]+) voxels apart", r.stdout)
        assert r.returncode == 0 and apart and low < float(apart.group(1)) < high, r.stdout + r.stderr
    r = run("snap_tifxyz.py", tmp_path / "five", *common, "--fine-grid=5")
    assert r.returncode == 0 and "gives vertices" not in r.stdout
    r = run("snap_tifxyz.py", tmp_path / "none", *common, "--fine-grid=0")
    assert r.returncode == 0 and not (tmp_path / "none" / "seg" / "snapped_fine.tifxyz").exists()


def test_wildcards_reach_the_patches_and_same_names_are_refused(tmp_path):
    ct, P, v = mesh_case()
    for s in ("a", "b"):
        d = tmp_path / "segs" / s / "patch_0"
        d.mkdir(parents=True)
        write_tifxyz(str(d / "segment.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="synth" + s)
        tifffile.imwrite(str(d / "volume.tif"), ct)
        json.dump(dict(offset_zyx=[0, 0, 0], scale_to_level=1.0, recto_sign=-1), open(d / "patch.json", "w"))
    r = run("snap_tifxyz.py", tmp_path / "one", tmp_path / "segs" / "a" / "patch_*")
    assert r.returncode == 0 and (tmp_path / "one" / "patch_0" / "snapped.tifxyz").exists(), r.stdout + r.stderr
    r = run("snap_tifxyz.py", tmp_path / "two", tmp_path / "segs" / "*" / "patch_0")
    assert r.returncode != 0 and "patch_0" in r.stderr and "overwrite" in r.stderr and not (tmp_path / "two").exists()


def test_fine_grid_renders_on_the_input_grids_pixels():
    """render_surface_volume uses round(1 / scale) pixels per vertex: 20 on the input grid, 5 on the
    fine one, so they cover the same pixels and agree where the fine grid holds the input's vertices."""
    ct, P, v = mesh_case()
    res = snap_mesh(ct, P, v, Params(), fine=5.0)
    N0, ok0 = vertex_normals(P, v)
    Nf, okf = vertex_normals(res.fine_P, res.fine_valid)
    H, W = (P.shape[0] - 1) * 20 + 1, (P.shape[1] - 1) * 20 + 1
    a = tile_pixels(P, N0, ok0, 20, 0, H, 0, W)
    b = tile_pixels(res.fine_P, Nf, okf, 5, 0, H, 0, W)
    assert a[0].shape == b[0].shape == (H, W, 3)
    d = np.linalg.norm(a[0] - b[0], axis=-1)[a[2] & b[2]]
    assert d.max() < 8.5                                    # the search range plus a little, never a misregistered pixel
    assert np.median(d) < 2.5


def test_eval_mesh_scores_the_output_it_names(tmp_path):
    ct, P, v = mesh_case()
    d = tmp_path / "patch_0"
    d.mkdir()
    write_tifxyz(str(d / "segment.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="synth")
    tifffile.imwrite(str(d / "volume.tif"), ct)
    json.dump(dict(offset_zyx=[0, 0, 0], scale_to_level=1.0, recto_sign=0), open(d / "patch.json", "w"))
    p = Params()
    r = {o: evaluate(str(d), p, 0, output=o) for o in ("dense", "fine", "grid")}
    assert [r[o]["output"] for o in r] == ["dense", "fine", "grid"]
    plain = evaluate(str(d), p, 0)
    assert plain["drift_test"]["snapper consistency"] == r["dense"]["drift_test"]["snapper consistency"]
    assert plain["face_alignment"] == r["dense"]["face_alignment"]
    # the original mesh and the refiner do not depend on which output is scored
    for o in ("fine", "grid"):
        assert r[o]["face_alignment"]["original"] == r["dense"]["face_alignment"]["original"]
        assert r[o]["drift_test"]["baseline consistency"] == r["dense"]["drift_test"]["baseline consistency"]
    c = {o: r[o]["drift_test"]["snapper consistency"]["mean"] for o in r}
    assert c["fine"] != c["dense"] and c["grid"] != c["dense"]
    a = {o: r[o]["face_alignment"]["snapped"]["mean_step"] for o in r}
    assert a["grid"] < a["fine"]


def test_eval_mesh_scores_only_where_the_output_exists(tmp_path, monkeypatch):
    ct, P, v = mesh_case()
    P, v = P.copy(), v.copy()
    P[2, 2] = -1
    v[2, 2] = False
    d = tmp_path / "patch_0"
    d.mkdir()
    write_tifxyz(str(d / "segment.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="synth")
    tifffile.imwrite(str(d / "volume.tif"), ct)
    json.dump(dict(offset_zyx=[0, 0, 0], scale_to_level=1.0, recto_sign=0), open(d / "patch.json", "w"))
    p = Params()
    whole = evaluate(str(d), p, 0, output="grid")["drift_test"]["snapper consistency"]["n"]
    real = eval_mesh.output_on_dense

    def with_a_gap(res, output):                            # samples this output does not hold are not scored
        Y, vy = real(res, output)
        vy = vy.copy()
        vy[:6] = False
        return Y, vy
    monkeypatch.setattr(eval_mesh, "output_on_dense", with_a_gap)
    assert evaluate(str(d), p, 0, output="grid")["drift_test"]["snapper consistency"]["n"] < whole


def test_output_on_dense_matches_the_written_files(tmp_path):
    seg, vol, P = write_case(tmp_path)
    p = Params()
    run_one(str(tmp_path / "w"), "m", seg, vol, [0, 0, 0], 1.0, 0, p, fine=5.0)
    res = snap_mesh(tifffile.imread(vol), P, np.ones(P.shape[:2], bool), p, fine=5.0)
    for o, fn in (("dense", "snapped_dense"), ("grid", "snapped"), ("fine", "snapped_fine")):
        Q, v, _ = read_tifxyz(str(tmp_path / "w" / "m" / f"{fn}.tifxyz"))
        g = (Q.shape[0] - 1) // (P.shape[0] - 1)
        Y, vy = output_on_dense(res, o)
        if o == "dense":
            assert np.abs(Y[vy] - Q[v]).max() < 1e-3
        else:
            N, ok = vertex_normals(Q, v)
            Z, _, vz = densify(Q, N, ok, res.factor, g=g)
            assert np.array_equal(vy, vz) and np.abs(Y[vy] - Z[vz]).max() < 1e-3
