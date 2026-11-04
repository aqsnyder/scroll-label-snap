"""What the snapper and its tools do at the edges of normal use: --target-offset's sign and reach, flat bands,
empty or fully frozen inputs, interrupted and failing folder runs, the recto side of a short mesh, and a
few smaller input checks. Most of these run the real command-line tools on small synthetic data."""
import csv
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
from regression_cases import wavy_sheets  # noqa: E402
from scroll_label_snap.core import Params, estimate_convention, parse_params, snap, surface_points, warp_labels  # noqa: E402
from scroll_label_snap.mesh import axis_fit_is_meaningful, expand_paths, snap_mesh  # noqa: E402
from scroll_label_snap.tifxyz import densified_scale, read_tifxyz, write_tifxyz  # noqa: E402

ENV = dict(os.environ, SNAP_THREADS="1", PYTHONUTF8="1")


def run(tool, *args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, tool), *[str(a) for a in args]],
                          capture_output=True, text=True, env=ENV)


# --------------------------------------------------------------------------- target offset

BASE = 40.0
SHEET, GAP = 8, 10


def face_y(x, z):
    """y of the recto face of the third sheet, in the scene below; gap on the +y side."""
    return BASE + 5 * np.sin(2 * np.pi * x / 70) + 3 * np.cos(2 * np.pi * z / 55) + 2 * (SHEET + GAP) + SHEET


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A scan, as a tif and a zarr, and a coarse mesh on one recto face."""
    root = tmp_path_factory.mktemp("scene")
    ct, _, _ = wavy_sheets((150, 170, 190), thickness=SHEET, gap=GAP, blur=1.0, n_sheets=5, base=BASE)
    tifffile.imwrite(root / "vol.tif", ct)
    zs, xs = np.arange(30, 121, 15.0), np.arange(30, 161, 15.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    P = np.stack([Z, face_y(X, Z), X], -1).astype(np.float32)
    write_tifxyz(str(root / "seg.tifxyz"), P, np.ones(P.shape[:2], bool), {"scale": [1 / 15, 1 / 15]}, uuid="synth")
    return root, ct, P


def landed(path):
    """Where a snapped dense surface sits relative to the true face, along papyrus -> gap (median, voxels)."""
    D, vd, _ = read_tifxyz(str(path))
    q = D[vd]
    return float(np.median(q[:, 1] - face_y(q[:, 2], q[:, 0])))


@pytest.mark.parametrize("target", [-2.0, 2.0])
def test_target_offset_snap_tifxyz(scene, tmp_path, target):
    root, _, _ = scene
    r = run("snap_tifxyz.py", tmp_path, "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif",
            "--normals", "flipped", f"--target-offset={target}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert abs(landed(tmp_path / "seg" / "snapped_dense.tifxyz") - target) < 0.5


def run_segment(root, out, *args):
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    import zarr
    scan = root / "scan.zarr"
    if not scan.exists():
        ct = tifffile.imread(root / "vol.tif")
        g = zarr.open_group(zarr.DirectoryStore(str(scan), dimension_separator="/"), mode="w")
        g.create_dataset("0", data=ct, chunks=(64, 64, 64), compressor=None)
    return run("snap_segment.py", out, "--segment", root / "seg.tifxyz", "--volume", scan, "--tile", 8, "--overlap", 2,
               "--workers", 1, "--sample-tiles", 4, "--normals", "flipped", "--name", "s", *args)


def test_target_offset_snap_segment_and_stale_tiles(scene, tmp_path):
    root, _, _ = scene
    r = run_segment(root, tmp_path, "--target-offset=-2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert abs(landed(tmp_path / "s_snapped_dense.tifxyz") + 2.0) < 0.5
    summary = json.load(open(tmp_path / "snap_segment.json"))
    assert summary["convention_offset"] == 2.0 and summary["params"]["template_follows_offset"]
    # the finished tiles were made for another offset, so they must not be reused for this one
    r = run_segment(root, tmp_path, "--target-offset=2")
    assert r.returncode != 0 and "other settings" in r.stderr


def test_segment_default_keeps_the_template_at_each_tile(scene, tmp_path):
    root, _, _ = scene
    r = run_segment(root, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.load(open(tmp_path / "snap_segment.json"))
    assert not summary["params"]["template_follows_offset"]
    assert abs(landed(tmp_path / "s_snapped_dense.tifxyz") - (0.0 - summary["convention_offset"])) < 0.5


def test_convention_offset_moves_both_terms(scene):
    _, ct, P = scene
    valid = np.ones(P.shape[:2], bool)
    both = snap_mesh(ct, P, valid, Params(convention_offset=2.0), normal_sign=-1)
    edge = snap_mesh(ct, P, valid, Params(convention_offset=2.0, template_follows_offset=False), normal_sign=-1)

    def where(res):
        q = res.dense_P[res.dense_valid]
        return float(np.median(q[:, 1] - face_y(q[:, 2], q[:, 0])))
    assert abs(where(both) + 2.0) < 0.5                 # the mesh's edge is 2 voxels ahead of it: mesh inside the papyrus
    assert abs(where(edge) + 2.0) > abs(where(both) + 2.0)


# --------------------------------------------------------------------------- flat bands

@pytest.mark.parametrize("thick,off", [(2, 26), (3, 25), (4, 24), (5, 27)])
def test_flat_band_survives_a_zero_warp(thick, off):
    lab = np.zeros((64, 64, 64), np.uint8)
    lab[off:off + thick] = 1
    p = Params()
    pts, n, _ = surface_points(lab, p)
    assert np.array_equal(warp_labels(lab, pts, n, np.zeros(len(pts))), lab)


@pytest.mark.parametrize("slope,thick", [(0.01, 4.0), (0.1, 3.0)])
def test_nearly_flat_band_survives_a_zero_warp(slope, thick):
    z, _, x = np.mgrid[:64, :64, :64].astype(np.float32)
    lab = (np.abs(z - (30 + slope * (x - 32))) < thick / 2).astype(np.uint8)
    p = Params()
    pts, n, _ = surface_points(lab, p)
    assert np.array_equal(warp_labels(lab, pts, n, np.zeros(len(pts))), lab)


# --------------------------------------------------------------------------- empty or fully frozen

CT = np.random.default_rng(0).integers(40, 180, (64, 64, 64)).astype(np.uint8)


def test_estimate_convention_with_no_samples():
    empty = np.zeros((0, 3), np.float32)
    assert estimate_convention(CT.astype(np.float32), empty, empty, Params()) == (0.0, 1.0)


@pytest.mark.parametrize("region,value", [(None, 1), (slice(None), 2), ((3, 30, 30), 1), (slice(0, 2), 1)])
def test_labels_with_nothing_to_move_are_returned_unchanged(region, value):
    lab = np.zeros(CT.shape, np.uint8)
    if region is not None:
        lab[region] = value
    res = snap(CT, lab, Params())
    assert np.array_equal(res.labels, lab) and not res.movable.any()


def test_small_cube_is_returned_unchanged():
    lab = np.zeros((16, 16, 16), np.uint8)
    lab[7:10] = 1
    res = snap(CT[:16, :16, :16], lab, Params())
    assert np.array_equal(res.labels, lab)


def mesh_grid():
    zs, xs = np.arange(20, 45, 5.0), np.arange(20, 45, 5.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    return np.stack([Z, np.full_like(Z, 30.0), X], -1).astype(np.float32)


@pytest.mark.parametrize("case", ["all invalid", "one row", "outside the crop"])
def test_meshes_with_nothing_to_move(case):
    P, sign = mesh_grid(), 1
    valid = np.ones(P.shape[:2], bool)
    if case == "all invalid":
        P, valid = np.full_like(P, -1), np.zeros_like(valid)
    elif case == "one row":
        P, valid = P[:1], valid[:1]
    else:
        P = P + 1000
    res = snap_mesh(CT, P, valid, Params(), normal_sign=sign)
    assert res.info["n_movable"] == 0 and not np.abs(res.dense_shift).any()


def test_tile_with_no_samples_does_not_vote_on_the_offset():
    import snap_segment
    # a single row of vertices has no normals, so a tile with nothing to sample
    H, W = 12, 12
    z, x = np.meshgrid(np.arange(H) * 5.0 + 10, np.arange(W) * 5.0 + 10, indexing="ij")
    P = np.stack([z, np.full_like(z, 30.0), x], -1).astype(np.float32)
    valid = np.zeros((H, W), bool)
    valid[3] = True
    snap_segment._STATE.update(seg=(P, valid, {}), vol=CT)
    offset, error = snap_segment.tile_convention((0, H, 0, W, (0, H, 0, W)), Params(), 1.0, 8, 2, 1)
    assert offset is None and error


def test_segment_of_one_tile_can_set_its_own_offset(scene, tmp_path):
    root, _, _ = scene
    pytest.importorskip("zarr")
    r = run_segment(root, tmp_path, "--tile", 40, "--overlap", 2)     # the whole grid is a single tile
    assert r.returncode == 0, r.stdout + r.stderr


def test_segment_reports_a_tile_it_cannot_move(scene, tmp_path):
    root, ct, P = scene
    Q = P.copy()
    Q[:, 5:, 0] += 1000                                  # the far part of the mesh lies beyond the scan
    write_tifxyz(str(root / "beyond.tifxyz"), Q, np.ones(Q.shape[:2], bool), {"scale": [1 / 15, 1 / 15]}, uuid="beyond")
    pytest.importorskip("zarr")
    run_segment(root, tmp_path / "first")                # makes the zarr
    for _ in range(2):                                   # the second time, the tiles are already there
        r = run("snap_segment.py", tmp_path, "--segment", root / "beyond.tifxyz", "--volume", root / "scan.zarr", "--tile", 8,
                "--overlap", 2, "--workers", 1, "--sample-tiles", 4, "--normals", "flipped", "--name", "s")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "tiles failed" in r.stdout and json.load(open(tmp_path / "snap_segment.json"))["tiles_failed"]


# --------------------------------------------------------------------------- folder runs

def mini_set(root, k, empty=()):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    names = []
    for s in range(1, k + 1):
        ct, faces, (z, y, x) = wavy_sheets((48, 48, 48), n_sheets=2, base=8.0, seed=s)
        lab = np.zeros(ct.shape, np.uint8)
        if s not in empty:
            for f in faces:
                lab[(y >= f - 1.0) & (y < f + 1.0)] = 1
        name = f"sample_{s:05d}.tif"
        tifffile.imwrite(root / "images" / name, ct)
        tifffile.imwrite(root / "labels" / name, lab)
        names.append(name)
    return names


def rows_of(out):
    with open(out / "summary.csv", newline="") as f:
        return sorted(r["cube"] for r in csv.DictReader(f))


@pytest.fixture(scope="module")
def folder_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("folder")
    names = mini_set(root / "in", 3)
    r = run("snap_folder.py", root / "in", root / "out", "--workers", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    return root, names


def test_folder_run_leaves_no_temporary_files(folder_run):
    root, names = folder_run
    assert rows_of(root / "out") == [n[:-4] for n in names]
    left = [f for _, _, fs in os.walk(root / "out") for f in fs if f.endswith(".tmp")]
    assert not left


def test_folder_resume_redoes_a_cube_with_a_missing_output(folder_run, tmp_path):
    root, names = folder_run
    out = tmp_path / "out"
    shutil.copytree(root / "out", out)
    (out / "flags" / names[0]).unlink()
    r = run("snap_folder.py", root / "in", out, "--workers", 1)
    assert r.returncode == 0 and "1 to process" in r.stdout and (out / "flags" / names[0]).exists()
    assert rows_of(out) == [n[:-4] for n in names]


def test_folder_row_survives_a_lost_main_process(folder_run, tmp_path):
    # a worker can finish its cube after the main process is gone: outputs and row file, no line in summary.csv
    root, names = folder_run
    out = tmp_path / "out"
    shutil.copytree(root / "out", out)
    (out / "summary.csv").unlink()
    r = run("snap_folder.py", root / "in", out, "--workers", 1)
    assert r.returncode == 0 and "0 to process" in r.stdout
    assert rows_of(out) == [n[:-4] for n in names]


def test_folder_failure_is_reported_in_the_exit_code(tmp_path):
    names = mini_set(tmp_path / "in", 3)
    (tmp_path / "in" / "images" / names[1]).write_bytes(b"not a tiff")
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "out", "--workers", 1)
    assert r.returncode != 0 and names[1] in r.stderr
    assert rows_of(tmp_path / "out") == [names[0][:-4], names[2][:-4]]


def test_folder_and_cube_handle_an_empty_label(tmp_path):
    names = mini_set(tmp_path / "in", 3, empty=(2,))
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "out", "--workers", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert np.array_equal(tifffile.imread(tmp_path / "out" / "labels" / names[1]),
                          tifffile.imread(tmp_path / "in" / "labels" / names[1]))
    pairs = [p for n in names for p in (tmp_path / "in" / "images" / n, tmp_path / "in" / "labels" / n)]
    r = run("snap_cube.py", tmp_path / "cube", *pairs)
    assert r.returncode == 0 and "unchanged" in r.stdout


def test_snap_cube_carries_on_after_a_failure(tmp_path):
    names = mini_set(tmp_path / "in", 2)
    (tmp_path / "in" / "images" / names[0]).write_bytes(b"not a tiff")
    pairs = [p for n in names for p in (tmp_path / "in" / "images" / n, tmp_path / "in" / "labels" / n)]
    r = run("snap_cube.py", tmp_path / "cube", *pairs)
    assert r.returncode != 0 and "FAILED" in r.stdout
    assert (tmp_path / "cube" / "labels" / names[1]).exists()


# --------------------------------------------------------------------------- recto side

def test_short_mesh_needs_more_than_its_own_curve(scene, tmp_path):
    root, _, _ = scene
    r = run("snap_tifxyz.py", tmp_path, "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif")
    assert r.returncode != 0 and "--normals" in r.stderr
    r = run("snap_tifxyz.py", tmp_path, "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif", "--normals", "ct")
    assert r.returncode == 0 and "from the CT" in r.stdout


def arc(radius, degrees, n=40):
    a = np.radians(np.linspace(0, degrees, n))
    z = np.arange(6) * 20.0
    A, Z = np.meshgrid(a, z)
    return np.stack([Z, 3000 + radius * np.sin(A), 3000 + radius * np.cos(A)], -1).astype(np.float32), np.ones(A.shape, bool)


def test_axis_fit_gate():
    P, v = arc(1200, 150)                 # a long stretch of a wrap
    assert axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    P, v = arc(150, 150)                  # the same turn, but only a sheet's local waviness in size
    assert not axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    P, v = arc(50000, 3)                  # nearly flat
    assert not axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    assert not axis_fit_is_meaningful(P[:0], v[:0], (0.0, 0.0))[0]


# --------------------------------------------------------------------------- --param

def test_parse_params_converts_each_field():
    got = parse_params(["t_max=6", "border=4", "keep_topology=on", "orient_cube=0", "convention_offset=-1.5", "smooth_warp=No"])
    assert got == dict(t_max=6.0, border=4, keep_topology=True, orient_cube=False, convention_offset=-1.5, smooth_warp=False)
    assert isinstance(got["t_max"], float) and isinstance(got["border"], int) and got["keep_topology"] is True
    assert parse_params([]) == {}


@pytest.mark.parametrize("bad,says", [("keep_topology", "NAME=VALUE"), ("border=0.5", "whole number"), ("t_max=abc", "a number"),
                                      ("keep_topology=enable", "true or false"), ("orient_cube=2", "true or false"),
                                      ("asdict=1", "unknown parameter"), ("nosuch=1", "unknown parameter")])
def test_parse_params_stops_with_a_message(bad, says):
    with pytest.raises(SystemExit, match=says):
        parse_params([bad])


def test_all_four_tools_stop_on_a_bad_param_before_doing_anything(scene, tmp_path):
    root, _, _ = scene
    names = mini_set(tmp_path / "in", 1)
    cube = (tmp_path / "in" / "images" / names[0], tmp_path / "in" / "labels" / names[0])
    calls = (("snap_cube.py", (tmp_path / "a", *cube)), ("snap_folder.py", (tmp_path / "in", tmp_path / "b")),
             ("snap_tifxyz.py", (tmp_path / "c", "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif", "--normals", "flipped")),
             # the address answers nothing, so a download attempted first would show up as a network error
             ("snap_segment.py", (tmp_path / "d", "--segment", "http://127.0.0.1:9/none.tifxyz/", "--volume", "none")))
    for tool, args in calls:
        for bad in ("keep_topology", "t_max=abc", "nosuch=1"):
            r = run(tool, *args, "--param", bad)
            assert r.returncode != 0 and "Traceback" not in r.stderr and bad.split("=")[0] in r.stderr, (tool, bad, r.stderr)
        assert not os.path.exists(tmp_path / "d" / "tiles")


@pytest.mark.parametrize("name", ["keep_topology", "orient_cube", "smooth_warp"])
def test_mesh_tools_refuse_what_only_labels_use(scene, tmp_path, name):
    root, _, _ = scene
    r = run("snap_tifxyz.py", tmp_path / "a", "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif", "--normals", "flipped",
            "--param", f"{name}=true")
    assert r.returncode != 0 and name in r.stderr and "labels" in r.stderr
    r = run_segment(root, tmp_path / "b", "--param", f"{name}=true")
    assert r.returncode != 0 and name in r.stderr and "labels" in r.stderr


def test_a_given_offset_is_not_silently_replaced(scene, tmp_path):
    root, _, _ = scene
    tif = ("snap_tifxyz.py", tmp_path / "a", "--tifxyz", root / "seg.tifxyz", "--volume", root / "vol.tif", "--normals", "flipped")
    r = run(*tif, "--target-offset=-2", "--param", "calibrate=true")
    assert r.returncode != 0 and "calibrate" in r.stderr
    r = run(*tif, "--target-offset=-2", "--param", "convention_offset=1")
    assert r.returncode != 0 and "convention_offset" in r.stderr
    r = run(*tif, "--param", "convention_offset=1", "--param", "calibrate=true")
    assert r.returncode != 0 and "calibrate" in r.stderr
    r = run_segment(root, tmp_path / "b", "--target-offset=-2", "--param", "calibrate=true")
    assert r.returncode != 0 and "calibrate" in r.stderr
    r = run_segment(root, tmp_path / "b", "--param", "convention_offset=1")
    assert r.returncode != 0 and "--target-offset" in r.stderr


def test_segment_honours_template_follows_offset(scene, tmp_path):
    root, _, _ = scene
    # --target-offset moves the learned profile too, unless told otherwise, which is how the ink test's second try ran
    r = run_segment(root, tmp_path / "off", "--target-offset", 0, "--param", "template_follows_offset=false")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not json.load(open(tmp_path / "off" / "snap_segment.json"))["params"]["template_follows_offset"]
    r = run_segment(root, tmp_path / "on", "--param", "template_follows_offset=true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / "on" / "snap_segment.json"))["params"]["template_follows_offset"]
    r = run_segment(root, tmp_path / "both", "--recentre-template", "--param", "template_follows_offset=false")
    assert r.returncode != 0 and "opposite" in r.stderr


# --------------------------------------------------------------------------- input checks

def test_snap_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        snap(CT[:32, :32, :32], np.zeros(CT.shape, np.uint8), Params())


def test_scale_keeps_both_components(tmp_path):
    assert densified_scale({"scale": [0.05, 0.05]}, 7) == [0.35, 0.35]
    assert densified_scale({"scale": [0.05, 0.1]}, 7) == [0.35, 0.7]
    assert densified_scale({}, 3) == [3.0, 3.0]
    P = mesh_grid()
    write_tifxyz(str(tmp_path / "s.tifxyz"), P, np.ones(P.shape[:2], bool), scale=[0.35, 0.7])
    assert read_tifxyz(str(tmp_path / "s.tifxyz"))[2]["scale"] == [0.35, 0.7]
    write_tifxyz(str(tmp_path / "t.tifxyz"), P, np.ones(P.shape[:2], bool), scale=0.35)
    assert read_tifxyz(str(tmp_path / "t.tifxyz"))[2]["scale"] == [0.35, 0.35]


def test_patch_wildcards_are_expanded(tmp_path):
    for k in (1, 0):
        (tmp_path / f"patch_{k}").mkdir()
    (tmp_path / "other").mkdir()
    got = expand_paths([str(tmp_path / "patch_*"), str(tmp_path / "other"), str(tmp_path / "missing_*")])
    assert got == [str(tmp_path / "patch_0"), str(tmp_path / "patch_1"), str(tmp_path / "other"), str(tmp_path / "missing_*")]


class _FlakyHandler(http.server.SimpleHTTPRequestHandler):
    """Serves a folder; the paths in `fail` answer 503 that many times first."""
    fail = {}

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            self.send_error(503)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


def test_fetch_retries_and_does_not_mistake_a_failure_for_no_mask(tmp_path, monkeypatch):
    import fetch_segment
    P = mesh_grid()
    src = tmp_path / "srv" / "s.tifxyz"
    write_tifxyz(str(src), P, np.ones(P.shape[:2], bool), {"scale": [0.05, 0.05]})
    tifffile.imwrite(src / "mask.tif", np.full(P.shape[:2], 255, np.uint8))
    monkeypatch.setattr(fetch_segment.time, "sleep", lambda s: None)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_FlakyHandler, directory=str(tmp_path / "srv")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/s.tifxyz/"
    try:
        _FlakyHandler.fail = {"mask.tif": 2}                   # a hiccup: recovered without a word
        assert "mask.tif" in fetch_segment.download_tifxyz(url, str(tmp_path / "a"))
        _FlakyHandler.fail = {"mask.tif": 99}                  # a real failure is an error, not "no mask"
        with pytest.raises(OSError):
            fetch_segment.download_tifxyz(url, str(tmp_path / "b"))
        _FlakyHandler.fail = {}
        (src / "mask.tif").unlink()                            # really no mask
        assert "mask.tif" not in fetch_segment.download_tifxyz(url, str(tmp_path / "c"))
    finally:
        srv.shutdown()
        srv.server_close()
