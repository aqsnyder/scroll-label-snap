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
from regression_cases import mesh_case, wavy_sheets  # noqa: E402
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


def run_segment(root, out, *args, segment="seg.tifxyz", volume="scan.zarr"):
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    import zarr
    scan = root / "scan.zarr"
    if not scan.exists():
        ct = tifffile.imread(root / "vol.tif")
        g = zarr.open_group(zarr.DirectoryStore(str(scan), dimension_separator="/"), mode="w")
        g.create_dataset("0", data=ct, chunks=(64, 64, 64), compressor=None)
    return run("snap_segment.py", out, "--segment", root / segment, "--volume", root / volume, "--tile", 8, "--overlap", 2,
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


def as_old_folder(out):
    """A finished folder as a run from before tiles/settings.json leaves it."""
    os.remove(out / "tiles" / "settings.json")
    s = json.load(open(out / "snap_segment.json"))
    del s["fine_factor"]
    for k in ("template_follows_offset", "keep_topology", "orient_cube"):
        del s["params"][k]
    json.dump(s, open(out / "snap_segment.json", "w"))


def test_old_folder_is_only_reused_by_a_run_that_fits_it(scene, tmp_path):
    root, _, _ = scene
    assert run_segment(root, tmp_path).returncode == 0
    as_old_folder(tmp_path)
    dense = (tmp_path / "s_snapped_dense.tifxyz" / "x.tif").read_bytes()
    for args, says in ((("--target-offset=-2",), "template_follows_offset"), (("--recentre-template",), "template_follows_offset"),
                       (("--param", "t_max=8"), "t_max"), (("--tile", 6), "--tile"), (("--normals", "grid"), "normal_sign")):
        r = run_segment(root, tmp_path, *args)
        assert r.returncode != 0 and "before this tool recorded" in r.stderr and says in r.stderr, (args, r.stderr)
        assert not (tmp_path / "tiles" / "settings.json").exists()
    # the same settings may reuse the tiles and add a fine grid, without being stamped as if the tool knew how they were made
    for args in ((), ("--fine-grid", 5)):
        r = run_segment(root, tmp_path, *args)
        assert r.returncode == 0 and "0 to go" in r.stdout, r.stdout + r.stderr
        assert not (tmp_path / "tiles" / "settings.json").exists()
        assert (tmp_path / "s_snapped_dense.tifxyz" / "x.tif").read_bytes() == dense
    assert (tmp_path / "s_snapped_fine.tifxyz").exists()


def test_old_folder_with_a_given_offset(scene, tmp_path):
    root, _, _ = scene
    # the old sign: an old run's --target-offset=-2 is recorded as convention_offset -2, which is T = +2 now
    same = ("--target-offset=2", "--param", "template_follows_offset=false")
    assert run_segment(root, tmp_path, *same).returncode == 0
    as_old_folder(tmp_path)
    assert run_segment(root, tmp_path, *same).returncode == 0
    r = run_segment(root, tmp_path, "--target-offset=-2", "--param", "template_follows_offset=false")
    assert r.returncode != 0 and "convention_offset" in r.stderr
    r = run_segment(root, tmp_path, "--target-offset=2")
    assert r.returncode != 0 and "template_follows_offset=false" in r.stderr
    r = run_segment(root, tmp_path)
    assert r.returncode != 0 and "--target-offset" in r.stderr


def test_interrupted_old_folder_has_no_record_of_its_settings(scene, tmp_path):
    root, _, _ = scene
    assert run_segment(root, tmp_path).returncode == 0
    as_old_folder(tmp_path)
    os.remove(tmp_path / "snap_segment.json")
    r = run_segment(root, tmp_path, "--param", "t_max=8")
    assert r.returncode != 0 and "nothing records" in r.stderr
    assert run_segment(root, tmp_path).returncode == 0
    os.remove(tmp_path / "snap_segment.json")
    (tmp_path / "tiles" / "r09999_c00000.npz").write_bytes(b"")         # not one of this segment's windows
    r = run_segment(root, tmp_path)
    assert r.returncode != 0 and "windows" in r.stderr
    os.remove(tmp_path / "tiles" / "r09999_c00000.npz")
    os.remove(tmp_path / "convention.json")                              # tiles and no offset: made with a given one
    r = run_segment(root, tmp_path)
    assert r.returncode != 0 and "--target-offset" in r.stderr


def test_tiles_are_not_reused_for_another_segment_or_scan(scene, tmp_path):
    root, _, P = scene
    other = P.copy()
    other[..., 1] -= SHEET + GAP                                         # the next sheet's face
    write_tifxyz(str(root / "next.tifxyz"), other, np.ones(other.shape[:2], bool), {"scale": [1 / 15, 1 / 15]}, uuid="next")
    assert run_segment(root, tmp_path).returncode == 0
    r = run_segment(root, tmp_path, segment="next.tifxyz")
    assert r.returncode != 0 and "other settings (segment)" in r.stderr
    shutil.copytree(root / "scan.zarr", tmp_path / "elsewhere.zarr")
    r = run_segment(root, tmp_path, volume=tmp_path / "elsewhere.zarr")
    assert r.returncode != 0 and "other settings (volume)" in r.stderr and "another address" in r.stderr
    for args, says in ((("--margin", 16), "margin"), (("--overlap", 1), "overlap"), (("--normals", "grid"), "sign"), (("--sample-tiles", 3), "sample_tiles")):
        r = run_segment(root, tmp_path, *args)
        assert r.returncode != 0 and says in r.stderr, (args, r.stderr)
    r = run_segment(root, tmp_path, "--tile", 6)
    assert r.returncode != 0 and "tile" in r.stderr


def test_a_run_that_made_nothing_does_not_hold_the_folder(scene, tmp_path):
    root, _, _ = scene
    r = run_segment(root, tmp_path, volume="no_such.zarr")
    assert r.returncode != 0 and "cannot read the scan" in r.stderr
    assert run_segment(root, tmp_path).returncode == 0
    (tmp_path / "convention.json").unlink()
    for f in os.listdir(tmp_path / "tiles"):
        if f.endswith(".npz"):
            os.remove(tmp_path / "tiles" / f)
    assert run_segment(root, tmp_path, "--param", "t_max=6").returncode == 0     # settings.json alone: nothing to protect


def test_offset_of_a_limited_run_does_not_carry_over_to_a_longer_one(scene, tmp_path):
    root, _, _ = scene
    assert run_segment(root, tmp_path, "--limit", 2).returncode == 0
    r = run_segment(root, tmp_path, "--limit", 1)
    assert r.returncode == 0, r.stdout + r.stderr                        # a shorter run is fine, it has seen those tiles
    r = run_segment(root, tmp_path)
    assert r.returncode != 0 and "--limit" in r.stderr
    assert run_segment(root, tmp_path / "whole").returncode == 0
    assert run_segment(root, tmp_path / "whole", "--limit", 2).returncode == 0


def test_every_tile_failing_is_an_error_and_writes_no_surface(scene, tmp_path):
    root, _, _ = scene
    pytest.importorskip("zarr")
    import zarr
    far = tmp_path / "far.zarr"
    g = zarr.open_group(zarr.DirectoryStore(str(far), dimension_separator="/"), mode="w")
    g.create_dataset("0", data=np.full((20, 20, 20), 60, np.uint8), chunks=(16, 16, 16), compressor=None)
    assert run_segment(root, tmp_path / "first").returncode == 0         # makes the zarr
    r = run_segment(root, tmp_path / "out", "--target-offset", 0, volume=far)
    assert r.returncode != 0 and "tiles failed" in r.stderr
    assert not (tmp_path / "out" / "snap_segment.json").exists() and not (tmp_path / "out" / "s_snapped.tifxyz").exists()
    r = run_segment(root, tmp_path / "out", "--target-offset", 0, volume=far)  # and a rerun does not pass it off as done
    assert r.returncode != 0 and not (tmp_path / "out" / "snap_segment.json").exists()
    r = run_segment(root, tmp_path / "sampled", volume=far)              # with no offset given, the sample tiles already say so
    assert r.returncode != 0 and "too few sample tiles" in r.stderr


def test_sample_tile_outside_the_scan_does_not_vote_on_the_offset():
    import snap_segment
    P = mesh_grid() + 1000
    snap_segment._STATE.update(seg=(P, np.ones(P.shape[:2], bool), {}), vol=CT)
    offset, error = snap_segment.tile_convention((0, 5, 0, 5, (0, 5, 0, 5)), Params(), 1.0, 8, 1, 1)
    assert offset is None and "inside the scan" in error


def test_grid_past_the_scan_does_not_pull_the_offset_to_zero(scene, tmp_path):
    root, _, _ = scene
    zs, xs = np.arange(30, 121, 15.0), np.arange(30, 271, 15.0)        # the scan ends at x = 190
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    Q = np.stack([Z, face_y(X, Z) + 3.0, X], -1).astype(np.float32)      # 3 voxels off the face, in the gap
    write_tifxyz(str(root / "past.tifxyz"), Q, np.ones(Q.shape[:2], bool), {"scale": [1 / 15, 1 / 15]}, uuid="past")
    r = run_segment(root, tmp_path / "out", "--sample-tiles", 12, segment="past.tifxyz")
    assert r.returncode == 0, r.stdout + r.stderr
    conv = json.load(open(tmp_path / "out" / "convention.json"))
    assert None in conv["offsets"] and 0.0 not in conv["offsets"]
    assert abs(conv["median"] + 3.0) < 1.0


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


def test_folder_refuses_to_overwrite_its_input(tmp_path):
    names = mini_set(tmp_path / "in", 1)
    before = (tmp_path / "in" / "labels" / names[0]).read_bytes()
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "in", "--workers", 1)
    assert r.returncode != 0 and "input" in r.stderr
    assert (tmp_path / "in" / "labels" / names[0]).read_bytes() == before and not (tmp_path / "in" / "confidence").exists()
    # labels with nothing beside them are not this tool's output: a copy of the input, say
    (tmp_path / "copy" / "labels").mkdir(parents=True)
    shutil.copy(tmp_path / "in" / "labels" / names[0], tmp_path / "copy" / "labels" / names[0])
    r = run("snap_folder.py", tmp_path / "in", tmp_path / "copy", "--workers", 1)
    assert r.returncode != 0 and "not from this tool" in r.stderr
    assert (tmp_path / "copy" / "labels" / names[0]).read_bytes() == before


def test_folder_files_appear_only_when_complete(tmp_path, monkeypatch):
    import snap_folder
    path = tmp_path / "a.tif"
    path.write_bytes(b"old")

    def refuse(*args):
        raise PermissionError("held open")
    monkeypatch.setattr(snap_folder.os, "replace", refuse)
    with pytest.raises(PermissionError):
        snap_folder.write_tif(str(path), np.zeros((4, 4, 4), np.uint8))
    assert path.read_bytes() == b"old" and os.listdir(tmp_path) == ["a.tif"]


def test_folder_writes_the_row_first_and_the_labels_last(tmp_path, monkeypatch):
    import snap_folder
    names = mini_set(tmp_path / "in", 1)
    out = tmp_path / "out"
    for sub in snap_folder.OUTPUTS:
        (out / sub).mkdir(parents=True)                  # no rows/: run_one makes it, so it can be called on its own
    row = out / "rows" / (names[0][:-4] + ".json")
    order = []
    real = snap_folder.write_tif

    def spy(path, data):
        order.append((os.path.basename(os.path.dirname(path)), row.exists()))
        real(path, data)
    monkeypatch.setattr(snap_folder, "write_tif", spy)
    r = snap_folder.run_one(str(tmp_path / "in" / "images" / names[0]), str(tmp_path / "in" / "labels" / names[0]), str(out))
    assert order == [("confidence", False), ("flags", False), ("labels", True)]
    assert json.load(open(row)) == r and (out / "labels" / names[0]).exists()


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace a file another program has open")
def test_summary_held_open_is_not_a_cube_failure(tmp_path):
    names = mini_set(tmp_path / "in", 2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.csv").write_text("cube,samples\n")
    with open(out / "summary.csv"):                      # as a spreadsheet holds it
        r = run("snap_folder.py", tmp_path / "in", out, "--workers", 1)
    assert r.returncode == 0 and "FAILED" not in r.stdout and r.stdout.count("warning: can't write summary.csv") == 1, r.stdout + r.stderr
    assert (out / "labels" / names[1]).exists() and not [f for f in os.listdir(out) if f.endswith(".tmp")]
    r = run("snap_folder.py", tmp_path / "in", out, "--workers", 1)
    assert r.returncode == 0 and "0 to process" in r.stdout
    assert rows_of(out) == [n[:-4] for n in names]


def test_a_failed_cube_is_still_reported_when_the_summary_is_held_open(tmp_path):
    names = mini_set(tmp_path / "in", 2)
    (tmp_path / "in" / "images" / names[1]).write_bytes(b"not a tiff")
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.csv").write_text("cube,samples\n")
    with open(out / "summary.csv"):
        r = run("snap_folder.py", tmp_path / "in", out, "--workers", 1)
    assert r.returncode != 0 and "1 of 2 cubes failed" in r.stderr


@pytest.mark.parametrize("text", ["bom", "extra column", "not a summary", "not text"])
def test_summary_edited_by_hand_does_not_stop_a_run(tmp_path, text):
    names = mini_set(tmp_path / "in", 2)
    out = tmp_path / "out"
    assert run("snap_folder.py", tmp_path / "in", out, "--workers", 1, "--limit", 1).returncode == 0
    path = out / "summary.csv"
    if text == "bom":                                    # as "CSV UTF-8" saves it
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    elif text == "extra column":
        path.write_text(path.read_text().replace("seconds", "seconds,note", 1).rstrip() + ",mine\n")
    elif text == "not a summary":
        path.write_text("garbage,x\n1,2\n")
    else:
        path.write_bytes(b"\xff\xfe\x00garbage")
    r = run("snap_folder.py", tmp_path / "in", out, "--workers", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("rebuilt" in r.stdout) == (text.startswith("not"))
    assert rows_of(out) == [n[:-4] for n in names]
    assert open(path).readline().strip() == ",".join(json.load(open(out / "rows" / (names[0][:-4] + ".json"))))


def test_a_mistyped_folder_leaves_no_settings_behind(tmp_path):
    mini_set(tmp_path / "in", 1)
    out = tmp_path / "out"
    r = run("snap_folder.py", tmp_path / "TYPO", out, "--workers", 1, "--param", "keep_topology=1")
    assert r.returncode != 0 and "images/" in r.stderr and "Traceback" not in r.stderr and not out.exists()
    # settings left by a try that made no cube don't hold the folder, either way round
    out.mkdir()
    (out / "params.json").write_text(json.dumps(dict(keep_topology=True)))
    r = run("snap_folder.py", tmp_path / "in", out, "--workers", 1, "--param", "orient_cube=1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(out / "params.json")) == dict(orient_cube=True)
    other = tmp_path / "other"
    other.mkdir()
    (other / "params.json").write_text(json.dumps(dict(keep_topology=True)))
    assert run("snap_folder.py", tmp_path / "in", other, "--workers", 1).returncode == 0
    assert not (other / "params.json").exists()


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


def test_snap_cube_carries_on_after_a_failed_write(tmp_path, monkeypatch):
    import snap_cube
    names = mini_set(tmp_path / "in", 2)
    pairs = [str(p) for n in names for p in (tmp_path / "in" / "images" / n, tmp_path / "in" / "labels" / n)]
    real = snap_cube.write_tif

    def held_open(path, data):
        if names[0] in path:
            raise PermissionError("held open")
        real(path, data)
    monkeypatch.setattr(snap_cube, "write_tif", held_open)
    with pytest.raises(SystemExit, match="1 cubes failed"):
        snap_cube.main(str(tmp_path / "out"), *pairs)
    assert (tmp_path / "out" / "labels" / names[1]).exists() and not (tmp_path / "out" / "labels" / names[0]).exists()


def test_snap_cube_takes_pairs_and_no_wildcards(tmp_path):
    names = mini_set(tmp_path / "in", 2)
    im, la = (tmp_path / "in" / "images" / names[0], tmp_path / "in" / "labels" / names[0])
    for paths, says in (((im,), "pairs"), ((im, la, tmp_path / "in" / "images" / names[1]), "pairs"),
                        ((tmp_path / "in" / "images" / "*.tif", tmp_path / "in" / "labels" / "*.tif"), "wildcards")):
        r = run("snap_cube.py", tmp_path / "out", *paths)
        assert r.returncode != 0 and says in r.stderr, r.stdout + r.stderr
        assert not (tmp_path / "out").exists()


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


def test_short_segment_stops_and_names_the_rule(tmp_path, capsys):
    import fetch_segment
    P, v = arc(900, 58)                   # 100% agreement and a radius near its extent: only the length fails
    write_tifxyz(str(tmp_path / "arc.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="arc")
    r = run("snap_segment.py", tmp_path / "a", "--segment", tmp_path / "arc.tifxyz", "--volume", "not read")
    assert r.returncode != 0 and "curvature" in r.stderr and "1,600 voxels across" in r.stderr     # run_0841.ps1 looks for "curvature"
    r = run("snap_tifxyz.py", tmp_path / "b", "--tifxyz", tmp_path / "arc.tifxyz", "--volume", "not read")
    assert r.returncode != 0 and "1,600 voxels across" in r.stderr
    orient = fetch_segment.recto_orientation(P, v)
    assert "recto_sign" not in orient and "1,600 voxels across" in capsys.readouterr().out
    P, v = arc(1200, 150)                 # long enough
    assert fetch_segment.recto_orientation(P, v)["recto_sign"] in (1, -1)


def test_patch_folder_warns_when_only_the_ct_vote_is_left(tmp_path):
    ct, P, v = mesh_case()
    patch = tmp_path / "segs" / "patch_0"
    os.makedirs(patch)
    write_tifxyz(str(patch / "segment.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="synth")
    tifffile.imwrite(patch / "volume.tif", ct)
    json.dump(dict(offset_zyx=[0, 0, 0], scale_to_level=1.0), open(patch / "patch.json", "w"))
    r = run("snap_tifxyz.py", tmp_path / "out", patch)
    assert r.returncode == 0 and "warning" in r.stdout and "CT vote" in r.stdout, r.stdout + r.stderr
    # the whole segment next to the patches is too short to settle it, and the tool says why
    write_tifxyz(str(tmp_path / "segs" / "segment.tifxyz"), P, v, {"scale": [0.05, 0.05]}, uuid="whole")
    r = run("snap_tifxyz.py", tmp_path / "out", patch)
    assert r.returncode == 0 and "1,600 voxels across" in r.stdout and "CT vote" in r.stdout, r.stdout + r.stderr


def test_axis_fit_gate():
    P, v = arc(1200, 150)                 # a long stretch of a wrap
    assert axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    P, v = arc(150, 150)                  # the same turn, but only a sheet's local waviness in size
    assert not axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    P, v = arc(50000, 3)                  # nearly flat
    assert not axis_fit_is_meaningful(P, v, (3000.0, 3000.0))[0]
    assert not axis_fit_is_meaningful(P[:0], v[:0], (0.0, 0.0))[0]
    # the extent rule itself, just under and just over 1,600 voxels across
    P, v = arc(850, 150)
    good, _, extent = axis_fit_is_meaningful(P, v, (3000.0, 3000.0))
    assert 1500 < extent < 1600 and not good
    P, v = arc(870, 150)
    good, _, extent = axis_fit_is_meaningful(P, v, (3000.0, 3000.0))
    assert 1600 < extent < 1700 and good


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
    assert r.returncode != 0 and f"only labels use {name}" in r.stderr
    r = run_segment(root, tmp_path / "b", "--param", f"{name}=true")
    assert r.returncode != 0 and f"only labels use {name}" in r.stderr


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
