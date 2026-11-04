"""tools/eval_snap.py: the rows with no graph cut, on cubes where the answer is known, and cubes with nothing to move."""
import json
import os
import sys
import warnings

import numpy as np
import pytest
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
import eval_snap  # noqa: E402
from eval_snap import median_on_sheet, per_sample_shifts  # noqa: E402
from regression_cases import wavy_sheets  # noqa: E402
from scroll_label_snap.core import Params, prepare  # noqa: E402
from test_robustness import mini_set, run  # noqa: E402


def part_displaced(extra=2.5):
    """Sheets labelled on their +y faces, with the labels at x >= 56 moved `extra` voxels into the gap: most of the labels
    are right, so the labels' own offset stays near 0, and only that part has somewhere to go."""
    shape = (64, 96, 88)
    ct, faces, (z, y, x) = wavy_sheets(shape, n_sheets=3, base=14.0)
    lab = np.zeros(shape, np.uint8)
    for f in faces:
        c = f + extra * (x >= 56)
        lab[(y >= c - 1.0) & (y < c + 1.0)] = 1
    return ct, lab


def test_rows_with_no_graph_cut_find_the_known_face():
    ct, lab = part_displaced()
    p = Params()
    prep = prepare(ct, lab, p)
    pts, n, comp = prep[1], prep[2], prep[5]
    out = per_sample_shifts(lab.shape, prep, p)
    assert sorted(out) == ["argmax", "gradient", "gradient median"]
    inside = np.all((pts >= p.border) & (pts <= np.array(lab.shape) - 1 - p.border), axis=1)
    moved = pts[:, 2] >= 62                                     # away from the join
    still = pts[:, 2] <= 50
    for name, s in out.items():
        along_y = s * n[:, 1]                                   # 2.5 voxels along y, on sheets that lean a little
        assert -3.0 <= np.median(along_y[moved & inside]) <= -1.25, name
        assert abs(np.median(along_y[still & inside])) <= 0.5, name
    # the border stays put, as for the snapper
    assert not np.any(out["argmax"][np.any(pts < p.border, axis=1)])
    assert out["argmax"].dtype == np.float32 and len(np.unique(comp)) > 1


def test_median_stays_on_its_sheet():
    # one sheet's samples are 3 voxels apart, and a denser sheet lies 1.5 voxels beside it, so most near samples are the other's
    a, b = np.meshgrid(np.arange(0, 30, 3.0), np.arange(0, 30, 3.0), indexing="ij")
    c, d = np.meshgrid(np.arange(0, 30, 1.0), np.arange(0, 30, 1.0), indexing="ij")
    pts = np.vstack([np.stack([a.ravel(), 0 * a.ravel(), b.ravel()], 1), np.stack([c.ravel(), 0 * c.ravel() + 1.5, d.ravel()], 1)]).astype(np.float32)
    comp = np.r_[np.zeros(a.size, int), np.ones(c.size, int)]
    s = np.where(comp == 0, 1.0, -1.0).astype(np.float32)
    s[0] = 9.0                                                  # one wild sample does not survive the median
    got = median_on_sheet(s, pts, comp)
    assert (got[comp == 0] == 1).all() and (got[comp == 1] == -1).all()


def test_median_with_few_samples():
    assert median_on_sheet(np.array([2.0], np.float32), np.zeros((1, 3), np.float32), np.zeros(1, int)).tolist() == [2.0]
    assert median_on_sheet(np.zeros(0, np.float32), np.zeros((0, 3), np.float32), np.zeros(0, int)).shape == (0,)


def test_rows_on_a_flat_scan_have_no_edge_to_read():
    ct, lab = part_displaced()
    p = Params()
    prep = prepare(np.full(ct.shape, 90, np.uint8), lab, p)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # np.median of nothing would warn, and give NaN
        out = per_sample_shifts(lab.shape, prep, p)
    assert all(np.isfinite(s).all() for s in out.values())


@pytest.mark.parametrize("case", ["empty", "near the border"])
def test_eval_snap_runs_on_a_cube_with_nothing_to_move(tmp_path, case):
    ct, _, _ = wavy_sheets((64, 64, 64), n_sheets=2, base=10.0)
    lab = np.zeros(ct.shape, np.uint8)
    if case != "empty":
        lab[:6, 20:40, 20:40] = 1
    tifffile.imwrite(tmp_path / "img.tif", ct)
    tifffile.imwrite(tmp_path / "lab.tif", lab)
    r = run("eval_snap.py", tmp_path / "out.json", tmp_path / "img.tif", tmp_path / "lab.tif")
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.load(open(tmp_path / "out.json"))["cubes"][0]
    assert got["n_points"] == 0 or got["n_trusted"] == 0
    assert got["drift_test"]["argmax"]["n"] == 0


def test_each_cube_records_its_seed_and_the_rows_use_the_snappers_warp(tmp_path, monkeypatch):
    names = mini_set(tmp_path / "in", 2)
    pairs = [str(q) for n in names for q in (tmp_path / "in" / "images" / n, tmp_path / "in" / "labels" / n)]
    eval_snap.main(str(tmp_path / "out.json"), *pairs)
    got = json.load(open(tmp_path / "out.json"))["cubes"]
    assert [c["seed"] for c in got] == [0, 1]
    calls = []
    real = eval_snap.warp_labels

    def spy(*args, **kwargs):
        calls.append(kwargs.get("keep_topology", False))
        return real(*args, **kwargs)
    monkeypatch.setattr(eval_snap, "warp_labels", spy)
    eval_snap.evaluate(pairs[0], pairs[1], Params(keep_topology=True), 0)
    assert sum(calls) == 6                                      # three methods with no graph cut, on clean and drifted labels


def test_eval_snap_takes_pairs_and_no_wildcards(tmp_path):
    for paths, says in ((("a.tif",), "pairs"), (("images/*.tif", "labels/*.tif"), "wildcards")):
        r = run("eval_snap.py", tmp_path / "out.json", *paths)
        assert r.returncode != 0 and says in r.stderr, r.stdout + r.stderr
