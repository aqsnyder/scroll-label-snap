"""tools/eval_orient_test.py: the cube list in the plan, its measures on small synthetic cubes, and the plan's decision rule."""
import copy
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
RESULTS = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
from eval_orient_test import (contested_samples, cube_mean, decide, draw_cubes, fibre_signature, flip_share,  # noqa: E402
                              measure_cube, orientation_numbers, plan_lists, sample_sheets)
from regression_cases import wavy_sheets  # noqa: E402
from scroll_label_snap.core import Params, neighbour_graph, orient_normals, surface_points  # noqa: E402


def test_plan_lists_reproduce_the_drawn_cubes():
    lists = plan_lists()
    with open(os.path.join(RESULTS, "full_run_summary.csv"), newline="") as f:
        names = [r["cube"] for r in csv.DictReader(f)]
    assert len(names) == 892
    dev = [f"sample_{k:05d}" for k in range(1, 21)]
    held = json.load(open(os.path.join(RESULTS, "heldout_selection.json")))["names"]
    assert len(lists["convention audit"]) == 180 and len(lists["prevalence check"]) == 40 and len(held) == 100
    excluded = set(dev) | set(held) | set(lists["convention audit"]) | set(lists["prevalence check"])
    drawn = draw_cubes(names, excluded)
    assert drawn == lists["drawn"] and len(set(drawn)) == 100
    assert not set(drawn) & excluded


def flat_samples(y, lo, hi, step=3.0):
    a, b = np.meshgrid(np.arange(lo, hi, step), np.arange(lo, hi, step), indexing="ij")
    return np.stack([a.ravel(), np.full(a.size, float(y)), b.ravel()], 1).astype(np.float32)


def test_orientation_numbers_on_known_cases():
    # two flat sheets, with papyrus on the -y side of both: one axis, every vote agrees
    ct = np.full((64, 64, 64), 60, np.float32)
    ct[:, 14:20] = ct[:, 34:40] = 170.0
    ct = ndi.gaussian_filter(ct, 1.0)
    pts = np.vstack([flat_samples(20, 12, 52), flat_samples(40, 12, 52)])
    n = np.tile(np.array([0, 1.0, 0], np.float32), (len(pts), 1))
    comp = np.repeat([0, 1], len(pts) // 2)
    r = orientation_numbers(pts, n, comp, ct)
    assert r["axis_share"] == pytest.approx(1.0) and not r["guard_triggered"]
    assert r["sign_margin"] == pytest.approx(1.0)
    assert r["across_axis"] == 0 and r["against_vote"] == 0
    # the second sheet turned against its vote: one sign now disagrees, in equal weight, whichever frame the normals are in
    m = n.copy()
    m[comp == 1] *= -1
    r = orientation_numbers(pts, m, comp, ct)
    assert r["against_vote"] == pytest.approx(0.5)
    assert r["sign_margin"] == pytest.approx(1.0)        # the votes are read back, so the two sheets still agree on the axis
    # normals with no single axis: half along y, half along x
    q = n.copy()
    q[comp == 1] = [0, 0, 1.0]
    r = orientation_numbers(pts, q, comp, ct)
    assert r["axis_share"] == pytest.approx(0.5, abs=1e-6) and r["guard_triggered"]


def test_flip_share_counts_reversed_normals_nearby():
    pts = flat_samples(20, 0, 30)
    n = np.tile(np.array([0, 1.0, 0], np.float32), (len(pts), 1))
    m = n.copy()
    m[: len(pts) // 2] *= -1
    assert flip_share(pts, n, pts, n) == 0
    assert flip_share(pts, n, pts, m) == pytest.approx(0.5)
    assert flip_share(pts, n, pts + [0, 20, 0], m) == 0     # nothing within 6 voxels


def test_contested_samples_are_two_sheets_going_to_one_place():
    a, b = flat_samples(20, 0, 30), flat_samples(26, 0, 30)
    pts = np.vstack([a, b])
    n = np.tile(np.array([0, 1.0, 0], np.float32), (len(pts), 1))
    sheet = np.repeat([1, 2], len(a))
    apart = np.concatenate([np.full(len(a), -1.0), np.full(len(b), 1.0)])
    together = np.concatenate([np.full(len(a), 2.0), np.full(len(b), -2.0)])
    assert not contested_samples(pts, n, apart, sheet).any()
    assert contested_samples(pts, n, together, sheet).all()
    assert not contested_samples(pts, n, together, np.ones(len(pts), int)).any()     # one sheet is not a contest
    lab = np.zeros((10, 10, 10), np.int32)
    lab[5, 5, 5] = 7
    assert sample_sheets(np.array([[5.2, 4.9, 5.0], [1.0, 1.0, 1.0]], np.float32), lab).tolist() == [7, 7]


def textured_plate():
    """A plate 10 voxels thick, y 40-49. Close to its +y face the texture varies along z (the scroll axis), close to its
    -y face around the scroll, x. Samples on its +y face, normals to +y."""
    z, y, x = np.mgrid[:96, :96, :96].astype(np.float32)
    ct = np.full(z.shape, 40.0, np.float32)
    plate = (y >= 40) & (y < 50)
    ct[plate] = 170.0
    ct += 40 * ((y >= 45) & (y < 50)) * np.sin(2 * np.pi * z / 7) + 40 * ((y >= 40) & (y < 45)) * np.sin(2 * np.pi * x / 7)
    pts = flat_samples(50, 18, 78)
    n = np.tile(np.array([0, 1.0, 0], np.float32), (len(pts), 1))
    return np.clip(ct, 0, 255).astype(np.uint8), pts, n


def test_fibre_signature_follows_the_side_of_the_texture():
    img, pts, n = textured_plate()
    disputed = np.arange(len(pts)) % 2 == 0
    r = fibre_signature(img, pts, n, disputed)
    assert r["n"] == len(pts) and r["disputed_n"] == disputed.sum()
    assert r["S"] > 0.5 and r["disputed_S"] > 0.5
    assert abs(r["placebo"]) < 0.25
    # reversed normals read the same plate from the other side, so they disagree with the texture
    assert fibre_signature(img, pts, -n, disputed)["S"] == pytest.approx(-r["S"], abs=0.05)
    assert fibre_signature(img, pts, n, np.zeros(len(pts), bool))["disputed_S"] is None


def small_pair(root, name="sample_00001"):
    """Wavy sheets labelled on their +y faces and a thin flake in one gap labelled on its far face, whose
    vote points the other way."""
    shape = (72, 96, 72)
    ct, faces, (z, y, x) = wavy_sheets(shape, n_sheets=3, base=18.0)
    lab = np.zeros(shape, np.uint8)
    for f in faces:
        lab[(y >= f - 1.0) & (y < f + 1.0)] = 1
    patch = (z >= 24) & (z < 48) & (x >= 24) & (x < 48)
    ct[patch & (y >= faces[0] + 5) & (y < faces[0] + 8)] = 170
    lab[patch & (y >= faces[0] + 4) & (y < faces[0] + 6)] = 1
    (root / "images").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    tifffile.imwrite(root / "images" / f"{name}.tif", ct)
    tifffile.imwrite(root / "labels" / f"{name}.tif", lab)
    return ct, lab


@pytest.fixture(scope="module")
def small_cube(tmp_path_factory):
    root = tmp_path_factory.mktemp("orient")
    ct, lab = small_pair(root)
    return root, ct, lab, measure_cube(str(root / "images" / "sample_00001.tif"), str(root / "labels" / "sample_00001.tif"), 3)


def test_measures_on_a_small_pair(small_cube):
    _, ct, lab, r = small_cube
    assert r["cube"] == "sample_00001" and r["id"] == 1 and r["k"] == 3
    assert sorted(r["seeds"]) == ["103", "203", "3"] and sorted(r["inputs"]) == ["103", "203", "3", "clean"]
    assert r["n_fixed"] > 0 and r["n_points"] > r["n_fixed"]
    for row in r["seeds"].values():
        for arm in ("default", "orient_cube", "orient_cube_keep"):
            assert row[arm]["consistency"]["n"] > 0 and row[arm]["drift"]["n"] > 0
            assert 0 <= row[arm]["consistency"]["within_1"] <= 1 and 0 <= row[arm]["orientation_change"] <= 1
    assert r["axis_share"] > 0.7 and r["guard_triggers"] == 0 and r["sign_margin"] > 0.9
    # the two orientations disagree on the flake's cube, orient_cube has the sheets all on one side, and it is steadier
    assert r["disputed"] > 0.1
    assert cube_mean(r, "orient_cube", "consistency") > cube_mean(r, "default", "consistency")
    p = Params()
    pts, raw, _ = surface_points(lab, p)
    i, j = neighbour_graph(pts, raw, p)
    ct_s = ndi.gaussian_filter(ct.astype(np.float32), p.sigma_ct)
    n_ori = orient_normals(pts, raw, i, j, ct_s, orient_cube=True)[0]
    assert np.mean(n_ori[:, 1] > 0) > 0.99
    # the third arm keeps the input's topology, which the plain warp does not promise
    assert r["arms"]["orient_cube_keep"]["betti_equal"]
    json.dumps(r)                                        # all plain types


def test_seeds_set_the_drift(tmp_path, small_cube):
    root, _, _, r = small_cube
    again = measure_cube(str(root / "images" / "sample_00001.tif"), str(root / "labels" / "sample_00001.tif"), 3)
    assert again["seeds"] == r["seeds"]


def test_tool_runs_part_of_a_list_and_calls_it_incomplete(tmp_path):
    small_pair(tmp_path, "sample_00001")
    small_pair(tmp_path, "sample_00002")
    env = dict(os.environ, SNAP_THREADS="1", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "eval_orient_test.py"), str(tmp_path), str(tmp_path / "out"),
                        "--workers", "1", "--cubes", "sample_00001", "sample_00002", "--only", "sample_00002"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.load(open(tmp_path / "out" / "orient_test.json"))
    assert res["complete"] is False and res["adopt"] is False and res["planned"] == 2 and res["all"]["cubes"] == 1
    c = json.load(open(tmp_path / "out" / "cubes" / "sample_00002.json"))
    assert c["k"] == 1 and sorted(c["seeds"]) == ["1", "101", "201"]
    with open(tmp_path / "out" / "orient_test_cubes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["cube"] for x in rows] == ["sample_00002"]


# --------------------------------------------------------------------------- the decision rule

def fake_cube(k, cons=(80.0, 85.0), drift=(90.0, 90.5), change=0.001, disputed_n=0, disputed_s=None, keep_equal=True, events=0):
    def arm(c, d, chg):
        return dict(consistency=dict(within_1=c / 100), drift=dict(within_1=d / 100), orientation_change=chg)
    seeds = {str(off + k): dict(default=arm(cons[0], drift[0], 0.05), orient_cube=arm(cons[1], drift[1], change),
                                orient_cube_keep=arm(cons[1] - 0.1, drift[1] - 0.1, change)) for off in (0, 100, 200)}
    keep = dict(betti_equal=keep_equal, merger_events=events, contested=0.002)
    return dict(cube=f"sample_{k:05d}", id=k, seeds=seeds, guard_triggers=0, sign_margin_min=0.95, axis_share_min=0.85,
                fibre=dict(disputed_n=disputed_n, disputed_S=disputed_s), arms=dict(orient_cube_keep=keep))


def fake_set():
    rng = np.random.default_rng(0)
    cubes = [fake_cube(k + 200 if k < 10 else k + 300, cons=(80.0, 85.0 + rng.normal(0, 1)), drift=(90.0, 90.5 + rng.normal(0, 1)))
             for k in range(30)]
    for c in cubes[:4]:
        c["fibre"] = dict(disputed_n=1500, disputed_S=0.3)
    return cubes


def test_rule_adopts_only_when_everything_holds():
    d = decide(fake_set(), 30)
    assert d["complete"] and d["adopt"] and all(d["checks"].values())
    assert d["third_arm"]["passed"]
    assert d["stretch_161_280"]["cubes"] == 10 and d["other_ids"]["cubes"] == 20
    assert d["all"]["consistency_diff"]["low"] > 0
    assert not decide(fake_set()[:29], 30)["adopt"]                       # a cube is missing


def test_primary_needs_the_interval_above_zero():
    cubes = fake_set()
    rng = np.random.default_rng(1)
    for c in cubes:
        v = 80.0 + rng.normal(0, 3)
        for s in c["seeds"]:
            c["seeds"][s]["orient_cube"]["consistency"]["within_1"] = v / 100
    d = decide(cubes, 30)
    assert d["all"]["consistency_diff"]["low"] <= 0 and not d["checks"]["primary"] and not d["adopt"]


def test_drift_guard_allows_one_point():
    cubes = fake_set()
    for c in cubes:
        for s in c["seeds"]:
            c["seeds"][s]["orient_cube"]["drift"]["within_1"] = 0.895
    d = decide(cubes, 30)
    assert d["all"]["drift_diff"]["mean"] == pytest.approx(-0.5) and d["checks"]["drift"]
    for c in cubes:
        for s in c["seeds"]:
            c["seeds"][s]["orient_cube"]["drift"]["within_1"] = 0.885
    d = decide(cubes, 30)
    assert not d["checks"]["drift"] and not d["adopt"]


def test_orientation_change_guard_is_a_mean_of_at_most_one_percent():
    cubes = fake_set()
    for c in cubes:
        for s in c["seeds"]:
            c["seeds"][s]["orient_cube"]["orientation_change"] = 0.012
    d = decide(cubes, 30)
    assert d["all"]["orientation_change"]["orient_cube"] == pytest.approx(1.2) and not d["checks"]["orientation_change"]


def test_fibre_guard_counts_cubes_with_enough_data():
    cubes = fake_set()
    assert decide(cubes, 30)["all"]["fibre"] == dict(cubes_with_data=4, positive=4, share=1.0)
    cubes[0]["fibre"] = dict(disputed_n=1500, disputed_S=-0.1)
    assert decide(cubes, 30)["checks"]["fibre"]                           # 3 of 4 is 75%
    cubes[1]["fibre"] = dict(disputed_n=1500, disputed_S=-0.1)
    assert not decide(cubes, 30)["checks"]["fibre"]
    # cubes under 1,000 disputed samples don't count, and with none that count the guard is not met
    for c in cubes[:4]:
        c["fibre"] = dict(disputed_n=999, disputed_S=-0.1)
    d = decide(cubes, 30)
    assert d["all"]["fibre"]["cubes_with_data"] == 0 and not d["checks"]["fibre"] and not d["adopt"]


def test_third_arm_must_hold_on_every_cube_and_does_not_decide_adoption():
    cubes = fake_set()
    assert decide(cubes, 30)["third_arm"]["passed"]
    bad = copy.deepcopy(cubes)
    bad[3]["arms"]["orient_cube_keep"]["merger_events"] = 1
    d = decide(bad, 30)
    assert not d["third_arm"]["passed"] and d["third_arm"]["merger_events_zero"] == 29 and d["adopt"]
    bad = copy.deepcopy(cubes)
    bad[3]["arms"]["orient_cube_keep"]["betti_equal"] = False
    assert not decide(bad, 30)["third_arm"]["passed"]
    assert cube_mean(cubes[0], "orient_cube_keep", "consistency") < cube_mean(cubes[0], "orient_cube", "consistency")
