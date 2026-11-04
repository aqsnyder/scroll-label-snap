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
import eval_orient_test  # noqa: E402
from eval_orient_test import (bootstrap, contested_samples, cube_mean, decide, draw_cubes, errors, fibre_signature, flip_share,  # noqa: E402
                              measure_cube, orientation_numbers, plan_lists, run_record, sample_sheets)
from eval_snap import trusted_samples  # noqa: E402
from regression_cases import wavy_sheets  # noqa: E402
from scroll_label_snap.core import Params, neighbour_graph, orient_normals, solve_surface, surface_points  # noqa: E402


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
    # the guard is core.orient_normals's 70%: a share of 60% is under it, and 80% is over it
    for share, under in ((0.6, True), (0.8, False)):
        q = n.copy()
        q[: int(len(q) * (1 - share))] = [0, 0, 1.0]
        r = orientation_numbers(pts, q, comp, ct)
        assert r["axis_share"] == pytest.approx(share, abs=0.01) and r["guard_triggered"] is under


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


def test_disputed_signature_reads_only_the_disputed_samples():
    img, pts, n = textured_plate()
    # turn the texture over on one half of the plate: read over the whole plate the signature nearly cancels
    img = img.copy()
    img[:48, 40:50] = img[:48, 49:39:-1]
    half = pts[:, 0] < 44
    r = fibre_signature(img, pts, n, half)
    assert r["disputed_n"] == half.sum() and r["n"] == len(pts)
    assert r["disputed_S"] < -0.3 and fibre_signature(img, pts, n, ~half)["disputed_S"] > 0.3 and abs(r["S"]) < 0.3


def test_errors_count_only_where_both_bands_exist():
    r = errors(np.array([0.5, 1.5, 2.5, np.nan], np.float32))
    assert r["n"] == 3 and r["within_1"] == pytest.approx(1 / 3) and r["over_2"] == pytest.approx(1 / 3)
    assert r["mean"] == pytest.approx(1.5) and r["missing"] == pytest.approx(0.25)
    assert errors(np.array([np.nan, np.nan])) == dict(within_1=None, over_2=None, mean=None, n=0, missing=1.0)
    assert errors(np.zeros(0))["missing"] is None


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
    """One run of the tool on the small pair, with what it passed to band_position and warp_labels."""
    root = tmp_path_factory.mktemp("orient")
    ct, lab = small_pair(root)
    seen = dict(lines=[], drift=[], keep=[], trusted=[])
    band_position, warp_labels, trusted = eval_orient_test.band_position, eval_orient_test.warp_labels, eval_orient_test.trusted_samples

    def lines(lab_in, pts, n):
        seen["lines"].append((pts.copy(), n.copy()))
        return band_position(lab_in, pts, n)

    def warps(lab_in, pts, n, shift, **kw):
        if "comp" not in kw:
            seen["drift"].append(n.copy())
        if kw.get("keep_topology"):
            seen["keep"].append((n.copy(), np.array(shift)))
        return warp_labels(lab_in, pts, n, shift, **kw)

    def trusts(lab_in, prep, p):
        seen["trusted"].append(prep[2].copy())
        return trusted(lab_in, prep, p)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(eval_orient_test, "band_position", lines)
        mp.setattr(eval_orient_test, "warp_labels", warps)
        mp.setattr(eval_orient_test, "trusted_samples", trusts)
        r = measure_cube(str(root / "images" / "sample_00001.tif"), str(root / "labels" / "sample_00001.tif"), 3)
    return root, ct, lab, r, seen


@pytest.fixture(scope="module")
def worked_out(small_cube):
    """What the tool should find on the small pair, worked out here from the same labels."""
    _, ct, lab, _, _ = small_cube
    p = Params()
    ct_s = ndi.gaussian_filter(ct.astype(np.float32), p.sigma_ct)
    pts, raw, _ = surface_points(lab, p)
    i, j = neighbour_graph(pts, raw, p)
    n_def, comp = orient_normals(pts, raw, i, j, ct_s)
    n_ori = orient_normals(pts, raw, i, j, ct_s, orient_cube=True)[0]
    inner = np.all((pts >= 10) & (pts <= np.array(lab.shape) - 11), axis=1)
    tr_def = trusted_samples(lab, (ct_s, pts, n_def, i, j, comp), p)
    tr_ori = trusted_samples(lab, (ct_s, pts, n_ori, i, j, comp), p)
    shift_ori = solve_surface(lab.shape, (ct_s, pts, n_ori, i, j, comp), p)[0]
    return dict(pts=pts, n_def=n_def, n_ori=n_ori, inner=inner, tr_def=tr_def, tr_ori=tr_ori, shift_ori=shift_ori)


def test_measures_on_a_small_pair(small_cube):
    _, ct, lab, r, _ = small_cube
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
    root, _, _, r, _ = small_cube
    again = measure_cube(str(root / "images" / "sample_00001.tif"), str(root / "labels" / "sample_00001.tif"), 3)
    assert again["seeds"] == r["seeds"]
    assert len({json.dumps(row, sort_keys=True) for row in r["seeds"].values()}) == 3      # each seed draws its own drift


def test_every_position_is_read_on_one_set_of_points_and_lines(small_cube, worked_out):
    _, _, _, r, seen = small_cube
    pi, ni = worked_out["pts"][worked_out["inner"]], worked_out["n_def"][worked_out["inner"]]
    assert r["n_points"] == len(pi) < len(worked_out["pts"])     # at least 10 voxels from every face
    assert len(seen["lines"]) == 1 + 3 + 3 * 3                   # the original, the clean arms, the drifted arms
    for pts, n in seen["lines"]:
        assert np.array_equal(pts, pi) and np.array_equal(n, ni)


def test_fixed_points_are_those_trusted_under_both_orientations(small_cube, worked_out):
    _, _, _, r, seen = small_cube
    both = (worked_out["tr_def"] & worked_out["tr_ori"])[worked_out["inner"]].sum()
    assert r["n_fixed"] == both
    assert len(seen["trusted"]) == 2 and np.array_equal(seen["trusted"][0], worked_out["n_def"])
    assert np.array_equal(seen["trusted"][1], worked_out["n_ori"])
    for row in r["seeds"].values():
        for arm in ("default", "orient_cube", "orient_cube_keep"):
            d = row[arm]["drift"]
            assert 0 < d["n"] <= both and d["n"] == round((1 - d["missing"]) * both)   # on those points, less any with no band
            c = row[arm]["consistency"]
            assert c["n"] == round((1 - c["missing"]) * r["n_points"])


def test_drifted_labels_are_built_along_the_default_normals(small_cube, worked_out):
    _, _, _, _, seen = small_cube
    assert len(seen["drift"]) == 3
    for n in seen["drift"]:
        assert np.array_equal(n, worked_out["n_def"])


def test_third_arm_walks_orient_cubes_solve(small_cube, worked_out):
    _, _, _, _, seen = small_cube
    n, shift = seen["keep"][0]                                   # the clean labels
    assert np.array_equal(n, worked_out["n_ori"]) and np.array_equal(shift, worked_out["shift_ori"])
    assert len(seen["keep"]) == 4                                # and the three drifts


def test_orientation_change_is_small_for_the_arm_that_keeps_its_orientation(small_cube):
    _, _, _, r, _ = small_cube
    for row in r["seeds"].values():
        assert row["orient_cube"]["orientation_change"] < 0.01    # its own clean normals against its own drifted ones


def test_disputed_samples_and_their_fibre_signature(small_cube, worked_out):
    _, ct, _, r, _ = small_cube
    disputed = np.einsum("ij,ij->i", worked_out["n_def"], worked_out["n_ori"]) < 0
    assert 0 < disputed.mean() < 1 and r["disputed"] == pytest.approx(disputed.mean())
    assert r["fibre"] == fibre_signature(ct, worked_out["pts"], worked_out["n_ori"], disputed)


def test_tool_runs_part_of_a_list_and_calls_it_incomplete(tmp_path):
    small_pair(tmp_path, "sample_00001")
    small_pair(tmp_path, "sample_00002")
    env = dict(os.environ, SNAP_THREADS="1", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "eval_orient_test.py"), str(tmp_path), str(tmp_path / "out"),
                        "--workers", "1", "--cubes", "sample_00001", "sample_00002", "--only", "sample_00002"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.load(open(tmp_path / "out" / "orient_test.json"))
    assert res["plan_list"] is False and res["planned"] == 2 and res["all"]["cubes"] == 1
    assert res["complete"] is None and res["adopt"] is None and res["third_arm"]["passed"] is None     # not the plan's list
    c = json.load(open(tmp_path / "out" / "cubes" / "sample_00002.json"))
    assert c["k"] == 1 and sorted(c["seeds"]) == ["1", "101", "201"]
    assert c["run"]["k"] == 1 and len(c["run"]["plan_sha256"]) == 64 and res["run"]["list_sha256"] == c["run"]["list_sha256"]
    with open(tmp_path / "out" / "orient_test_cubes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["cube"] for x in rows] == ["sample_00002"]
    assert 0 <= float(rows[0]["missing_consistency_default"]) <= 1 and 0 <= float(rows[0]["missing_drift_orient_cube"]) <= 1
    # the same command again measures nothing, and another position in another list is refused, whatever it would decide
    args = [sys.executable, os.path.join(TOOLS, "eval_orient_test.py"), str(tmp_path), str(tmp_path / "out"), "--workers", "1"]
    r = subprocess.run(args + ["--cubes", "sample_00001", "sample_00002", "--only", "sample_00002"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "0 to run" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(args + ["--cubes", "sample_00002"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "another k" in r.stderr
    if os.name == "nt":                                  # a csv held open loses nothing: the numbers are in the JSON files
        import msvcrt
        with open(tmp_path / "out" / "orient_test_cubes.csv", "r+") as h:
            msvcrt.locking(h.fileno(), msvcrt.LK_NBLCK, 1)
            r = subprocess.run(args + ["--cubes", "sample_00001", "sample_00002", "--only", "sample_00002"],
                               capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "can't write orient_test_cubes.csv" in r.stdout, r.stdout + r.stderr


def run_tool(*args):
    env = dict(os.environ, SNAP_THREADS="1", PYTHONUTF8="1")
    return subprocess.run([sys.executable, os.path.join(TOOLS, "eval_orient_test.py"), *[str(a) for a in args], "--workers", "1"],
                          capture_output=True, text=True, env=env)


@pytest.mark.parametrize("key", ["revision", "plan_sha256", "in_dir", "params", "k", "no record"])
def test_a_stored_cube_made_another_way_is_refused(tmp_path, key):
    (tmp_path / "out" / "cubes").mkdir(parents=True)
    run = dict(run_record(str(tmp_path), ["sample_00001"]), k=0)
    if key == "no record":
        run = None
    else:
        run[key] = 7
    json.dump(dict(cube="sample_00001", k=0, run=run), open(tmp_path / "out" / "cubes" / "sample_00001.json", "w"))
    r = run_tool(tmp_path, tmp_path / "out", "--cubes", "sample_00001")
    assert r.returncode != 0 and "use a new OUT_DIR" in r.stderr, r.stdout + r.stderr
    assert key == "no record" or key in r.stderr


@pytest.mark.parametrize("args,says", [(("--cubes", "sample_*"), "wildcards"), (("--cubes", "cube"), "number"),
                                       (("--cubes", "sample_00001", "--only", "sample_00001.tif"), "not in the list")])
def test_names_are_checked_before_any_work(tmp_path, args, says):
    r = run_tool(tmp_path, tmp_path / "out", *args)
    assert r.returncode != 0 and says in r.stderr and not (tmp_path / "out").exists(), r.stdout + r.stderr


def test_plans_own_list_is_the_one_that_can_decide(tmp_path):
    # no images, so the one cube chosen fails at once; what matters is how the decision reads
    r = run_tool(tmp_path, tmp_path / "out", "--only", plan_lists()["drawn"][0])
    assert r.returncode != 0
    res = json.load(open(tmp_path / "out" / "orient_test.json"))
    assert res["plan_list"] is True and res["complete"] is False and res["adopt"] is False and res["planned"] == 100
    assert res["third_arm"]["passed"] is False


def test_a_cube_with_no_point_far_from_its_faces_is_not_a_failure(tmp_path):
    ct, _, (z, y, x) = wavy_sheets((24, 96, 72), n_sheets=3, base=18.0)
    lab = np.zeros(ct.shape, np.uint8)
    lab[:6][(y[:6] >= 30) & (y[:6] < 32)] = 1            # all within 10 voxels of the top face
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    tifffile.imwrite(tmp_path / "images" / "sample_00001.tif", ct)
    tifffile.imwrite(tmp_path / "labels" / "sample_00001.tif", lab)
    r = run_tool(tmp_path, tmp_path / "out", "--cubes", "sample_00001")
    assert r.returncode == 0 and "n/a" in r.stdout, r.stdout + r.stderr


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


def test_bootstrap_is_the_plans():
    d = np.array([1.0, 2.0, 4.0, 8.0, -3.0, 0.5])
    b = np.random.default_rng(0).choice(d, (5000, len(d))).mean(1)       # 5,000 resamples of cubes, seed 0
    got = bootstrap(d)
    assert got["low"] == pytest.approx(np.percentile(b, 2.5)) and got["high"] == pytest.approx(np.percentile(b, 97.5))
    assert got["mean"] == pytest.approx(d.mean()) and got["cubes"] == 6


def test_a_cube_counts_by_its_mean_over_the_seeds():
    c = fake_cube(1)
    for s, v in zip(sorted(c["seeds"]), (0.8, 0.9, 1.0)):
        c["seeds"][s]["orient_cube"]["consistency"]["within_1"] = v
    assert cube_mean(c, "orient_cube", "consistency") == pytest.approx(90.0)
    c["seeds"]["1"]["orient_cube"]["consistency"]["within_1"] = None
    assert cube_mean(c, "orient_cube", "consistency") is None


def test_rule_on_cubes_and_seeds_that_differ():
    # by hand, in points: three cubes at +10, +4 and +1, so a mean of +5; one cube's orientation changes in 3% of points on
    # one seed and the rest in none, so a mean of 1/3 of 1% across its seeds and of 1/9 across cubes, and a maximum of 1%
    cubes = [fake_cube(k) for k in (1, 2, 3)]
    for c, diff in zip(cubes, (10.0, 4.0, 1.0)):
        for s, extra in zip(sorted(c["seeds"]), (-2.0, 0.0, 2.0)):
            c["seeds"][s]["default"]["consistency"]["within_1"] = 0.8
            c["seeds"][s]["orient_cube"]["consistency"]["within_1"] = 0.8 + (diff + extra) / 100
            c["seeds"][s]["orient_cube"]["orientation_change"] = 0.0
    first = sorted(cubes[0]["seeds"])[0]
    cubes[0]["seeds"][first]["orient_cube"]["orientation_change"] = 0.03
    d = decide(cubes, 3)
    diff = d["all"]["consistency_diff"]
    assert diff["mean"] == pytest.approx(5.0) and diff["cubes"] == 3 and diff["better"] == 3 and diff["worse"] == 0
    assert d["all"]["orientation_change"]["orient_cube"] == pytest.approx(1 / 3)
    assert d["all"]["orientation_change"]["orient_cube_max"] == pytest.approx(1.0)
    # the guard is on the mean: one cube over 1% does not trip it
    cubes[0]["seeds"][first]["orient_cube"]["orientation_change"] = 0.06
    d = decide(cubes, 3)
    assert d["all"]["orientation_change"]["orient_cube_max"] == pytest.approx(2.0) and d["checks"]["orientation_change"]


def test_only_the_plans_list_decides():
    d = decide(fake_set(), 30, plan_list=False)
    assert d["adopt"] is None and d["complete"] is None and d["third_arm"]["passed"] is None and not d["plan_list"]
    assert all(d["checks"].values())                     # the numbers are still there to read
    assert decide(fake_set(), 30)["plan_list"]


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


def test_revision_ignores_tags(tmp_path, monkeypatch):
    """A tag added later must not change the name of a commit, or a run would refuse its own folder."""
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "f").write_text("a")
    git("add", "f")
    git("commit", "-qm", "one")
    monkeypatch.setattr(eval_orient_test, "HERE", str(tmp_path))
    head = git("rev-parse", "HEAD")
    assert eval_orient_test.code_revision() == head
    git("tag", "-a", "v1", "-m", "tag")
    assert eval_orient_test.code_revision() == head
    (tmp_path / "f").write_text("b")
    assert eval_orient_test.code_revision() == head + "-dirty"
