"""tools/eval_topology.py on small shapes with known answers."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
from eval_topology import betti, measure  # noqa: E402
from test_topology import betti as betti_other_way  # noqa: E402


def triple(b, conn):
    r = betti(b, conn)
    assert r["euler"] == r["b0"] - r["b1"] + r["b2"]
    return r["b0"], r["b1"], r["b2"]


def test_betti_on_known_shapes():
    z = np.zeros((16, 17, 19), bool)
    ball, torus, two = z.copy(), z.copy(), z.copy()
    ball[3:10, 3:10, 3:10] = True
    shell = ball.copy()
    shell[5:8, 5:8, 5:8] = False
    torus[3:10, 3:10, 5:7] = True
    torus[5:8, 5:8, 5:7] = False
    two[2:4, 2:12, 2:12] = True
    two[6:8, 2:12, 2:12] = True
    for conn in (6, 26):
        assert triple(z, conn) == (0, 0, 0)
        assert triple(ball, conn) == (1, 0, 0)
        assert triple(shell, conn) == (1, 0, 1)
        assert triple(torus, conn) == (1, 1, 0)
        assert triple(two, conn) == (2, 0, 0)
    # a shape on the cube's own face: the outside counts as background
    edge = z.copy()
    edge[0:3, 0:3, 0:3] = True
    assert triple(edge, 26) == (1, 0, 0) == triple(edge, 6)


def test_the_two_connectivities_differ_where_they_should():
    z = np.zeros((7, 7, 7), bool)
    corner = z.copy()
    corner[2, 2, 2] = corner[3, 3, 3] = True
    assert triple(corner, 26) == (1, 0, 0)
    assert triple(corner, 6) == (2, 0, 0)
    # four voxels around an empty one, touching along edges: a ring only if diagonal contact counts
    ring = z.copy()
    for y, x in ((2, 3), (3, 2), (3, 4), (4, 3)):
        ring[3, y, x] = True
    assert triple(ring, 26) == (1, 1, 0)
    assert triple(ring, 6) == (4, 0, 0)
    # a closed shell with one corner missing: its cavity leaks only through a 26-connected background
    hollow = z.copy()
    hollow[1:6, 1:6, 1:6] = True
    hollow[2:5, 2:5, 2:5] = False
    hollow[1, 1, 1] = False
    assert triple(hollow, 26) == (1, 0, 1)
    assert triple(hollow, 6) == (1, 0, 0)


@pytest.mark.parametrize("seed", range(4))
def test_betti_agrees_with_counting_cells_another_way(seed):
    b = np.random.default_rng(seed).random((12, 13, 14)) < 0.3 + 0.1 * seed
    for conn in (6, 26):
        assert triple(b, conn) == betti_other_way(b, conn)


def plates():
    """Two plates of 40 x 40 voxels, 3 thick, with 5 voxels between them; one corner not annotated."""
    lab = np.zeros((48, 30, 48), np.uint8)
    lab[4:44, 8:11, 4:44] = 1
    lab[4:44, 16:19, 4:44] = 1
    lab[:2, :, :2] = 2
    return lab


def run_measure(tmp_path, orig, snapped, **kw):
    tifffile.imwrite(tmp_path / "o.tif", orig)
    tifffile.imwrite(tmp_path / "s.tif", snapped)
    return measure(str(tmp_path / "o.tif"), str(tmp_path / "s.tif"), **kw)


def test_nothing_moved(tmp_path):
    r = run_measure(tmp_path, plates(), plates())
    assert r["betti"]["original"] == r["betti"]["snapped"]
    assert r["betti"]["snapped"]["26"] == dict(b0=2, b1=0, b2=0, euler=2)
    assert r["merger_events"] == 0 and r["contact_voxels"] == 0 and r["label2_same"]
    assert r["components"] == dict(original=2, snapped=2) and r["fragments"] == dict(original=0, snapped=0)


def test_rigid_roll_changes_nothing(tmp_path):
    lab = plates()
    r = run_measure(tmp_path, lab, np.roll(lab, 1, axis=0))
    assert r["betti"]["original"] == r["betti"]["snapped"]
    assert r["merger_events"] == 0 and r["fragments"]["snapped"] == 0 and not r["label2_same"]


def test_plates_moved_into_contact_are_one_merger_with_a_known_contact(tmp_path):
    lab = plates()
    moved = np.zeros_like(lab)
    moved[4:44, 10:13, 4:44] = 1                            # each plate 2 voxels toward the other, now touching
    moved[4:44, 13:16, 4:44] = 1
    moved[lab == 2] = 2
    r = run_measure(tmp_path, lab, moved)
    assert r["betti"]["snapped"]["26"] == dict(b0=1, b1=0, b2=0, euler=1)
    assert r["merger_events"] == 1
    m = r["merged"][0]
    assert m["sheets"] == [1, 2] and m["sizes"] == [4800, 4800] and m["size"] == 9600
    # the two middle layers meet along one plane of 40 x 40 voxels
    assert r["contact_voxels"] == 2 * 40 * 40
    assert r["contact_pairs"] == [dict(pair=[1, 2], contact_voxels=2 * 40 * 40)]


def test_a_bridge_is_a_merger_and_two_make_a_tunnel(tmp_path):
    lab = plates()
    one = lab.copy()
    one[20, 11:16, 20] = 1
    r = run_measure(tmp_path, lab, one)
    assert r["merger_events"] == 1
    assert r["betti"]["snapped"]["26"]["b0"] == 1 and r["betti"]["snapped"]["26"]["b1"] == 0
    two = one.copy()
    two[30, 11:16, 30] = 1
    r = run_measure(tmp_path, lab, two)
    assert r["merger_events"] == 1
    assert r["betti"]["snapped"]["26"]["b1"] == 1


def test_fragments_are_small_components_and_do_not_merge(tmp_path):
    lab = plates()
    specks = lab.copy()
    specks[10:12, 13, 10:12] = 1                            # 4 voxels between the plates, touching neither
    specks[30, 25, 30] = 1
    r = run_measure(tmp_path, lab, specks)
    assert r["fragments"] == dict(original=0, snapped=2)
    assert r["fragment_voxels"] == dict(original=0, snapped=5)
    assert r["components"] == dict(original=2, snapped=2)
    assert r["betti"]["snapped"]["26"]["b0"] == 4
    assert r["merger_events"] == 0
    assert run_measure(tmp_path, lab, specks, min_size=1)["fragments"]["snapped"] == 0


def test_small_sheets_take_no_part_in_merger_events(tmp_path):
    lab = np.zeros((32, 32, 32), np.uint8)
    lab[8:12, 8:11, 8:12] = 1
    lab[8:12, 14:17, 8:12] = 1
    joined = lab.copy()
    joined[8:12, 11:14, 8:12] = 1
    assert run_measure(tmp_path, lab, joined)["merger_events"] == 0
    assert run_measure(tmp_path, lab, joined, min_size=10)["merger_events"] == 1


def test_the_merger_rule_can_count_sheets_that_stay_separate(tmp_path):
    # a rule, not a tracking: the small sheet moves off, and the large one moves to where it was
    lab = np.zeros((48, 60, 48), np.uint8)
    lab[4:44, 20:23, 4:44] = 1
    lab[10:30, 28:30, 10:40] = 1
    moved = np.zeros_like(lab)
    moved[4:44, 26:29, 4:44] = 1
    moved[10:30, 33:35, 10:40] = 1
    r = run_measure(tmp_path, lab, moved)
    assert r["components"] == dict(original=2, snapped=2) and r["betti"]["snapped"]["26"]["b0"] == 2
    assert r["merger_events"] == 1


def test_a_band_moved_out_of_reach_is_not_a_merger(tmp_path):
    lab = plates()
    r = run_measure(tmp_path, lab, np.zeros_like(lab))
    assert r["voxels"]["snapped"] == 0 and r["merger_events"] == 0 and r["betti"]["snapped"]["6"]["b0"] == 0
    far = np.zeros((48, 60, 48), np.uint8)
    far[4:44, 8:11, 4:44] = 1
    moved = np.zeros_like(far)
    moved[4:44, 40:43, 4:44] = 1
    assert run_measure(tmp_path, far, moved)["merger_events"] == 0


def run_tool(*args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, "eval_topology.py"), *[str(a) for a in args]],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))


def test_tool_on_folders_and_on_a_pair(tmp_path):
    lab = plates()
    joined = lab.copy()
    joined[20, 11:16, 20] = 1
    for d, cube in (("o/labels", lab), ("s/labels", joined)):
        (tmp_path / d).mkdir(parents=True)
        tifffile.imwrite(tmp_path / d / "sample_00001.tif", cube)
        tifffile.imwrite(tmp_path / d / "sample_00002.tif", lab)
    tifffile.imwrite(tmp_path / "o" / "labels" / "unmatched.tif", lab)
    r = run_tool(tmp_path / "o", tmp_path / "s", tmp_path / "out", "--workers", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(tmp_path / "out" / "topology.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["cube"] for x in rows] == ["sample_00001", "sample_00002"]
    assert rows[0]["merger_events"] == "1" and rows[1]["merger_events"] == "0"
    assert rows[1]["b0_26_snapped"] == rows[1]["b0_6_snapped"] == "2"
    cubes = json.load(open(tmp_path / "out" / "topology.json"))
    assert cubes[0]["merged"][0]["sheets"] == [1, 2]
    # a rerun measures nothing again
    r = run_tool(tmp_path / "o", tmp_path / "s", tmp_path / "out", "--workers", 1)
    assert r.returncode == 0 and "0 to measure" in r.stdout

    r = run_tool(tmp_path / "o" / "labels" / "sample_00001.tif", tmp_path / "s" / "labels" / "sample_00001.tif",
                 tmp_path / "pair", "--workers", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(tmp_path / "pair" / "topology.json"))[0]["merger_events"] == 1


def test_tool_reports_a_failure(tmp_path):
    tifffile.imwrite(tmp_path / "o.tif", plates())
    tifffile.imwrite(tmp_path / "s.tif", plates()[:40])
    r = run_tool(tmp_path / "o.tif", tmp_path / "s.tif", tmp_path / "out", "--workers", 1)
    assert r.returncode != 0 and "shape" in r.stdout
    assert run_tool(tmp_path / "o.tif", tmp_path, tmp_path / "out", "--workers", 1).returncode != 0
