"""tools/surface_moves.py on synthetic meshes: net shift, roughness, codes and held-back blocks."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, os.path.join(HERE, ".."))
from scroll_label_snap.tifxyz import write_tifxyz  # noqa: E402


def plane(rows=40, cols=50):
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return np.stack([np.full_like(r, 100.0), 40 + 20 * r, 40 + 20 * c], -1)


def run(*args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, "surface_moves.py")] + [str(a) for a in args],
                          check=True, capture_output=True, text=True)


def test_moves_shift_roughness_codes_and_blocks(tmp_path):
    rng = np.random.default_rng(4)
    P = plane()
    valid = np.ones(P.shape[:2], bool)
    rough = rng.normal(0, 2.0, P.shape[:2])
    S = P.copy()
    S[..., 0] += -3.0 + rough                       # a net shift plus vertex-scale roughness, along the normal (z)
    held = np.zeros_like(valid)
    held[5:10, 5:12] = True                         # one held-back block
    held[30, 40] = True                             # and one single vertex
    vs = valid & ~held
    O = P.copy()
    O[..., 0] += 1.5 * np.sin(np.mgrid[0:40, 0:50][1] / 8.0)     # a smooth other snap
    write_tifxyz(str(tmp_path / "orig.tifxyz"), P, valid, {"scale": [0.05, 0.05]})
    write_tifxyz(str(tmp_path / "snap.tifxyz"), np.where(vs[..., None], S, -1), vs, {"scale": [0.05, 0.05]})
    write_tifxyz(str(tmp_path / "other.tifxyz"), O, valid, {"scale": [0.05, 0.05]})
    abstain = np.where(held, 255, 0).astype(np.uint8)
    abstain[20:25, 20:25] = 2
    tifffile.imwrite(tmp_path / "abstain.tif", abstain)
    tifffile.imwrite(tmp_path / "release_mask.tif", held.astype(np.uint8))
    run(tmp_path / "m.json", "--original", tmp_path / "orig.tifxyz", "--snapped", tmp_path / "snap.tifxyz",
        "--compare", tmp_path / "other.tifxyz", "--codes", f"abstain={tmp_path / 'abstain.tif'}",
        "--codes", f"release_mask={tmp_path / 'release_mask.tif'}", "--blocks", "release_mask")
    r = json.load(open(tmp_path / "m.json"))
    m = r["moves"]
    assert r["grid"] == [40, 50] and r["original_vertices"] == 2000 and r["snapped_vertices"] == 2000 - 36
    assert r["snapped_where_original_has_none"] == 0
    sign = np.sign(m["mean"])                         # the grid normal of this plane is +z or -z
    move = sign * (-1) * (-3.0 + rough)               # what the tool should see, up to the normal's sign
    assert abs(m["mean"]) == pytest.approx(3.0, abs=0.2) and m["sd"] == pytest.approx(2.0, abs=0.15)
    assert abs(m["neighbour_r"]) < 0.15 and m["smoothed_1_vertex"]["neighbour_r"] > 0.7   # white noise smoothed at 1 vertex: about exp(-1/4)
    assert m["tangential_mean"] < 1e-3
    assert m["share_abs_over_10"] == pytest.approx(float((np.abs(move[vs]) > 10).mean()), abs=1e-9)
    assert r["codes"]["abstain"]["counts"] == {"0": 2000 - 36 - 25, "2": 25, "255": 36}
    assert r["blocks"] == dict(code="release_mask", n=2, largest=35, sizes=[35, 1])
    c = r["compare"]["moves"]
    assert c["neighbour_r"] > 0.9 and abs(c["correlation_with_snapped"]) < 0.2


def test_grids_must_match(tmp_path):
    write_tifxyz(str(tmp_path / "a.tifxyz"), plane(10, 12), np.ones((10, 12), bool), {"scale": [0.05, 0.05]})
    write_tifxyz(str(tmp_path / "b.tifxyz"), plane(11, 12), np.ones((11, 12), bool), {"scale": [0.05, 0.05]})
    with pytest.raises(subprocess.CalledProcessError):
        run(tmp_path / "x.json", "--original", tmp_path / "a.tifxyz", "--snapped", tmp_path / "b.tifxyz")
