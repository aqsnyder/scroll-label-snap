"""The depth diagnostic's helpers (tools/depth_sweep.py) on synthetic data: the mirrored mesh, the
negated depth and the map comparison."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, ".."))
import depth_sweep as ds  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def run(*args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, "depth_sweep.py")] + [str(a) for a in args],
                          check=True, capture_output=True, text=True).stdout


def grid(rows=6, cols=7, z0=100.0):
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return np.stack([np.full_like(r, z0), 40 + 20 * r, 40 + 20 * c], -1)


def test_mirror_moves_the_other_way(tmp_path):
    P = grid()
    valid = np.ones(P.shape[:2], bool)
    shift = np.zeros_like(P)
    shift[..., 0] = np.linspace(-2, 3, P.shape[1])[None, :]
    S = P + shift
    v_snap = valid.copy()
    v_snap[0, 0] = False
    write_tifxyz(str(tmp_path / "orig.tifxyz"), P, valid, {"scale": [0.05, 0.05]})
    write_tifxyz(str(tmp_path / "snap.tifxyz"), S, v_snap, {"scale": [0.05, 0.05]})
    run("mirror", tmp_path / "mirror.tifxyz", "--original", tmp_path / "orig.tifxyz", "--snapped", tmp_path / "snap.tifxyz")
    M, vm, meta = read_tifxyz(str(tmp_path / "mirror.tifxyz"))
    assert not vm[0, 0] and vm.sum() == valid.sum() - 1
    np.testing.assert_allclose(M[vm], (P - shift)[vm], atol=1e-5)
    assert meta["scale"] == [0.05, 0.05]


def test_negate_keeps_the_rest(tmp_path):
    depth = np.array([[1.5, np.nan], [-0.5, 2.0]], np.float32)
    np.savez_compressed(tmp_path / "d.npz", depth=depth, valid=np.isfinite(depth), roi=np.array([0, 2, 0, 2]),
                        stats=np.array(json.dumps({"target": "t"})))
    run("negate", tmp_path / "n.npz", tmp_path / "d.npz")
    z = np.load(tmp_path / "n.npz")
    np.testing.assert_array_equal(z["depth"], -depth)
    np.testing.assert_array_equal(z["valid"], np.isfinite(depth))
    np.testing.assert_array_equal(z["roi"], [0, 2, 0, 2])
    st = json.loads(str(z["stats"]))
    assert st["target"] == "t" and st["negated_from"].endswith("d.npz")


def render_zarr(path, valid, layers=21, value=90):
    z = zarr.open_array(str(path), mode="w", shape=(layers,) + valid.shape, chunks=(layers, 128, 128), dtype="u1")
    vol = np.zeros((layers,) + valid.shape, np.uint8)
    vol[:, valid] = value
    z[:] = vol
    return str(path)


def test_compare(tmp_path):
    rng = np.random.default_rng(0)
    H, W = 512, 768
    valid = np.ones((H, W), bool)
    valid[:, :100] = False
    ref = rng.integers(60, 190, (H, W)).astype(np.uint8)
    ref[~valid] = 0
    lower = np.clip(ref.astype(int) - 10, 0, 255).astype(np.uint8)
    lower[~valid] = 0
    for name, img in (("ref", ref), ("same", ref.copy()), ("lower", lower)):
        tifffile.imwrite(tmp_path / f"{name}.tif", img)
    r = render_zarr(tmp_path / "r.zarr", valid)
    run("compare", tmp_path / "c.json", "--reference", f"ref={tmp_path / 'ref.tif'}",
        "--map", f"same={tmp_path / 'same.tif'}", "--map", f"lower={tmp_path / 'lower.tif'}",
        "--map", f"gone={tmp_path / 'gone.tif'}", "--renders", r)
    res = json.load(open(tmp_path / "c.json"))
    assert res["pixels"] == int(valid.sum())
    same, low = res["maps"]["same"], res["maps"]["lower"]
    assert same["identical"] and same["max_abs_difference"] == 0
    assert same["pearson_r"] == pytest.approx(1.0) and same["d_p99_median"] == 0
    assert not low["identical"] and low["max_abs_difference"] == 10
    assert low["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert low["d_p99_median"] == pytest.approx(-10 / 255, abs=1e-6)
    assert low["share_tiles_d_p99_below_minus_0_02"] == 1.0 and low["share_tiles_d_p99_above_0_02"] == 0.0
    assert low["whole_map"]["mean"] == pytest.approx(low["whole_map"]["mean_reference"] - 10 / 255, abs=1e-6)
    assert res["maps"]["gone"]["missing"]


def test_pearson_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, 100_000).astype(np.uint8)
    y = np.clip(x.astype(int) + rng.integers(-40, 40, x.size), 0, 255).astype(np.uint8)
    assert ds.pearson(x, y, band=7_777) == pytest.approx(np.corrcoef(x.astype(float), y.astype(float))[0, 1], abs=1e-9)
