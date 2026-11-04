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
    assert "margin" not in res


def test_compare_margin_leaves_out_pixels_near_holes(tmp_path):
    H, W = 300, 400
    valid = np.ones((H, W), bool)
    valid[100:120, 200:230] = False                       # a hole in one render
    near = ds.away_from_invalid(valid, 10)
    assert not near[:10].any() and not near[:, -10:].any()  # the image edge counts as invalid
    assert near[10:90, 10:390].all() and not near[90:130, 190:240].any()
    assert near[89, 189] and near[131, 241]
    # the hole raises one map around it: with the margin, the compare no longer sees that
    ref = np.full((H, W), 100, np.uint8)
    holey = ref.copy()
    holey[80:140, 180:250] = 140
    holey[~valid] = 0
    for name, img in (("ref", ref), ("holey", holey)):
        tifffile.imwrite(tmp_path / f"{name}.tif", img)
    full = np.ones((H, W), bool)
    r1, r2 = render_zarr(tmp_path / "r1.zarr", full), render_zarr(tmp_path / "r2.zarr", valid)
    run("compare", tmp_path / "m.json", "--reference", f"ref={tmp_path / 'ref.tif'}", "--map", f"holey={tmp_path / 'holey.tif'}",
        "--renders", r1, r2, "--tile", 50, "--margin", 30)
    res = json.load(open(tmp_path / "m.json"))
    assert res["margin"] == 30 and res["pixels_before_margin"] == int(valid.sum())
    assert res["pixels"] == int(ds.away_from_invalid(valid, 30).sum())
    assert res["maps"]["holey"]["whole_map"]["mean"] == pytest.approx(100 / 255)
    assert np.array_equal(ds.away_from_invalid(valid, 0), valid)


def test_pearson_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, 100_000).astype(np.uint8)
    y = np.clip(x.astype(int) + rng.integers(-40, 40, x.size), 0, 255).astype(np.uint8)
    assert ds.pearson(x, y, band=7_777) == pytest.approx(np.corrcoef(x.astype(float), y.astype(float))[0, 1], abs=1e-9)


def test_filter_parts_add_up_and_random_phase_keeps_the_spread(tmp_path):
    rng = np.random.default_rng(3)
    P = grid(rows=40, cols=50)
    valid = np.ones(P.shape[:2], bool)
    valid[:3, :4] = False
    r, c = np.mgrid[0:40, 0:50]
    move = 1.5 * np.sin(c / 9.0) + rng.normal(0, 1.0, (40, 50))    # a large-scale part plus vertex-scale roughness
    S = P.copy()
    S[..., 0] += move                                                 # the grid normal of this plane is along z
    write_tifxyz(str(tmp_path / "orig.tifxyz"), P, valid, {"scale": [0.05, 0.05]})
    write_tifxyz(str(tmp_path / "snap.tifxyz"), S, valid, {"scale": [0.05, 0.05]})
    outs = {}
    for name, opt in (("lp", ["--low-pass", "2"]), ("hp", ["--high-pass", "2"]), ("rp", ["--random-phase", "7"]), ("rp2", ["--random-phase", "7"])):
        run("filter", tmp_path / f"{name}.tifxyz", "--original", tmp_path / "orig.tifxyz", "--snapped", tmp_path / "snap.tifxyz", *opt)
        outs[name] = read_tifxyz(str(tmp_path / f"{name}.tifxyz"))
    Pr, vr, _ = read_tifxyz(str(tmp_path / "orig.tifxyz"))
    v = outs["lp"][1]
    assert v.sum() > 0.9 * valid.sum() and not v[:3, :4].any()
    dz = lambda Q: (Q[..., 0] - Pr[..., 0])  # noqa: E731
    # the large-scale part and the rest add up to the snapper's move, which runs along the normal (z)
    np.testing.assert_allclose((dz(outs["lp"][0]) + dz(outs["hp"][0]))[v], move[v], atol=1e-3)
    np.testing.assert_allclose(outs["lp"][0][..., 1:][v], Pr[..., 1:][v], atol=1e-3)
    # the large-scale part is smooth: neighbouring vertices move alike
    assert ds.lag1(dz(outs["lp"][0]), v) > 0.9 > ds.lag1(dz(outs["hp"][0]), v)
    # random phases: same mean and spread, reproducible, unrelated to the move
    e = dz(outs["rp"][0])
    assert e[v].mean() == pytest.approx(move[v].mean(), abs=1e-3) and e[v].std() == pytest.approx(move[v].std(), rel=1e-3)
    np.testing.assert_array_equal(outs["rp"][0], outs["rp2"][0])
    assert abs(np.corrcoef(e[v], move[v])[0, 1]) < 0.3
    # --minus-mean: the same moves (or their large-scale part) at the original's average depth
    S2 = P.copy()
    S2[..., 0] += move + 3.0
    write_tifxyz(str(tmp_path / "snap_off.tifxyz"), S2, valid, {"scale": [0.05, 0.05]})
    for name, opt in (("mm", ["--minus-mean"]), ("lpmm", ["--low-pass", "2", "--minus-mean"])):
        run("filter", tmp_path / f"{name}.tifxyz", "--original", tmp_path / "orig.tifxyz", "--snapped", tmp_path / "snap_off.tifxyz", *opt)
        outs[name] = read_tifxyz(str(tmp_path / f"{name}.tifxyz"))
    mm, lpmm = dz(outs["mm"][0]), dz(outs["lpmm"][0])
    vm = outs["mm"][1]
    assert mm[vm].mean() == pytest.approx(0.0, abs=1e-4) and lpmm[vm].mean() == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(mm[vm], (move + 3.0)[vm] - (move + 3.0)[vm].mean(), atol=1e-3)
    np.testing.assert_allclose(lpmm[vm], dz(outs["lp"][0])[vm] - dz(outs["lp"][0])[vm].mean(), atol=1e-3)
    with pytest.raises(subprocess.CalledProcessError):
        run("filter", tmp_path / "none.tifxyz", "--original", tmp_path / "orig.tifxyz", "--snapped", tmp_path / "snap.tifxyz")
