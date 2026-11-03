"""The 0841 follow-up tools on synthetic data: depth carried between parallel planes, a render at
that depth, the fine-to-coarse label transfer and the paired scoring."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, ".."))
from scroll_label_snap.tifxyz import write_tifxyz  # noqa: E402


def plane(path, z0, rows=12, cols=14, step=20.0, tilt=0.0, scale=0.05):
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float64)
    P = np.stack([z0 + tilt * c * step, 40 + r * step, 40 + c * step], -1).astype(np.float32)
    write_tifxyz(str(path), P, np.ones((rows, cols), bool), {"scale": [scale, scale]})
    return str(path)


def run(*args):
    subprocess.run([sys.executable] + [str(a) for a in args], check=True)


def test_depth_transfer_and_render(tmp_path):
    tgt = plane(tmp_path / "t.tifxyz", 100.0)
    srf = plane(tmp_path / "s.tifxyz", 103.0)
    run(os.path.join(TOOLS, "depth_transfer.py"), tmp_path / "d.npz", "--target", tgt, "--surface", srf,
        "--roi", 0, 200, 0, 240)
    d = np.load(tmp_path / "d.npz")
    dep = d["depth"]
    have = np.isfinite(dep)
    assert have.mean() > 0.95
    assert np.allclose(np.abs(dep[have]), 3.0, atol=1e-3)
    run(os.path.join(TOOLS, "depth_transfer.py"), tmp_path / "d2.npz", "--target", tgt, "--surface", srf,
        "--roi", 0, 200, 0, 240, "--max-depth", 2)
    assert not np.isfinite(np.load(tmp_path / "d2.npz")["depth"]).any()
    # a scan whose value is its z coordinate: the render at depth t samples z = 100 + t + offset
    vol = np.broadcast_to(np.arange(200, dtype=np.float32)[:, None, None], (200, 300, 340)).astype(np.uint8)
    g = zarr.open_group(zarr.DirectoryStore(str(tmp_path / "scan.zarr"), dimension_separator="/"), mode="w")
    g.create_dataset("0", data=vol, chunks=(64, 64, 64), compressor=None)
    run(os.path.join(TOOLS, "render_depth.py"), tmp_path / "r.zarr", "--target", tgt, "--depth", tmp_path / "d.npz",
        "--volume", tmp_path / "scan.zarr", "--layers", 5, "--workers", 2)
    z = zarr.open_array(str(tmp_path / "r.zarr"), mode="r")
    out = np.asarray(z[:])
    assert out.shape == (5, 200, 240)
    sign = np.sign(np.nanmedian(dep))          # the depth is along the target's grid normal
    m = have & (out[2] > 0)
    expect = 100 + dep[m]
    assert np.abs(out[2][m] - expect).max() <= 1
    assert np.abs(out[3][m].astype(float) - out[2][m] - sign * 1).max() <= 1   # +1 layer: one voxel along the normal


def test_fine_labels(tmp_path):
    corr = dict(fr=4.0, orr=1.0, fc=4.0, oc=1.0, coarse_k=20, fine_k=20, coarse_grid=[30, 30])
    json.dump(corr, open(tmp_path / "corr.json", "w"))
    # fine label image: coarse pixel (r, c) covers fine pixels around R = 4r + 20, C = 4c + 20
    Hf = Wf = 2400
    sup = np.zeros((Hf, Wf), np.uint8)
    lab = np.zeros((Hf, Wf), np.uint8)
    sup[420:1220, 420:1620] = 255                       # coarse rows 100-300, cols 100-400
    lab[420:820, 420:1620] = 255                        # ink in coarse rows 100-200
    tifffile.imwrite(tmp_path / "sup.tif", sup)
    tifffile.imwrite(tmp_path / "lab.tif", lab)
    run(os.path.join(TOOLS, "fine_labels.py"), "roi", tmp_path / "roi.json", "--corr", tmp_path / "corr.json",
        "--supervision", tmp_path / "sup.tif", "--margin", 10)
    roi = json.load(open(tmp_path / "roi.json"))["roi"]
    assert roi == [90, 310, 90, 410]
    run(os.path.join(TOOLS, "fine_labels.py"), "transfer", tmp_path / "lab.npz", "--corr", tmp_path / "corr.json",
        "--labels", tmp_path / "lab.tif", "--supervision", tmp_path / "sup.tif", "--roi", *roi)
    L = np.load(tmp_path / "lab.npz")
    s, ink = L["supervised"], L["ink"]
    assert s.shape == (220, 320)
    assert s[10:210, 10:310].mean() > 0.99 and not s[:8].any()
    assert ink[12:108, 12:308].all() and not ink[112:208].any()
    # maps: one tracks the ink, one is noise; renders valid everywhere
    good = np.where(ink, 180, 90).astype(np.uint8)
    noise = np.random.default_rng(0).integers(60, 190, s.shape).astype(np.uint8)
    tifffile.imwrite(tmp_path / "good.tif", good)
    tifffile.imwrite(tmp_path / "noise.tif", noise)
    vol = np.full((21,) + s.shape, 90, np.uint8)
    zr = zarr.open_array(str(tmp_path / "r.zarr"), mode="w", shape=vol.shape, chunks=(21, 64, 64), dtype="u1")
    zr[:] = vol
    run(os.path.join(TOOLS, "fine_labels.py"), "score", tmp_path / "score.json", "--labels-npz", tmp_path / "lab.npz",
        "--map", f"original={tmp_path / 'noise.tif'}", "--map", f"snapped={tmp_path / 'good.tif'}",
        "--renders", tmp_path / "r.zarr", "--reference", "original", "--blocks", 32, "--boot", 200)
    r = json.load(open(tmp_path / "score.json"))
    assert r["scores"]["snapped"]["auc"] == 1.0 and abs(r["scores"]["original"]["auc"] - 0.5) < 0.05
    ci = r["scores"]["snapped"]["minus_reference"]["auc_95"]
    assert ci["lo"] > 0.3 and r["pixels"] == int(s.sum())
