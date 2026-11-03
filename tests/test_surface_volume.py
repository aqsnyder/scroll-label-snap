"""tools/render_surface_volume.py on a small synthetic scan: the tiled, parallel render must equal
sampling the densified mesh directly, be zero off the mesh, and a --rows/--cols window must equal
the same crop of the full render."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("fsspec")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from regression_cases import wavy_sheets  # noqa: E402
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import densify, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import write_tifxyz  # noqa: E402

TOOL = os.path.join(HERE, "..", "tools", "render_surface_volume.py")


def test_render_matches_direct_sampling(tmp_path):
    ct, _, _ = wavy_sheets((300, 160, 360), thickness=8, gap=10, blur=1.0, n_sheets=5, base=30.0)
    g = zarr.open_group(zarr.DirectoryStore(str(tmp_path / "scan.zarr"), dimension_separator="/"), mode="w")
    g.create_dataset("0", data=ct, chunks=(64, 64, 64), compressor=None)
    zs, xs = np.arange(30, 271, 20.0), np.arange(30, 331, 20.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    Y = 30 + 5 * np.sin(2 * np.pi * X / 70) + 3 * np.cos(2 * np.pi * Z / 55) + 44
    P = np.stack([Z, Y, X], -1).astype(np.float32)
    valid = np.ones(P.shape[:2], bool)
    valid[:3, :4] = False                                   # a hole in one corner
    P[~valid] = -1
    write_tifxyz(str(tmp_path / "seg.tifxyz"), P, valid, {"scale": [0.05, 0.05]}, uuid="synth")

    def run(out, *extra):
        r = subprocess.run([sys.executable, TOOL, str(out), "--segment", str(tmp_path / "seg.tifxyz"),
                            "--volume", str(tmp_path / "scan.zarr"), "--workers", "2", "--tile", "128",
                            "--chunk", "64", *extra], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return zarr.open_array(str(out), mode="r")[:]

    sv = run(tmp_path / "sv.zarr")
    H, W = P.shape[:2]
    assert sv.shape == (21, (H - 1) * 20 + 1, (W - 1) * 20 + 1) and sv.dtype == np.uint8
    N, ok = vertex_normals(P, valid)
    Pd, Nd, vd = densify(P, N, ok, 20)
    assert (sv[:, ~vd] == 0).all()                          # nothing off the mesh
    rng = np.random.default_rng(0)
    idx = np.argwhere(vd)[rng.choice(int(vd.sum()), 3000, replace=False)]
    q, n = Pd[idx[:, 0], idx[:, 1]], Nd[idx[:, 0], idx[:, 1]]
    for li, d in enumerate(np.arange(-10, 11)):
        want = np.clip(np.rint(sample(ct.astype(np.float32), q + d * n)), 0, 255)
        assert np.abs(sv[li, idx[:, 0], idx[:, 1]].astype(int) - want).max() <= 1, li
    assert json.load(open(str(tmp_path / "sv.zarr") + ".json"))["shape"] == list(sv.shape)

    win = run(tmp_path / "win.zarr", "--rows", "3", "10", "--cols", "4", "13")
    ref = sv[:, 3 * 20:9 * 20 + 1, 4 * 20:12 * 20 + 1]
    d = np.abs(win.astype(int) - ref)
    print("window vs full: max diff", d.max(), "share differing", (d > 0).mean(), "shapes", win.shape, ref.shape)
    assert win.shape == ref.shape and d.max() <= 1 and (d > 0).mean() < 0.001


def test_compare_finds_shift_and_layer_alignment(tmp_path):
    rng = np.random.default_rng(1)
    from scipy import ndimage as ndi
    base = ndi.gaussian_filter(rng.normal(size=(21, 240, 300)), (2, 3, 3)).astype(np.float32)
    ours = np.clip(128 + 600 * base, 1, 255).astype(np.uint8)
    zarr.open_array(str(tmp_path / "ours.zarr"), mode="w", shape=ours.shape, chunks=(21, 64, 64), dtype="u1")[:] = ours
    ref = np.roll(ours[::-1][2:19], (3, -2), axis=(1, 2))           # reversed, offset, shifted in plane
    ref_last = np.moveaxis(ref, 0, -1)                               # layers last, as some volumes are
    zarr.open_array(str(tmp_path / "ref.zarr"), mode="w", shape=ref_last.shape, chunks=(64, 64, 17), dtype="u1")[:] = ref_last
    tool = os.path.join(HERE, "..", "tools", "compare_surface_volumes.py")
    r = subprocess.run([sys.executable, tool, str(tmp_path / "cmp.json"), str(tmp_path / "ours.zarr"),
                        str(tmp_path / "ref.zarr")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.load(open(tmp_path / "cmp.json"))
    b = res["best_layer_alignment"]
    assert res["inplane_shift_yx"] == [3, -2]
    assert b["order"] == "reversed" and b["ref_layer_of_our_layer_0"] == 18 and b["mean_corr"] > 0.99
