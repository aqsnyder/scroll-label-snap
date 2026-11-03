"""Whole-segment snapping (tools/snap_segment.py) on a small synthetic scan: tiling and blending
must give the same surface as snapping the mesh in one piece."""
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
from scroll_label_snap.core import Params  # noqa: E402
from scroll_label_snap.mesh import snap_mesh  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def test_tiled_segment_matches_one_piece(tmp_path):
    from dataclasses import replace
    ct, _, _ = wavy_sheets((300, 160, 360), thickness=8, gap=10, blur=1.0, n_sheets=5, base=30.0)
    g = zarr.open_group(zarr.DirectoryStore(str(tmp_path / "scan.zarr"), dimension_separator="/"), mode="w")
    g.create_dataset("0", data=ct, chunks=(64, 64, 64), compressor=None)
    zs, xs = np.arange(30, 271, 20.0), np.arange(30, 331, 20.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    face = 30 + 5 * np.sin(2 * np.pi * X / 70) + 3 * np.cos(2 * np.pi * Z / 55) + 2 * 18 + 8
    P = np.stack([Z, face + 1.5 * np.sin(2 * np.pi * X / 60) * np.cos(2 * np.pi * Z / 50), X], -1).astype(np.float32)
    valid = np.ones(P.shape[:2], bool)
    write_tifxyz(str(tmp_path / "seg.tifxyz"), P, valid, {"scale": [0.05, 0.05]}, uuid="synth")
    tool = os.path.join(HERE, "..", "tools", "snap_segment.py")
    subprocess.run([sys.executable, tool, str(tmp_path / "out"), "--segment", str(tmp_path / "seg.tifxyz"),
                    "--volume", str(tmp_path / "scan.zarr"), "--tile", "8", "--overlap", "2", "--workers", "1",
                    "--sample-tiles", "4", "--normals", "flipped"], check=True, capture_output=True)
    summary = json.load(open(tmp_path / "out" / "snap_segment.json"))
    assert not summary["tiles_failed"]
    one = snap_mesh(ct, P, valid, replace(Params(), convention_offset=summary["convention_offset"]), normal_sign=-1)
    D, vd, _ = read_tifxyz(str(tmp_path / "out" / "synth_snapped_dense.tifxyz"))
    both = vd & one.dense_valid
    diff = np.linalg.norm(D - one.dense_P, axis=-1)[both]
    assert both.mean() > 0.95 and np.mean(diff <= 0.5) > 0.95 and diff.max() < 1.0
    G, vg, _ = read_tifxyz(str(tmp_path / "out" / "synth_snapped.tifxyz"))
    assert vg.all() and G.shape == P.shape
