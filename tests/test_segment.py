"""Whole-segment snapping (tools/snap_segment.py) on a small synthetic scan: tiling and blending
must give the same surface as snapping the mesh in one piece, and reading the scan over http(s)
through the chunk cache must give the same result as reading it from disk."""
import functools
import http.server
import json
import os
import subprocess
import sys
import threading

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

TOOL = os.path.join(HERE, "..", "tools", "snap_segment.py")


def make_scene(tmp_path, root):
    ct, _, _ = wavy_sheets((300, 160, 360), thickness=8, gap=10, blur=1.0, n_sheets=5, base=30.0)
    g = zarr.open_group(zarr.DirectoryStore(str(root / "scan.zarr"), dimension_separator="/"), mode="w")
    g.create_dataset("0", data=ct, chunks=(64, 64, 64), compressor=None)
    zs, xs = np.arange(30, 271, 20.0), np.arange(30, 331, 20.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    face = 30 + 5 * np.sin(2 * np.pi * X / 70) + 3 * np.cos(2 * np.pi * Z / 55) + 2 * 18 + 8
    P = np.stack([Z, face + 1.5 * np.sin(2 * np.pi * X / 60) * np.cos(2 * np.pi * Z / 50), X], -1).astype(np.float32)
    valid = np.ones(P.shape[:2], bool)
    write_tifxyz(str(tmp_path / "seg.tifxyz"), P, valid, {"scale": [0.05, 0.05]}, uuid="synth")
    return ct, P, valid


def run_tool(out, volume, workers=1, cache=None):
    cmd = [sys.executable, TOOL, str(out), "--segment", str(out.parent / "seg.tifxyz"), "--volume", volume,
           "--tile", "8", "--overlap", "2", "--workers", str(workers), "--sample-tiles", "4",
           "--normals", "flipped", "--name", "synth"]
    if cache:
        cmd += ["--cache", str(cache)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.load(open(out / "snap_segment.json"))


def test_tiled_segment_matches_one_piece(tmp_path):
    from dataclasses import replace
    ct, P, valid = make_scene(tmp_path, tmp_path)
    summary = run_tool(tmp_path / "out", str(tmp_path / "scan.zarr"))
    assert not summary["tiles_failed"]
    one = snap_mesh(ct, P, valid, replace(Params(), convention_offset=summary["convention_offset"], template_follows_offset=False), normal_sign=-1)
    D, vd, _ = read_tifxyz(str(tmp_path / "out" / "synth_snapped_dense.tifxyz"))
    both = vd & one.dense_valid
    diff = np.linalg.norm(D - one.dense_P, axis=-1)[both]
    assert both.mean() > 0.95 and np.mean(diff <= 0.5) > 0.95 and diff.max() < 1.0
    G, vg, _ = read_tifxyz(str(tmp_path / "out" / "synth_snapped.tifxyz"))
    assert vg.all() and G.shape == P.shape


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_segment_over_http_with_chunk_cache(tmp_path):
    pytest.importorskip("aiohttp")
    root = tmp_path / "srv"
    make_scene(tmp_path, root)
    # a masked scan leaves empty chunks out; this one is under the mesh, so it is read
    os.remove(root / "scan.zarr" / "0" / "2" / "1" / "2")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/scan.zarr/"
    cache = tmp_path / "cache"
    try:
        run_tool(tmp_path / "http", url, workers=2, cache=cache)
    finally:
        srv.shutdown()
        srv.server_close()
    run_tool(tmp_path / "local", str(root / "scan.zarr"))
    run_tool(tmp_path / "offline", url, workers=2, cache=cache)       # server gone: all from the cache
    names = [f for _, _, fs in os.walk(cache) for f in fs]
    assert any(f.endswith(".missing") for f in names) and not any(f.endswith(".tmp") for f in names)
    D0, v0, _ = read_tifxyz(str(tmp_path / "local" / "synth_snapped_dense.tifxyz"))
    for run in ("http", "offline"):
        D, v, _ = read_tifxyz(str(tmp_path / run / "synth_snapped_dense.tifxyz"))
        assert (v == v0).all() and np.abs(D[v] - D0[v0]).max() < 1e-5, run
