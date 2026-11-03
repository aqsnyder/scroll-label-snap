"""tools/face_step_check.py on synthetic renders: a mesh on the face scores all gap outside and all
papyrus inside; one sitting 2 voxels too deep doesn't, and the layer direction is honoured."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "face_step_check.py")


def render(path, face_at, normals, center=0.0, layers=21, shape=(300, 200)):
    """Layers along the render's normal; papyrus (200) below the face, gap (40) beyond it, where the
    face is at `face_at` voxels along the snapper's normal (= grid normal here, normal_sign 1)."""
    offs = center + np.arange(layers) - (layers - 1) / 2
    g = 1 if normals == "grid" else -1
    t = g * offs                                   # along the snapper's normal
    col = np.where(t < face_at, 200, 40).astype(np.uint8)
    vol = np.broadcast_to(col[:, None, None], (layers,) + shape).copy()
    vol[:, :, :20] = 0                             # off the mesh
    z = zarr.open_array(str(path), mode="w", shape=vol.shape, chunks=(layers, 64, 64), dtype="u1")
    z[:] = vol
    z.attrs.update(normals=normals, layer_offsets=[float(o) for o in offs])
    return str(path)


def test_face_step(tmp_path):
    good = render(tmp_path / "good.zarr", 0.5, "grid")
    deep = render(tmp_path / "deep.zarr", 2.5, "grid")          # the mesh sits 2 voxels inside the face
    flipped = render(tmp_path / "flip.zarr", 0.5, "flipped", center=1.0)
    subprocess.run([sys.executable, TOOL, str(tmp_path / "out.json"), "--render", f"good={good}", "--render",
                    f"deep={deep}", "--render", f"flipped={flipped}", "--face", "0.5", "--band", "100"], check=True)
    r = json.load(open(tmp_path / "out.json"))["renders"]
    assert r["good"]["both"] == 1.0 and r["flipped"]["both"] == 1.0
    assert r["deep"]["gap_outside"] == 0.0 and r["deep"]["papyrus_inside"] == 1.0
    assert r["good"]["pixels"] == 300 * 180
