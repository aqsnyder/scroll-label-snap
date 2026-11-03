"""Mesh snapping on synthetic wavy sheets: a coarse tifxyz-like grid on the recto face of one
sheet. Run with: python -m pytest tests/test_mesh.py"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params  # noqa: E402
from scroll_label_snap.mesh import prepare_mesh, snap_mesh  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402

T, G = 8, 10                      # sheet thickness and gap (voxels)
SHAPE = (130, 150, 160)           # z, y, x


def wave(x, z):
    return 40 + 6 * np.sin(2 * np.pi * x / 90) + 4 * np.cos(2 * np.pi * z / 70)


def make_ct(seed=0):
    z, y, x = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]].astype(np.float32)
    base = wave(x, z)
    inside = np.zeros(SHAPE, bool)
    for k in range(4):
        lo = base + k * (T + G)
        inside |= (y >= lo) & (y < lo + T)
    ct = ndi.gaussian_filter(np.where(inside, 170.0, 60.0), 1.2)
    ct += np.random.default_rng(seed).normal(0, 10, SHAPE)
    return np.clip(ct, 0, 255).astype(np.uint8)


def face(x, z, depth=0.0):
    """y of the recto face of sheet 1 (gap on the +y side), or `depth` voxels into the papyrus."""
    return wave(x, z) + (T + G) + T - depth


def coarse_mesh(depth=0.0, step=20.0):
    zs, xs = np.arange(25, 106, step), np.arange(25, 136, step)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    P = np.stack([Z, face(X, Z, depth), X], -1).astype(np.float32)
    return P, np.ones(P.shape[:2], bool)


def face_error(Pd, vd, depth=0.0):
    q = Pd[vd]
    return q[:, 1] - face(q[:, 2], q[:, 0], depth)


CT = make_ct()


def test_snaps_coarse_mesh_onto_curved_face():
    P, v = coarse_mesh()
    prep = prepare_mesh(CT, P, v, Params())
    before = np.abs(face_error(prep.Pd, prep.vd))
    res = snap_mesh(CT, P, v, Params(), prep=prep)
    after = face_error(res.dense_P, res.dense_valid)
    assert res.factor == 7                                  # 20-voxel grid -> ~3-voxel samples
    assert np.mean(before <= 1) < 0.7                       # bilinear grid misses the curvature
    assert np.mean(np.abs(after) <= 1) > 0.9
    assert abs(np.median(after)) < 0.3


def test_orientation_does_not_depend_on_grid_order():
    P, v = coarse_mesh()
    a = snap_mesh(CT, P, v, Params())
    b = snap_mesh(CT, P[::-1].copy(), v, Params())
    assert a.info["normal_sign"] == -b.info["normal_sign"]
    assert np.allclose(a.dense_P, b.dense_P[::-1], atol=1e-3)


def test_recovers_drifted_mesh_and_keeps_its_convention():
    for depth in (0.0, 3.0):
        P, v = coarse_mesh(depth)
        N = prepare_mesh(CT, P, v, Params()).N
        rng = np.random.default_rng(1)
        k = rng.normal(size=(32, 3))
        k *= 2 * np.pi / 40 / np.linalg.norm(k, axis=1, keepdims=True)
        d = 1.5 * np.sqrt(2 / 32) * np.cos(P.reshape(-1, 3) @ k.T + rng.uniform(0, 6.3, 32)).sum(1)
        d = np.clip(d - np.median(d), -3.5, 3.5)            # zero median: the convention stays put
        Pd = P + d.reshape(P.shape[:2])[..., None] * N
        res = snap_mesh(CT, Pd, v, Params())
        e = face_error(res.dense_P, res.dense_valid, depth)
        assert np.mean(np.abs(e) <= 1) > 0.85, depth
        assert abs(np.median(e)) < 0.6, depth


def test_tifxyz_round_trip(tmp_path):
    P, v = coarse_mesh()
    v[1, 2] = False
    write_tifxyz(str(tmp_path / "s.tifxyz"), P, v, {"scale": [0.05, 0.05]})
    Q, w, meta = read_tifxyz(str(tmp_path / "s.tifxyz"))
    assert np.array_equal(w, v) and np.allclose(Q[v], P[v]) and (Q[~v] == -1).all()
    assert meta["format"] == "tifxyz" and meta["scale"] == [0.05, 0.05]
    assert np.allclose(meta["bbox"][0], [P[v][:, 2].min(), P[v][:, 1].min(), P[v][:, 0].min()])
