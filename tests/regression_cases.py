"""Small deterministic cases whose outputs are stored in tests/reference/regression_reference.npz, so a
change meant to be behaviour-neutral (an optimisation, a refactor) can be checked against them.

  python tests/make_regression_reference.py   # rewrite the reference after an intended change
  python -m pytest tests/test_regression.py   # compare against it
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params, snap  # noqa: E402
from scroll_label_snap.mesh import snap_mesh  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "regression_reference.npz")


def wavy_sheets(shape, thickness=8, gap=10, blur=1.0, noise=10.0, seed=0, n_sheets=4, base=20.0):
    """CT of parallel wavy sheets stacked along y, and the y of each sheet's +y (recto) face."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    s = base + 5 * np.sin(2 * np.pi * x / 70) + 3 * np.cos(2 * np.pi * z / 55)
    inside = np.zeros(shape, bool)
    faces = []
    for k in range(n_sheets):
        lo = s + k * (thickness + gap)
        inside |= (y >= lo) & (y < lo + thickness)
        faces.append(lo + thickness)
    ct = ndi.gaussian_filter(np.where(inside, 170.0, 60.0), blur)
    ct += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(ct, 0, 255).astype(np.uint8), faces, (z, y, x)


def label_case():
    """A 128^3 cube with recto labels drifted off the faces by a smooth field; one corner unannotated."""
    shape = (128, 128, 128)
    ct, faces, (z, y, x) = wavy_sheets(shape, n_sheets=4, base=24.0)
    drift = 2.5 * np.sin(2 * np.pi * x / 45 + 0.7) * np.cos(2 * np.pi * z / 60)
    lab = np.zeros(shape, np.uint8)
    for f in faces:
        c = f + drift
        lab[(y >= c - 1.0) & (y < c + 1.0)] = 1
    lab[:24, :, :24] = 2
    return ct, lab


def mesh_case():
    """The synthetic mesh of tests/test_mesh.py with a smooth drift along its normals."""
    shape = (130, 150, 160)
    ct, faces, _ = wavy_sheets(shape, thickness=8, gap=10, blur=1.2, n_sheets=4, base=40.0)
    zs, xs = np.arange(25, 106, 20.0), np.arange(25, 136, 20.0)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    face = 40 + 5 * np.sin(2 * np.pi * X / 70) + 3 * np.cos(2 * np.pi * Z / 55) + 18 + 8
    d = 1.5 * np.sin(2 * np.pi * X / 50) * np.cos(2 * np.pi * Z / 45)
    P = np.stack([Z, face + d, X], -1).astype(np.float32)
    return ct, P, np.ones(P.shape[:2], bool)


def run_all():
    """Outputs that should not change unless the method changes on purpose."""
    ct, lab = label_case()
    r = snap(ct, lab, Params())
    mct, P, valid = mesh_case()
    m = snap_mesh(mct, P, valid, Params())
    return dict(label_shift=r.shift.astype(np.float32), label_confidence=r.confidence.astype(np.float32),
                label_volume_sum=np.array([int((r.labels == 1).sum()), int((r.labels == 2).sum())]),
                label_unchanged_2=np.array([bool((r.labels[:24, :, :24] == 2).all())]),
                mesh_dense_shift=m.dense_shift.astype(np.float32), mesh_vertex_shift=m.vertex_shift.astype(np.float32))
