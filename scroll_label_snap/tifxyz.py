"""Read and write VC3D tifxyz surfaces.

A tifxyz surface is a folder holding meta.json and x.tif, y.tif, z.tif: three float32 images, one
pixel per grid vertex, giving the vertex's volume coordinates. Missing vertices are (-1, -1, -1).
An optional mask.tif (the grid's size or a whole multiple of it) removes vertices, and any other
.tif in the folder is an extra per-vertex channel. meta.json's "scale" is grid steps per voxel, so
0.05 means vertices about 20 voxels apart.

Here vertices are handled as an (H, W, 3) array in (z, y, x) order, like the CT volumes.
Rules follow volume-cartographer's QuadSurface loader (core/src/QuadSurface.cpp).
"""
import json
import os
import uuid as uuidlib

import numpy as np
import tifffile


def valid_vertices(x, y, z):
    """VC3D's rule: a vertex exists unless a coordinate is -1 or not finite, or z <= 0."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return ok & (x != -1) & (y != -1) & (z != -1) & (z > 0)


def mask_validity(m, shape):
    """Vertices kept by mask.tif: channel (page) 0, the grid's size or a whole multiple of it;
    a vertex is kept only when every mask value it covers is at least 255. None if the mask
    does not fit the grid (VC3D ignores it then)."""
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[0] if m.shape[0] < m.shape[-1] else m[..., 0]
    H, W = shape
    if m.ndim != 2 or m.shape[0] % H or m.shape[1] % W:
        return None
    sy, sx = m.shape[0] // H, m.shape[1] // W
    return (m.reshape(H, sy, W, sx) >= 255).all(axis=(1, 3))


def read_tifxyz(folder, use_mask=True):
    """Returns (P, valid, meta): P (H, W, 3) float32 in (z, y, x) order, valid (H, W) bool."""
    x, y, z = (tifffile.imread(os.path.join(folder, f"{c}.tif")).astype(np.float32) for c in "xyz")
    valid = valid_vertices(x, y, z)
    mp = os.path.join(folder, "mask.tif")
    if use_mask and os.path.exists(mp):
        mv = mask_validity(tifffile.imread(mp), x.shape)
        if mv is not None:
            valid &= mv
    meta = {}
    if os.path.exists(os.path.join(folder, "meta.json")):
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
    P = np.stack([z, y, x], -1)
    P[~valid] = -1
    return P, valid, meta


def write_tifxyz(folder, P, valid, meta=None, scale=None, channels=None, uuid=None):
    """Write a tifxyz folder. P (H, W, 3) in (z, y, x) order; invalid vertices are written as -1.
    meta is copied, with scale (grid steps per voxel), bbox, uuid (default: the folder name),
    format and type updated the way VC3D does on save. channels: extra per-vertex images,
    name -> (H, W) array."""
    os.makedirs(folder, exist_ok=True)
    P = np.asarray(P, np.float32).copy()
    P[~valid] = -1
    for k, c in enumerate("zyx"):
        tifffile.imwrite(os.path.join(folder, f"{c}.tif"), np.ascontiguousarray(P[..., k]))
    m = dict(meta or {})
    if scale is not None:
        m["scale"] = [float(scale), float(scale)]
    if valid.any():
        v = P[valid]
        m["bbox"] = [[float(v[:, 2].min()), float(v[:, 1].min()), float(v[:, 0].min())],
                     [float(v[:, 2].max()), float(v[:, 1].max()), float(v[:, 0].max())]]
    m["uuid"] = uuid or os.path.basename(os.path.normpath(folder)) or str(uuidlib.uuid4())
    m["format"] = "tifxyz"
    m["type"] = "seg"
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(m, f, indent=1)
    for name, img in (channels or {}).items():
        tifffile.imwrite(os.path.join(folder, f"{name}.tif"), np.ascontiguousarray(img))
