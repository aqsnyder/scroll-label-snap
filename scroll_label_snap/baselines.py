"""Other ways to snap a traced mesh, for comparison with the snapper.

snap_to_prediction is a port of mesh/snap_meshes.py by Erwin Nieuwlaar
(https://github.com/Nieuwlaar/pherc343-first-letters, MIT licence), which snapped the meshes of his
PHerc. 343 First Letters submission onto the m7 surface prediction. Per vertex of the grid it moves
along the mesh normal to the centre of the nearest predicted-sheet run within +-max_off voxels
(sub-voxel parabolic refinement on a [1, 2, 1]-smoothed profile). Then it regularises the offsets so
the mesh cannot hop between sheets: offsets more than `jump` voxels from the 5 x 5 nan-median of
their neighbourhood are rejected, gaps are refilled by normalised convolution from the neighbours,
and the field is smoothed with a Gaussian of `sigma` grid cells. Vertices with no sheet anywhere
in their neighbourhood keep their position. The port reads a crop of the prediction instead of
the whole zarr; the rules and defaults are his.

The ported part (snap_to_prediction and its helpers) is used under the MIT licence:

  Copyright (c) 2026 Erwin Nieuwlaar

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  and associated documentation files (the "Software"), to deal in the Software without
  restriction, including without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from scipy import ndimage as ndi

from .mesh import vertex_normals


def _nanmedian_filter(d, size=5):
    H, W = d.shape
    r = size // 2
    pad = np.pad(d, r, constant_values=np.nan)
    st = np.stack([pad[i:i + H, j:j + W] for i in range(size) for j in range(size)], 0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(st, axis=0)


def _normconv(d, sigma):
    m = np.isfinite(d).astype(np.float32)
    v = np.where(np.isfinite(d), d, 0).astype(np.float32)
    num = ndi.gaussian_filter(v, sigma)
    den = ndi.gaussian_filter(m, sigma)
    out = np.full(d.shape, np.nan, np.float32)
    ok = den > 1e-3
    out[ok] = num[ok] / den[ok]
    return out


def snap_to_prediction(pred, Q, valid, max_off=15, jump=6.0, sigma=1.0):
    """Snap grid vertices Q (H, W, 3; crop coordinates, z y x) onto a binary surface prediction
    `pred` (a crop on the same grid; > 0 = sheet). Returns (snapped Q, offset per vertex (nan
    where unchanged), info)."""
    N, ok = vertex_normals(Q, valid)
    K = int(max_off)
    offs = np.arange(-K, K + 1, dtype=np.float32)
    vy, vx = np.nonzero(ok)
    S = Q[vy, vx][:, None, :] + offs[None, :, None] * N[vy, vx][:, None, :]       # (V, 2K+1, 3)
    pi = np.rint(S).astype(np.int64)
    inb = np.all((pi >= 0) & (pi < np.array(pred.shape)), axis=-1)
    pc = np.clip(pi, 0, np.array(pred.shape) - 1)
    prof = (pred[pc[..., 0], pc[..., 1], pc[..., 2]] > 0) & inb
    nv = len(vy)
    d = np.full(nv, np.nan, np.float32)
    padded = np.concatenate([np.zeros((nv, 1), bool), prof, np.zeros((nv, 1), bool)], 1).astype(np.int8)
    dif = np.diff(padded, axis=1)
    has = prof.any(1)
    for i in np.flatnonzero(has):
        st, en = np.flatnonzero(dif[i] == 1), np.flatnonzero(dif[i] == -1)      # runs [st, en)
        centres = (st + en - 1) / 2.0 - K
        j = int(np.argmin(np.abs(centres)))
        sm = np.convolve(prof[i].astype(np.float32), [1, 2, 1], mode="same")
        c = min(max(int(round(centres[j] + K)), 1), len(offs) - 2)
        y0, y1, y2 = sm[c - 1], sm[c], sm[c + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
        d[i] = centres[j] if abs(frac) > 0.5 else 0.5 * (centres[j] + (c - K + frac))
    D = np.full(valid.shape, np.nan, np.float32)
    D[vy, vx] = d
    med = _nanmedian_filter(D, 5)
    rej = np.isfinite(D) & np.isfinite(med) & (np.abs(D - med) > jump)
    D2 = D.copy()
    D2[rej] = np.nan
    filled = _normconv(D2, max(sigma, 1.0))
    D3 = np.where(np.isfinite(D2), D2, filled)
    D3 = np.clip(_normconv(D3, sigma), -K, K)
    apply = ok & np.isfinite(D3) & np.isfinite(filled)
    out = Q.astype(np.float32).copy()
    out[apply] += D3[apply][:, None] * N[apply]
    off = np.where(apply, D3, np.nan).astype(np.float32)
    info = dict(frac_with_sheet=float(has.mean()) if nv else 0.0,
                frac_snapped=float(apply.sum() / max(1, valid.sum())),
                frac_rejected=float(rej.sum() / max(1, has.sum())),
                median_abs_offset=float(np.nanmedian(np.abs(off[apply]))) if apply.any() else float("nan"))
    return out, off, info
