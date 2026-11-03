"""Before/after renders of mesh snapping: the CT just inside and just outside the recto face, as
seen through the original mesh and through the snapped one.

Usage:
  python tools/render_mesh.py OUT_PNG SNAP_DIR PATCH_DIR [PATCH_DIR ...] [--depths -1.5 1.5] [--sections]

  SNAP_DIR is the OUT_DIR of tools/snap_tifxyz.py, holding <patch name>/snapped_dense.tifxyz and
  snap.json for each patch.

Each mesh is resampled at about 1 voxel spacing and the CT (lightly smoothed) is read at
`depth` voxels from where the mesh puts the recto face (its own convention offset along the
papyrus -> gap normal): negative depths are inside the papyrus, positive ones in the gap. A
mesh that sits on the face shows papyrus everywhere just inside and gap everywhere just outside.
With --sections, each patch also gets a CT cross-section with both meshes traced on it.
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import densify, mean_spacing, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402


def layers(ct_s, P, valid, sign, depths, step=1.0):
    """CT at the given offsets along the (oriented) normal, on the mesh resampled to ~step voxels."""
    N, ok = vertex_normals(P, valid)
    f = max(1, int(round(mean_spacing(P, ok) / step)))
    Pd, Nd, vd = densify(P, N, ok, f)
    out = []
    for d in depths:
        img = np.full(vd.shape, np.nan, np.float32)
        img[vd] = sample(ct_s, Pd[vd] + d * sign * Nd[vd])
        out.append(img)
    return out


def section(ax, ct, P0, v0, P1, v1, sign, title, width=130, height=44):
    """A CT slice through the middle of the patch, across the sheet, with both meshes traced."""
    N0, ok0 = vertex_normals(P0, v0)
    a, n = P0[ok0], N0[ok0]
    ax_n = int(np.argmax(np.abs(n.mean(0))))
    cut = [q for q in range(3) if q != ax_n][0]
    rest = [q for q in range(3) if q != cut]
    f0 = max(1, int(round(mean_spacing(P0, ok0))))
    A0, _, va = densify(P0, N0, ok0, f0)
    N1, ok1 = vertex_normals(P1, v1)
    A1, _, vb = densify(P1, N1, ok1, max(1, int(round(mean_spacing(P1, ok1)))))
    a, b = A0[va], A1[vb]
    ci = int(round(np.median(a[:, cut])))
    img = np.take(ct, ci, axis=cut).astype(np.float32)
    x0 = int(np.median(a[:, rest[1]]) - width / 2)
    sel = lambda q: (np.abs(q[:, cut] - ci) < 0.5) & (q[:, rest[1]] >= x0) & (q[:, rest[1]] < x0 + width)  # noqa: E731
    sa, sb = sel(a), sel(b)
    y0 = int(max(0, np.median(a[sa, rest[0]]) - height / 2)) if sa.any() else 0
    sl = img[y0:y0 + height, max(x0, 0):x0 + width]
    ax.imshow(sl, cmap="gray", interpolation="nearest",
              extent=(max(x0, 0) - 0.5, max(x0, 0) + sl.shape[1] - 0.5, y0 + sl.shape[0] - 0.5, y0 - 0.5))
    for q, s_, col, lab in ((a, sa, "red", "original"), (b, sb, "lime", "snapped")):
        o = np.argsort(q[s_, rest[1]])
        ax.plot(q[s_, rest[1]][o], q[s_, rest[0]][o], "-", color=col, lw=1.2, label=lab)
    m = np.flatnonzero(sa)
    if len(m):
        m = m[len(m) // 2]
        k = np.argmin(np.linalg.norm(P0[ok0] - a[m], axis=1))
        nn = n[k] * sign
        ax.annotate("", xy=(a[m, rest[1]] + 8 * nn[rest[1]], a[m, rest[0]] + 8 * nn[rest[0]]),
                    xytext=(a[m, rest[1]], a[m, rest[0]]), arrowprops=dict(color="cyan", arrowstyle="->", lw=1.5))
    ax.set_title(f"{title}: slice {'zyx'[cut]} = {ci} (crop); cyan = papyrus -> gap at the recto face", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_png")
    ap.add_argument("snap_dir")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--depths", type=float, nargs="+", default=[-1.5, 1.5])
    ap.add_argument("--sections", action="store_true", help="also draw a cross-section per patch")
    a = ap.parse_args()
    nd = len(a.depths)
    rows = len(a.patches) * (2 if a.sections else 1)
    fig = plt.figure(figsize=(3.3 * 2 * nd, (3.6 + (2.2 if a.sections else 0)) * len(a.patches)))
    gs = fig.add_gridspec(rows, 2 * nd, height_ratios=([3.6, 2.2] if a.sections else [3.6]) * len(a.patches))
    for pi, d in enumerate(a.patches):
        r0 = pi * (2 if a.sections else 1)
        row = [fig.add_subplot(gs[r0, k]) for k in range(2 * nd)]
        name = os.path.basename(os.path.normpath(d))
        off = np.asarray(json.load(open(os.path.join(d, "patch.json")))["offset_zyx"], np.float32)
        info = json.load(open(os.path.join(a.snap_dir, name, "snap.json")))
        c, sign = info["convention_offset"], info["normal_sign"]
        ct_s = ndi.gaussian_filter(tifffile.imread(os.path.join(d, "volume.tif")).astype(np.float32), 0.5)
        P0, v0, _ = read_tifxyz(os.path.join(d, "segment.tifxyz"))
        P1, v1, _ = read_tifxyz(os.path.join(a.snap_dir, name, "snapped_dense.tifxyz"))
        depths = [c + x for x in a.depths]
        imgs = layers(ct_s, P0 - off, v0, sign, depths) + layers(ct_s, P1 - off, v1, sign, depths)
        fin = np.concatenate([i[np.isfinite(i)] for i in imgs])
        lo, hi = np.percentile(fin, [1, 99])
        titles = [f"original, {x:+.1f}" for x in a.depths] + [f"snapped, {x:+.1f}" for x in a.depths]
        order = [0, nd] if nd == 1 else list(range(2 * nd))
        for ax, k in zip(row, order):
            ax.imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(f"{name}: {titles[k]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        if a.sections:
            ct = tifffile.imread(os.path.join(d, "volume.tif"))
            section(fig.add_subplot(gs[r0 + 1, :]), ct, P0 - off, v0, P1 - off, v1, sign, name)
    fig.suptitle("CT through each mesh at the given depth from its recto face (voxels; - = inside the papyrus, "
                 "+ = toward the gap)", fontsize=10)
    plt.tight_layout()
    plt.savefig(a.out_png, dpi=90)


if __name__ == "__main__":
    main()
