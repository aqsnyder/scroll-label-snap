"""Fibre-direction check for snapped meshes: is the surface on a recto face?

The recto's fibres run horizontally (around the scroll) and the verso's run vertically (along the
scroll's axis). A papyrus sheet is two fibre layers, recto layer toward the scroll centre. So
reading the CT through a surface that sits on a recto face, the papyrus just inside it should
show fibres running across the scroll axis, turning toward fibres along the axis deeper in. A
surface that wanders between layers blurs that pattern; one on a verso face reverses it.

Usage:
  python tools/fibre_check.py OUT_JSON PATCH_DIR [PATCH_DIR ...] --snapped SNAP_DIR
         [--control SNAP_DIR] [--png OUT_PNG] [--show PATCH_NAME]

  SNAP_DIR: output of tools/snap_tifxyz.py for these patches. --control: a second run to compare,
  e.g. one snapped onto the opposite faces (snap_tifxyz.py --normals with the other sign).

Per mesh, at depths of 1 to 6 voxels into the papyrus from the face (the mesh's own convention
offset along the papyrus -> gap normal), the CT is read through the surface at ~1 voxel spacing.
Texture direction is measured with a structure tensor (gradient sigma 1, window sigma 5), in the
surface, relative to the scroll axis (taken as the scan's z axis), over papyrus pixels away from
gaps (Otsu threshold, mask eroded 4 px). Anisotropy A = (Ea - Ew) / (Ea + Ew), where Ea and Ew
are the gradient energies along the axis and around the scroll: A > 0 means fibres run across the
axis (recto-like), A < 0 along it (verso-like). The summary per mesh is A near the face (1-2
voxels in) minus A deeper (5-6 voxels in): positive for the recto signature.
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import sample  # noqa: E402
from scroll_label_snap.mesh import densify, mean_spacing, vertex_normals  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402

DEPTHS = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]


def otsu(x, bins=256):
    h, e = np.histogram(x, bins)
    c = (e[:-1] + e[1:]) / 2
    wb = np.cumsum(h)
    wf = wb[-1] - wb
    mb = np.cumsum(h * c) / np.maximum(wb, 1)
    mf = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(wf, 1)
    return float(c[np.argmax(wb * wf * (mb - mf) ** 2)])


def surface(P, valid, step=1.0):
    """The mesh resampled at ~step voxels: positions, unit normals, valid mask, grid tangents."""
    N, ok = vertex_normals(P, valid)
    f = max(1, int(round(mean_spacing(P, ok) / step)))
    Pd, Nd, vd = densify(P, N, ok, f)
    return Pd, Nd, vd, np.gradient(Pd, axis=0), np.gradient(Pd, axis=1)


def _image_coords(u, Tr, Tc):
    """(a, b) per pixel with a * Tr + b * Tc = u (least squares)."""
    g11, g12, g22 = (Tr * Tr).sum(-1), (Tr * Tc).sum(-1), (Tc * Tc).sum(-1)
    b1, b2 = (Tr * u).sum(-1), (Tc * u).sum(-1)
    det = g11 * g22 - g12 ** 2
    det = np.where(np.abs(det) < 1e-9, np.nan, det)
    return (g22 * b1 - g12 * b2) / det, (g11 * b2 - g12 * b1) / det


def anisotropy(img, weight, N, Tr, Tc, sigma_d=1.0, sigma_w=5.0):
    """Per-pixel A = (Ea - Ew) / (Ea + Ew): gradient energy along the scroll axis (z) minus around
    the scroll, within the surface. Returns (A, ok)."""
    fill = np.where(np.isfinite(img), img, np.nanmedian(img))
    gr = ndi.gaussian_filter(fill, sigma_d, order=(1, 0))
    gc = ndi.gaussian_filter(fill, sigma_d, order=(0, 1))
    ez = np.zeros_like(N)
    ez[..., 0] = 1
    zt = ez - (ez * N).sum(-1, keepdims=True) * N            # scroll axis projected into the surface
    zn = np.linalg.norm(zt, axis=-1, keepdims=True)
    ua = zt / np.maximum(zn, 1e-6)
    uw = np.cross(N, ua)                                     # around the scroll
    a1, b1 = _image_coords(ua, Tr, Tc)
    a2, b2 = _image_coords(uw, Tr, Tc)
    da = np.nan_to_num(a1 * gr + b1 * gc)
    dw = np.nan_to_num(a2 * gr + b2 * gc)
    ok = (zn[..., 0] > 0.5) & np.isfinite(a1) & np.isfinite(a2)
    w = weight * ok
    norm = ndi.gaussian_filter(w, sigma_w) + 1e-6
    ea = ndi.gaussian_filter(da ** 2 * w, sigma_w) / norm
    ew = ndi.gaussian_filter(dw ** 2 * w, sigma_w) / norm
    return (ea - ew) / (ea + ew + 1e-6), ok


def profile(ct_s, thr, P, valid, offset, c_off, sign, erode=4, keep_images=False):
    """A and papyrus fraction at each depth into the papyrus from the mesh's face."""
    Pd, Nd, vd, Tr, Tc = surface(P - offset, valid)
    Nr = Nd * sign
    rows, images = [], []
    for d in DEPTHS:
        img = np.full(vd.shape, np.nan, np.float32)
        img[vd] = sample(ct_s, Pd[vd] + (c_off + d) * Nr[vd])
        pap = (np.nan_to_num(img) > thr) & vd
        core = ndi.binary_erosion(pap, iterations=erode) if erode else pap
        A, ok = anisotropy(img, core.astype(np.float32), Nr, Tr, Tc)
        m = core & ok
        rows.append(dict(depth=d, A=float(A[m].mean()) if m.sum() > 500 else float("nan"),
                         papyrus=float(pap[vd].mean()), n=int(m.sum())))
        if keep_images:
            images.append(img)
    return rows, images


def summary(rows):
    a = np.array([r["A"] for r in rows])
    return float(np.nanmean(a[:2]) - np.nanmean(a[4:6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--snapped", required=True)
    ap.add_argument("--control")
    ap.add_argument("--png")
    ap.add_argument("--show", help="patch whose CT layers the figure shows (default: the first)")
    a = ap.parse_args()
    runs = [("snapped", a.snapped)] + ([("control", a.control)] if a.control else [])
    out, keep = [], None
    for d in a.patches:
        name = os.path.basename(os.path.normpath(d))
        off = np.asarray(json.load(open(os.path.join(d, "patch.json")))["offset_zyx"], np.float32)
        ct_s = ndi.gaussian_filter(tifffile.imread(os.path.join(d, "volume.tif")).astype(np.float32), 0.5)
        thr = otsu(ct_s)
        info = {k: json.load(open(os.path.join(r, name, "snap.json"))) for k, r in runs}
        P0, v0, _ = read_tifxyz(os.path.join(d, "segment.tifxyz"))
        want = keep is None and (a.show is None or a.show == name)
        res = {"original": profile(ct_s, thr, P0, v0, off, info["snapped"]["convention_offset"],
                                   info["snapped"]["normal_sign"], keep_images=want)}
        for k, r in runs:
            P, v, _ = read_tifxyz(os.path.join(r, name, "snapped_dense.tifxyz"))
            res[k] = profile(ct_s, thr, P, v, off, info[k]["convention_offset"], info[k]["normal_sign"],
                             keep_images=want)
        if want:
            keep = (name, {k: v[1] for k, v in res.items()})
        rec = dict(patch=name, otsu=thr, **{k: dict(profile=v[0], near_minus_deep=summary(v[0])) for k, v in res.items()})
        out.append(rec)
        print(f"{name}: " + " | ".join(f"{k} near-minus-deep {rec[k]['near_minus_deep']:+.3f}" for k in res), flush=True)
    kinds = [k for k in ("original", "snapped", "control") if k in out[0]]
    pooled = {}
    for k in kinds:
        nd = np.array([r[k]["near_minus_deep"] for r in out])
        A = np.array([[x["A"] for x in r[k]["profile"]] for r in out])
        pap = np.array([[x["papyrus"] for x in r[k]["profile"]] for r in out])
        pooled[k] = dict(near_minus_deep_mean=float(nd.mean()), positive=int((nd > 0).sum()), patches=len(nd),
                         A_by_depth=dict(zip(map(str, DEPTHS), A.mean(0).tolist())),
                         papyrus_by_depth=dict(zip(map(str, DEPTHS), pap.mean(0).tolist())))
        print(f"{k:9s}: near-minus-deep {nd.mean():+.3f}, positive in {(nd > 0).sum()}/{len(nd)}; "
              f"papyrus 1-2 voxels in {pap[:, :2].mean():.0%}")
    json.dump(dict(depths=DEPTHS, pooled=pooled, patches=out), open(a.out_json, "w"), indent=1)

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colours = dict(original="#c0392b", snapped="#1e8449", control="#7f8c8d")
        labels = dict(original="original mesh", snapped="snapped (recto faces)", control="control: snapped onto the opposite faces")
        fig = plt.figure(figsize=(14, 7.6))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
        ax = fig.add_subplot(gs[0, :2])
        for k in kinds:
            A = np.array([[x["A"] for x in r[k]["profile"]] for r in out])
            for row in A:
                ax.plot(-np.array(DEPTHS), row, "-", color=colours[k], alpha=0.18, lw=1)
            ax.plot(-np.array(DEPTHS), A.mean(0), "-o", color=colours[k], lw=2.4, label=labels[k])
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("depth into the papyrus from the face (voxels)")
        ax.set_ylabel("fibre anisotropy A\n(+ across the scroll axis = recto-like)")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title(f"Fibre direction vs depth, {len(out)} patches (thin lines: single patches)", fontsize=10)
        ax2 = fig.add_subplot(gs[0, 2:])
        x = np.arange(len(out))
        wbar = 0.8 / len(kinds)
        for i, k in enumerate(kinds):
            ax2.bar(x + (i - (len(kinds) - 1) / 2) * wbar, [r[k]["near_minus_deep"] for r in out], wbar,
                    color=colours[k], label=labels[k])
        ax2.axhline(0, color="#999", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r["patch"].replace("patch_", "p") for r in out], fontsize=8)
        ax2.set_ylabel("A at 1-2 voxels minus A at 5-6 voxels")
        ax2.set_title("Recto signature per patch (positive = recto-like near the face)", fontsize=10)
        name, imgs = keep
        show = [("original", 1), ("snapped", 1), ("original", 5), ("snapped", 5)]   # depths -2 and -6
        allv = np.concatenate([imgs[k][i][np.isfinite(imgs[k][i])] for k, i in show])
        lo, hi = np.percentile(allv, [1, 99])
        for j, (k, i) in enumerate(show):
            axi = fig.add_subplot(gs[1, j])
            img = imgs[k][i]
            h, w = img.shape
            axi.imshow(img[h // 4: h // 4 + 200, w // 4: w // 4 + 200], cmap="gray", vmin=lo, vmax=hi,
                       interpolation="nearest")
            axi.set_title(f"{name}, {k}, {-DEPTHS[i]:.0f} voxels in", fontsize=9)
            axi.set_xticks([])
            axi.set_yticks([])
        fig.text(0.5, 0.005, "Bottom: 200 x 200 voxels of the CT read through each mesh; the scroll axis is vertical, "
                 "so recto fibres run left-right and verso fibres up-down.", ha="center", fontsize=9)
        plt.tight_layout(rect=(0, 0.02, 1, 1))
        plt.savefig(a.png, dpi=90)


if __name__ == "__main__":
    main()
