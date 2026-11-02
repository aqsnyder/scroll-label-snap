"""Inspect one image/label cube pair: shapes, dtypes, label values, and slice overlays.

Usage: python3 -I inspect_cube.py IMAGE_TIF LABEL_TIF OUT_PNG
"""
import sys

import numpy as np
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(img_path, lab_path, out_png):
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    print(f"image: shape={img.shape} dtype={img.dtype} min={img.min()} max={img.max()} "
          f"mean={img.mean():.1f} p1/p50/p99={np.percentile(img, [1, 50, 99])}")
    vals, counts = np.unique(lab, return_counts=True)
    print(f"label: shape={lab.shape} dtype={lab.dtype} values="
          + ", ".join(f"{v}:{c} ({100 * c / lab.size:.2f}%)" for v, c in zip(vals, counts)))
    for v in vals:
        if v == 0:
            continue
        m = lab == v
        print(f"  value {v}: mean image intensity {img[m].mean():.1f} vs overall {img.mean():.1f}")

    # three orthogonal center slices with label overlay
    n = img.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cuts = [("z", lambda a: a[n // 2]), ("y", lambda a: a[:, a.shape[1] // 2]), ("x", lambda a: a[:, :, a.shape[2] // 2])]
    for col, (name, cut) in enumerate(cuts):
        im, lb = cut(img), cut(lab)
        axes[0, col].imshow(im, cmap="gray")
        axes[0, col].set_title(f"{name}-slice (CT)")
        axes[1, col].imshow(im, cmap="gray")
        ov = np.ma.masked_where(lb == 0, lb)
        axes[1, col].imshow(ov, cmap="autumn", alpha=0.6, interpolation="nearest")
        axes[1, col].set_title(f"{name}-slice + label (values {np.unique(lb).tolist()})")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=70)
    print("wrote", out_png)


if __name__ == "__main__":
    main(*sys.argv[1:4])
