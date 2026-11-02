"""Close-ups of CT with label bands: CT alone, original label (red), snapped label (cyan).

Picks crops where the snapper moved the label the most, plus random ones.

Usage:
  python tools/render_compare.py OUT_PNG IMAGE_TIF ORIG_LABEL_TIF SNAPPED_LABEL_TIF [axis] [n_big] [n_random] [seed]
"""
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def pick_crops(a, b, n_big, n_random, seed, size=40, margin=24):
    diff = ndi.uniform_filter((a ^ b).astype(np.float32), 7)
    inner = (slice(margin, -margin),) * 3
    work = np.zeros_like(diff)
    work[inner] = diff[inner]
    crops = []
    for _ in range(n_big):
        c = np.unravel_index(np.argmax(work), work.shape)
        crops.append(("largest change", tuple(int(v) for v in c)))
        work[tuple(slice(max(0, v - size), v + size) for v in c)] = 0
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.nonzero(a[inner])
    for k in rng.choice(len(zz), n_random, replace=False):
        crops.append(("random", (int(zz[k]) + margin, int(yy[k]) + margin, int(xx[k]) + margin)))
    return crops


def main(out_png, img_path, orig_path, snap_path, axis="z", n_big=3, n_random=3, seed=1):
    img = tifffile.imread(img_path).astype(np.float32)
    a = tifffile.imread(orig_path) == 1
    b = tifffile.imread(snap_path) == 1
    ax_i = {"z": 0, "y": 1, "x": 2}[axis]
    crops = pick_crops(a, b, int(n_big), int(n_random), int(seed))
    lo, hi = np.percentile(img, [1, 99.5])
    red = ListedColormap([(0, 0, 0, 0), (1, 0.15, 0.15, 0.45)])
    cyan = ListedColormap([(0, 0, 0, 0), (0.1, 0.9, 1, 0.45)])
    h = 20
    fig, axes = plt.subplots(len(crops), 3, figsize=(9, 3.1 * len(crops)))
    for r, (kind, c) in enumerate(crops):
        sl = [slice(max(0, v - h), v + h) for v in c]
        sl[ax_i] = c[ax_i]
        sl = tuple(sl)
        im = np.clip((img[sl] - lo) / (hi - lo), 0, 1)
        for k, (title, mask, cm) in enumerate([("CT", None, None), ("original", a[sl], red),
                                               ("snapped", b[sl], cyan)]):
            ax = axes[r, k]
            ax.imshow(im, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            if mask is not None:
                ax.imshow(mask.astype(int), cmap=cm, interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f"{kind}\n{axis}={c[ax_i]}", fontsize=9)
            if r == 0:
                ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=80)
    print("wrote", out_png)


if __name__ == "__main__":
    main(*sys.argv[1:9])
