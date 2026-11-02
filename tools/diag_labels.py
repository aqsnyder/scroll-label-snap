"""Diagnostic: how far do recto-surface labels sit from the nearest papyrus edge?

For sampled label-1 points: estimate the sheet normal from the label band (Hessian of the
smoothed mask), centre the point on the band, orient the normal from papyrus toward the gap,
sample the CT profile along it, and locate the steepest bright->dark edge.

Usage: python3 -I diag_labels.py OUT_PREFIX IMAGE_TIF LABEL_TIF [IMAGE_TIF LABEL_TIF ...]
"""
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SAMPLES = 20000
T = np.arange(-8.0, 8.01, 0.5)        # profile offsets (voxels) along the normal
SEARCH = np.abs(T) <= 6.0             # where the edge is searched


def sample(vol, pts):
    """Trilinear sample of vol at float points (N,3) in z,y,x order."""
    return ndi.map_coordinates(vol, pts.T, order=1, mode="nearest")


def analyse(img_path, lab_path, rng):
    img = tifffile.imread(img_path).astype(np.float32)
    lab = tifffile.imread(lab_path)
    m = (lab == 1).astype(np.float32)
    ct = ndi.gaussian_filter(img, 1.0)               # light denoise for profiles
    s = ndi.gaussian_filter(m, 1.5)                  # smoothed label band

    # Hessian of the smoothed band -> sheet normal (eigvec of most negative eigenvalue)
    orders = {(0, 0): (2, 0, 0), (1, 1): (0, 2, 0), (2, 2): (0, 0, 2),
              (0, 1): (1, 1, 0), (0, 2): (1, 0, 1), (1, 2): (0, 1, 1)}
    hess = {k: ndi.gaussian_filter(m, 1.5, order=o) for k, o in orders.items()}

    zz, yy, xx = np.nonzero(lab == 1)
    margin = 12
    ok = ((zz >= margin) & (zz < lab.shape[0] - margin) & (yy >= margin) &
          (yy < lab.shape[1] - margin) & (xx >= margin) & (xx < lab.shape[2] - margin))
    idx = rng.choice(np.flatnonzero(ok), size=min(N_SAMPLES, ok.sum()), replace=False)
    p = np.stack([zz[idx], yy[idx], xx[idx]], 1).astype(np.float32)

    H = np.empty((len(p), 3, 3), np.float32)
    for (a, b), vol in hess.items():
        v = vol[zz[idx], yy[idx], xx[idx]]
        H[:, a, b] = v
        H[:, b, a] = v
    w, V = np.linalg.eigh(H)
    n = V[:, :, 0]                                   # most negative eigenvalue
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    # centre each point on the band (max of smoothed mask along n within +-3)
    tc = np.arange(-3, 3.01, 0.25)
    sv = np.stack([sample(s, p + t * n) for t in tc], 1)
    p = p + tc[np.argmax(sv, 1)][:, None] * n

    # orient n from papyrus (bright) toward gap (dark)
    side_pos = sample(ct, p + 3 * n)
    side_neg = sample(ct, p - 3 * n)
    flip = side_pos > side_neg
    n[flip] *= -1
    contrast = np.abs(side_pos - side_neg)

    prof = np.stack([sample(ct, p + t * n) for t in T], 1)     # (N, len(T))
    d = np.gradient(prof, T, axis=1)
    d_search = np.where(SEARCH[None, :], d, np.inf)
    t_edge = T[np.argmin(d_search, 1)]
    slope = -np.min(d_search, 1)
    return dict(t_edge=t_edge, slope=slope, contrast=contrast, prof=prof)


def main(out_prefix, *paths):
    rng = np.random.default_rng(0)
    pairs = list(zip(paths[0::2], paths[1::2]))
    results = []
    for img_path, lab_path in pairs:
        r = analyse(img_path, lab_path, rng)
        name = img_path.rsplit("/", 1)[-1].replace(".tif", "")
        good = r["contrast"] > 20            # points with a clear papyrus/gap difference
        te = r["t_edge"][good]
        med = np.median(te)
        dev = np.abs(te - med)
        print(f"{name}: {len(r['t_edge'])} pts, clear-edge {good.mean():.0%}; edge offset median "
              f"{med:+.1f} vox; |offset-median| <=1: {np.mean(dev <= 1):.0%}, <=2: {np.mean(dev <= 2):.0%}, "
              f">3: {np.mean(dev > 3):.0%}; mean |offset|: {np.mean(np.abs(te)):.2f}")
        results.append((name, r, good))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, r, good in results:
        axes[0].hist(r["t_edge"][good], bins=np.arange(-6.25, 6.3, 0.5), histtype="step", label=name)
        prof = r["prof"][good]
        axes[1].plot(T, prof.mean(0), label=f"{name}: aligned at label")
        # same profiles re-aligned at the detected edge (what perfect snapping would give)
        shifted = np.array([np.interp(T, T - te, pr) for pr, te in zip(prof, r["t_edge"][good])])
        axes[1].plot(T, shifted.mean(0), "--", label=f"{name}: aligned at edge")
    axes[0].set_xlabel("detected edge position relative to label (voxels)")
    axes[0].set_ylabel("points")
    axes[0].legend()
    axes[0].set_title("Where the papyrus edge is, relative to the label")
    axes[1].set_xlabel("offset along normal, papyrus -> gap (voxels)")
    axes[1].set_ylabel("mean CT intensity")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Mean profile: at label vs at detected edge")
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=70)
    print("wrote", out_prefix + ".png")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
