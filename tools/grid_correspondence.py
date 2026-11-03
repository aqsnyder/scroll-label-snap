"""Find how one segment's tifxyz grid on a fine scan maps onto its grid on a coarse scan of the same
scroll, so labels drawn on the fine scan's render can be carried to the coarse one.

Usage:
  python tools/grid_correspondence.py OUT_JSON COARSE_TIFXYZ FINE_TIFXYZ [--samples 20000]

For PHerc. 0841, the open data has each labelled segment twice: on the 9.366 um scan and on the
2.403 um scan, where its ink labels are drawn (on the render grid, 20 pixels per vertex). The two
grids sample the same surface at different vertex steps. This fits
    fine row = fr * coarse row + orr,   fine column = fc * coarse column + oc   (vertex units)
by matching the fine grid, sampled bilinearly at those positions, to the coarse grid through one
global affine map between the scans, and keeps the parameters with the smallest median residual.
Reports the residuals (in coarse-scan voxels) and the affine map.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402


def bilinear(P, v, a, b):
    i0 = np.clip(np.floor(a).astype(int), 0, P.shape[0] - 2)
    j0 = np.clip(np.floor(b).astype(int), 0, P.shape[1] - 2)
    ok = (a >= 0) & (b >= 0) & (a <= P.shape[0] - 1) & (b <= P.shape[1] - 1)
    ok &= v[i0, j0] & v[i0 + 1, j0] & v[i0, j0 + 1] & v[i0 + 1, j0 + 1]
    fa, fb = (a - i0)[:, None], (b - j0)[:, None]
    X = (P[i0, j0] * (1 - fa) * (1 - fb) + P[i0 + 1, j0] * fa * (1 - fb)
         + P[i0, j0 + 1] * (1 - fa) * fb + P[i0 + 1, j0 + 1] * fa * fb)
    return X, ok


def residuals(Pc, Pf, vf, I, J, fr, orr, fc, oc):
    Xf, ok = bilinear(Pf, vf, I * fr + orr, J * fc + oc)
    if ok.sum() < 100:
        return np.inf, np.inf, None, 0
    A = np.c_[Xf[ok], np.ones(ok.sum())]
    Xc = Pc[I[ok].astype(int), J[ok].astype(int)]
    coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    r = np.linalg.norm(A @ coef - Xc, axis=1)
    return float(np.median(r)), float(np.percentile(r, 95)), coef, int(ok.sum())


def fit(Pc, vc, Pf, vf, samples=20000, seed=0):
    I, J = np.nonzero(vc)
    sel = np.random.default_rng(seed).choice(len(I), min(samples, len(I)), replace=False)
    I, J = I[sel].astype(np.float64), J[sel].astype(np.float64)
    f0r, f0c = Pf.shape[0] / Pc.shape[0], Pf.shape[1] / Pc.shape[1]
    # coarse search with one factor for both axes, then each axis on its own, finer
    best = None
    for f, o in itertools.product(np.linspace(min(f0r, f0c) - 0.02, max(f0r, f0c) + 0.02, 41), np.arange(-1.0, 4.01, 0.25)):
        m = residuals(Pc, Pf, vf, I, J, f, o, f, o)[0]
        if best is None or m < best[0]:
            best = (m, f, o)
    fr = fc = best[1]
    orr = oc = best[2]
    for _ in range(2):
        for axis in (0, 1):
            cand = itertools.product(np.arange((fr if axis == 0 else fc) - 0.005, (fr if axis == 0 else fc) + 0.0051, 0.0005),
                                     np.arange((orr if axis == 0 else oc) - 0.3, (orr if axis == 0 else oc) + 0.301, 0.05))
            scores = []
            for f, o in cand:
                args = (f, o, fc, oc) if axis == 0 else (fr, orr, f, o)
                scores.append((residuals(Pc, Pf, vf, I, J, *args)[0], f, o))
            m, f, o = min(scores)
            if axis == 0:
                fr, orr = f, o
            else:
                fc, oc = f, o
    med, p95, coef, n = residuals(Pc, Pf, vf, I, J, fr, orr, fc, oc)
    return dict(fr=float(fr), orr=float(orr), fc=float(fc), oc=float(oc), median_residual=med, p95_residual=p95,
                pairs=n, affine_coarse_from_fine=np.round(coef, 6).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_json")
    ap.add_argument("coarse")
    ap.add_argument("fine")
    ap.add_argument("--samples", type=int, default=20000)
    a = ap.parse_args()
    Pc, vc, mc = read_tifxyz(a.coarse)
    Pf, vf, mf = read_tifxyz(a.fine)
    res = fit(Pc.astype(np.float64), vc, Pf.astype(np.float64), vf, a.samples)
    k = lambda m: max(1, int(round(1.0 / float(np.atleast_1d(m.get("scale", [1.0]))[0]))))  # noqa: E731
    res.update(coarse=a.coarse, fine=a.fine, coarse_grid=list(vc.shape), fine_grid=list(vf.shape),
               coarse_k=k(mc), fine_k=k(mf))
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"fine row = {res['fr']:.4f} x coarse row + {res['orr']:.2f}; fine col = {res['fc']:.4f} x coarse col + "
          f"{res['oc']:.2f}; median residual {res['median_residual']:.3f} voxels (95%: {res['p95_residual']:.3f})")
    if res["median_residual"] > 1.0:
        sys.exit("the grids don't correspond closely (median residual over 1 voxel); check the inputs")


if __name__ == "__main__":
    main()
