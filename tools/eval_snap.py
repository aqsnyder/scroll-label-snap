"""Evaluate the snapper against the existing alpha-composite refiner on real cubes.

1. Drift test: push each cube's labels off the surface with a known smooth random drift
   (real CT, real label geometry), run each method on the drifted labels, and measure
   how far the result ends up from where the labels were before drifting. Also measure
   consistency: does correcting drifted labels land where correcting the clean labels lands?
2. Edge alignment on the clean labels: intensity step across the label (papyrus side
   minus gap side) for original, snapped and baseline labels (the refiner with its default
   thresholds, and with thresholds from the cube's own levels and its median offset removed).

The refiner works to its own convention, so it is also scored with its median offset on the clean
labels removed: with VC3D's default thresholds, with thresholds from each cube's own gap and
papyrus levels (Otsu class means), and with each pair in a grid (summarize_eval.py reports the
pair that is best over all evaluated cubes, i.e. the refiner's most generous setting).

Three methods with no graph cut are scored the same way, and warped the same way as the snapper:
the best candidate of the snapper's own score per sample, and the strongest plain CT gradient along
the normal, raw and after a light median. Trusted points are picked with the snapper's own edge
score (the strongest edge is already at the label), so these rows show how much of a drift-test
result any polarity-aware edge search gets. They say nothing about surface smoothness.

Cube k gets drift seed k here and in eval_folder.py. Each cube's result records its seed.

Usage:
  python tools/eval_snap.py OUT_JSON IMAGE_TIF LABEL_TIF [IMAGE_TIF LABEL_TIF ...]
"""
import json
import os
import sys
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import (Params, prepare, snap, alphacomp_shifts,  # noqa: E402
                                    warp_labels, sample, unary_costs, unary_scores)


def drift_field(pts, amp, corr, rng, n_modes=64):
    """Smooth random scalar field (std ~amp, correlation length ~corr voxels)."""
    k = rng.normal(size=(n_modes, 3))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    k *= (2 * np.pi / corr) * rng.uniform(0.6, 1.4, size=(n_modes, 1))
    phi = rng.uniform(0, 2 * np.pi, n_modes)
    f = np.sqrt(2.0 / n_modes) * np.cos(pts @ k.T + phi).sum(1)
    return np.clip(amp * f, -3.5, 3.5).astype(np.float32)


def band_position(lab, pts, n, tmax=5.0):
    """Where the label-1 band is along each normal (NaN if no band nearby)."""
    s = ndi.gaussian_filter((lab == 1).astype(np.float32), 1.5)
    ts = np.arange(-tmax, tmax + 1e-6, 0.25, dtype=np.float32)
    sv = np.stack([sample(s, pts + t * n) for t in ts], 1)
    pos = ts[np.argmax(sv, 1)].astype(np.float32)
    pos[sv.max(1) < 0.15] = np.nan
    return pos


def baseline(img, lab, prep):
    _, pts, n, _, _, _ = prep
    return warp_labels(lab, pts, n, alphacomp_shifts(img, pts, n))


# Opacity thresholds (of 255) tried for the refiner; VC3D's defaults are 118 and 165.
BASELINE_GRID = [(lo, hi) for lo in (50, 70, 90, 118) for hi in (110, 140, 165, 200) if hi >= lo + 25]


def otsu_class_means(img):
    """Mean intensity below and above the Otsu threshold (gap and papyrus levels), ignoring 0."""
    v = img[img > 0].astype(np.float64)
    h, e = np.histogram(v, 256, (0, 256))
    c = (e[:-1] + e[1:]) / 2
    wb = np.cumsum(h)
    wf = wb[-1] - wb
    mb = np.cumsum(h * c) / np.maximum(wb, 1)
    mf = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(wf, 1)
    k = int(np.argmax(wb * wf * (mb - mf) ** 2))
    return float(mb[k]), float(mf[k])


def step_alignment(ct_e, pts, n, q):
    """CT step across a label at band positions q along the normals: papyrus side (2 voxels
    toward -n) minus gap side (2 voxels toward +n). ct_e is the CT blurred with sigma 0.5."""
    ok = np.isfinite(q)
    c = pts[ok] + q[ok, None] * n[ok]
    step = sample(ct_e, c - 2 * n[ok]) - sample(ct_e, c + 2 * n[ok])
    return dict(mean_step=float(np.mean(step)), frac_positive=float(np.mean(step > 0)),
                coverage=float(ok.mean()))


def stats(err):
    err = err[np.isfinite(err)]
    return dict(mean=float(np.mean(err)), median=float(np.median(err)),
                within_1=float(np.mean(err <= 1.0)), over_2=float(np.mean(err > 2.0)),
                n=int(len(err)))


def trusted_samples(lab, prep, p):
    """Samples whose original label sits on a strong, unambiguous papyrus edge (the
    strongest edge within +-4 voxels is within 0.5 voxel of the label and clearly beats
    any edge 2+ voxels away). There the original label can be used as ground truth."""
    ts, E, mov, _, _ = unary_costs(lab.shape, prep, p)
    peak = E.max(1)
    amax = np.argmax(E, 1)
    far = np.abs(ts[None, :] - ts[amax][:, None]) >= 2.0
    second = np.where(far, E, -np.inf).max(1)
    return mov & (np.abs(ts[amax]) <= 0.5) & (peak >= 0.8) & (second < 0.5 * peak)


def median_on_sheet(s, pts, comp, r=6.0, k=13):
    """Median of a shift field over up to k nearest samples within r voxels on the same sheet."""
    if len(pts) < 2:                        # a query for k <= 1 returns another shape
        return np.asarray(s, np.float32)
    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)), distance_upper_bound=r)
    ok = idx < len(pts)
    ic = np.where(ok, idx, 0)
    v = s[ic].astype(np.float32)
    v[~(ok & (comp[ic] == comp[:, None]))] = np.nan
    return np.nanmedian(v, 1).astype(np.float32)


def per_sample_shifts(shape, prep, p, h=1.0):
    """Shift per sample with no graph cut. 'argmax': the best candidate of the snapper's own score.
    'gradient': the strongest plain step ct(t - h) - ct(t + h) along the normal, with the labels'
    offset from that edge taken from the labels given (median position over the stronger half of
    the samples, as the snapper estimates its own). Samples at the border stay put, as for the snapper."""
    ct_s, pts, n, _, _, comp = prep
    ts, U, _, _, movable, _, _ = unary_scores(shape, prep, p)
    if not movable.any():
        return {m: np.zeros(len(pts), np.float32) for m in ("argmax", "gradient", "gradient median")}
    grid = np.arange(-1.5 * p.t_max - h, 1.5 * p.t_max + h + 1e-6, 0.5, dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + g * n) for g in grid], 1)
    kh = int(round(h / 0.5))
    es, g = grid[kh:-kh], prof[:, :-2 * kh] - prof[:, 2 * kh:]
    win = np.abs(es) <= p.t_max
    peak = g[movable][:, win].max(1)
    strong = peak > np.percentile(peak, 50)
    off = np.median(es[win][np.argmax(g[movable][:, win], 1)][strong]) if strong.any() else 0.0
    ix = np.clip(np.rint((ts + np.round(off * 2) / 2 - es[0]) / 0.5).astype(int), 0, len(es) - 1)
    t_grad = ts[np.argmax(g[:, ix], 1)]
    out = {"argmax": ts[np.argmax(U, 1)], "gradient": t_grad, "gradient median": median_on_sheet(t_grad, pts, comp)}
    return {m: np.where(movable, s, 0).astype(np.float32) for m, s in out.items()}


def evaluate(img_path, lab_path, p, seed):
    t0 = time.time()
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    rng = np.random.default_rng(seed)

    prep0 = prepare(img, lab, p)
    ct_s, pts0, n0 = prep0[0], prep0[1], prep0[2]
    lo, hi = 10, np.array(lab.shape) - 11
    inner = np.all((pts0 >= lo) & (pts0 <= hi), axis=1)

    drift = drift_field(pts0, 1.5, 40.0, rng)
    lab_d = warp_labels(lab, pts0, n0, drift)
    prep_d = prepare(img, lab_d, p)

    versions = {
        "original": lab,
        "drifted": lab_d,
        "snap(original)": snap(img, lab, p, prepared=prep0).labels,
        "snap(drifted)": snap(img, lab_d, p, prepared=prep_d).labels,
        "baseline(original)": baseline(img, lab, prep0),
        "baseline(drifted)": baseline(img, lab_d, prep_d),
    }
    # fairer versions of the refiner: its own convention offset removed (the median shift it
    # applies to the clean labels), with thresholds from this cube's own gap and papyrus levels,
    # and with each threshold pair of a grid (the best pair over all cubes is picked afterwards)
    pts_d, n_d = prep_d[1], prep_d[2]
    ac_vol = ndi.gaussian_filter(img.astype(np.float32) / 255.0, 1.4)

    def refiner_corrected(low, high, also_clean=False):
        s0 = alphacomp_shifts(img, pts0, n0, low=low, high=high, vol=ac_vol)
        m0 = float(np.median(s0))
        sd = alphacomp_shifts(img, pts_d, n_d, low=low, high=high, vol=ac_vol)
        clean = warp_labels(lab, pts0, n0, s0 - m0) if also_clean else None
        return warp_labels(lab_d, pts_d, n_d, sd - m0), m0, clean

    versions["baseline offset-corrected(drifted)"], off_default, _ = refiner_corrected(118 / 255, 165 / 255)
    gap_level, pap_level = otsu_class_means(img)
    (versions["baseline adaptive(drifted)"], off_adaptive,
     versions["baseline adaptive(original)"]) = refiner_corrected(gap_level / 255, pap_level / 255, also_clean=True)
    # no graph cut: best candidate per sample, warped the same way as the snapper's shifts
    for tag, pr, lb in (("original", prep0, lab), ("drifted", prep_d, lab_d)):
        for m, s in per_sample_shifts(lab.shape, pr, p).items():
            versions[f"{m}({tag})"] = warp_labels(lb, pr[1], pr[2], s, comp=pr[5], smooth=p.smooth_warp, keep_topology=p.keep_topology)
    pos = {k: band_position(v, pts0[inner], n0[inner]) for k, v in versions.items()}
    o = pos["original"]
    tr = trusted_samples(lab, prep0, p)[inner]
    grid = {}
    for lo, hi in BASELINE_GRID:
        lab_g, m0, _ = refiner_corrected(lo / 255, hi / 255)
        e = np.abs(band_position(lab_g, pts0[inner], n0[inner]) - o)[tr]
        grid[f"{lo}-{hi}"] = dict(offset_corrected=stats(e), median_offset=m0)
    drift_res = {
        # accuracy where the original label is trustworthy ground truth
        "drifted (no correction)": stats(np.abs(pos["drifted"] - o)[tr]),
        "snapper": stats(np.abs(pos["snap(drifted)"] - o)[tr]),
        "baseline": stats(np.abs(pos["baseline(drifted)"] - o)[tr]),
        "baseline, offset-corrected": stats(np.abs(pos["baseline offset-corrected(drifted)"] - o)[tr]),
        "baseline, adaptive thresholds, offset-corrected": stats(np.abs(pos["baseline adaptive(drifted)"] - o)[tr]),
        "argmax": stats(np.abs(pos["argmax(drifted)"] - o)[tr]),
        "gradient": stats(np.abs(pos["gradient(drifted)"] - o)[tr]),
        "gradient median": stats(np.abs(pos["gradient median(drifted)"] - o)[tr]),
        # consistency everywhere: does correcting drifted labels land where correcting
        # the clean labels lands?
        "snapper consistency": stats(np.abs(pos["snap(drifted)"] - pos["snap(original)"])),
        "baseline consistency": stats(np.abs(pos["baseline(drifted)"] - pos["baseline(original)"])),
        "baseline, adaptive thresholds consistency": stats(np.abs(pos["baseline adaptive(drifted)"]
                                                                   - pos["baseline adaptive(original)"])),
        "argmax consistency": stats(np.abs(pos["argmax(drifted)"] - pos["argmax(original)"])),
        "gradient consistency": stats(np.abs(pos["gradient(drifted)"] - pos["gradient(original)"])),
        "gradient median consistency": stats(np.abs(pos["gradient median(drifted)"]
                                                     - pos["gradient median(original)"])),
    }
    moved = pos["snap(original)"] - o
    clean_moves = dict(frac_moved_1vox=float(np.nanmean(np.abs(moved) >= 1.0)),
                       median_signed=float(np.nanmedian(moved)))

    # edge alignment on the clean labels: CT step across the label along the normal
    ct_e = ndi.gaussian_filter(img.astype(np.float32), 0.5)
    pi, ni = pts0[inner], n0[inner]
    align = {k: step_alignment(ct_e, pi, ni, pos[k])
             for k in ("original", "snap(original)", "baseline(original)", "baseline adaptive(original)",
                       "argmax(original)", "gradient median(original)")}
    name = os.path.basename(img_path).rsplit(".", 1)[0]
    return dict(cube=name, seed=seed, seconds=round(time.time() - t0, 1), n_points=int(inner.sum()),
                n_trusted=int(tr.sum()), drift_test=drift_res, edge_alignment=align,
                clean_label_moves=clean_moves, baseline_grid=grid,
                baseline_offsets=dict(default=off_default, adaptive=off_adaptive,
                                      adaptive_thresholds=[gap_level, pap_level]))


def main(out_json, *paths):
    if len(paths) % 2:
        sys.exit(f"image and label TIFFs come in pairs, and {len(paths)} paths were given")
    if any("*" in x or "?" in x for x in paths):
        sys.exit("wildcards are not expanded here: give each TIFF")
    p = Params()
    results = []
    for k, (img_path, lab_path) in enumerate(zip(paths[0::2], paths[1::2])):
        r = evaluate(img_path, lab_path, p, seed=k)
        results.append(r)
        d = r["drift_test"]
        a = r["edge_alignment"]
        print(f"{r['cube']} ({r['n_points']} pts, {r['n_trusted']} trusted, {r['seconds']} s)", flush=True)
        for m in ("drifted (no correction)", "snapper", "argmax", "gradient median", "baseline",
                  "snapper consistency", "argmax consistency", "baseline consistency"):
            print(f"   {m:26s} mean {d[m]['mean']:.2f}  median {d[m]['median']:.2f}  "
                  f"within 1 vox {d[m]['within_1']:.0%}  over 2 vox {d[m]['over_2']:.0%}")
        for m, v in a.items():
            print(f"   step across {m:20s} {v['mean_step']:6.1f}  positive {v['frac_positive']:.0%}")
        cm = r["clean_label_moves"]
        print(f"   clean labels moved >=1 vox {cm['frac_moved_1vox']:.0%}, median signed shift "
              f"{cm['median_signed']:+.2f}", flush=True)
        with open(out_json, "w") as f:
            json.dump(dict(params=p.asdict(), cubes=results), f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
