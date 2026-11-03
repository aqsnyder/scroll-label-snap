"""Scores of the pre-registered ink test (results/ink_test_plan.md), from the ink maps the official
inference wrote (uint8 TIFFs, probability x 255, truncated).

  python tools/ink_scores.py labels OUT_JSON --map NAME=TIF [--map ...] --renders R_ORIG R_SNAP \\
                                  --labels LABELS_ZARR --mask MASK_ZARR --check CHECK_JSON [--official TIF]
      Secondary outcome (w035): each map against the ink labels, within the supervision mask and
      where both renders are valid: ROC AUC and the best F1 over thresholds. Our maps are moved
      onto the labels' grid with the in-plane shift from the renderer check (CHECK_JSON); the
      labels are on the official surface volume's grid. --official adds the map from the official
      surface volume, scored on the same pixels.

  python tools/ink_scores.py sanity OUT_JSON --ours TIF --official TIF --renders R_ORIG --check CHECK_JSON
      Sanity check (w035): the pixel correlation of our original map and the official one.

  python tools/ink_scores.py tiles OUT_DIR --original TIF --snapped TIF --renders R_ORIG R_SNAP \\
                                 --dense SNAPPED_DENSE_TIFXYZ --snap-json SNAP_SEGMENT_JSON [--roi R0 R1 C0 C1] \\
                                 [--name w052] [--tile 256] [--min-valid 0.5]
      Descriptive: in 256 x 256-pixel tiles, the change from original to snapped in the mean ink
      probability and in its 99th percentile, against the tile's mean |snap shift| and its share of
      flagged (reversed-face or search-limit) samples, with Spearman correlations; the
      distribution of changes; a map of the change; and the 6 tiles with the largest drop and the
      6 with the largest gain in the 99th percentile. --roi: the maps cover only this part of the
      renders (the run's --roi).

Valid pixels: where both renders have at least one non-zero layer (amendment 1).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from ink_plan import occupancy  # noqa: E402


def load_map(tif):
    import tifffile
    return np.asarray(tifffile.imread(tif))


def valid_both(renders, roi=None, shape=None):
    v = None
    for r in renders:
        occ, _ = occupancy(r)
        v = occ if v is None else (v & occ)
    if roi is not None:
        v = v[roi[0]:roi[1], roi[2]:roi[3]]
    if shape is not None and v.shape != tuple(shape):
        raise ValueError(f"the renders' valid area is {v.shape}, the ink map is {tuple(shape)}")
    return v


def open_2d(path):
    """A label or mask zarr as a 2D boolean array (an array or group "0"; a layer axis is collapsed)."""
    import zarr
    root = zarr.open(path, mode="r")
    arr = root if hasattr(root, "shape") else root["0"]
    a = np.squeeze(np.asarray(arr[:]))
    if a.ndim == 3:
        a = (a > 0).any(int(np.argmin(a.shape)))
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image, got shape {a.shape}")
    return a > 0


def overlap(shape_a, shape_b, dy, dx):
    """Slices (sa, sb) with b[sb] lined up on a[sa], where b[y + dy, x + dx] matches a[y, x]."""
    ya0, xa0 = max(0, -dy), max(0, -dx)
    ya1, xa1 = min(shape_a[0], shape_b[0] - dy), min(shape_a[1], shape_b[1] - dx)
    if ya1 <= ya0 or xa1 <= xa0:
        raise ValueError("the two grids don't overlap at this shift")
    return (slice(ya0, ya1), slice(xa0, xa1)), (slice(ya0 + dy, ya1 + dy), slice(xa0 + dx, xa1 + dx))


def auc_f1(score_u8, truth):
    """ROC AUC (ties count half) and the best F1 over thresholds 'score >= t', from 256-bin counts."""
    s = np.asarray(score_u8).astype(np.int64).ravel()
    t = np.asarray(truth, bool).ravel()
    pos = np.bincount(s[t], minlength=256).astype(np.float64)
    neg = np.bincount(s[~t], minlength=256).astype(np.float64)
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return dict(auc=None, best_f1=None, threshold=None, positives=int(n1), negatives=int(n0))
    below = np.cumsum(neg) - neg
    auc = float((pos * (below + 0.5 * neg)).sum() / (n1 * n0))
    tp = np.cumsum(pos[::-1])[::-1]                     # predicted ink: score >= threshold
    fp = np.cumsum(neg[::-1])[::-1]
    f1 = 2 * tp / np.maximum(2 * tp + fp + (n1 - tp), 1)
    k = int(np.argmax(f1))
    return dict(auc=auc, best_f1=float(f1[k]), threshold=k, threshold_probability=round(k / 255, 4),
                positives=int(n1), negatives=int(n0))


def shift_of(check_json):
    c = json.load(open(check_json))
    dy, dx = (int(v) for v in c["inplane_shift_yx"])
    return dy, dx


def cmd_labels(a):
    maps = dict(m.split("=", 1) for m in a.map)
    if a.official:
        maps["official"] = a.official
    lab, mask = open_2d(a.labels), open_2d(a.mask)
    if lab.shape != mask.shape:
        raise ValueError(f"labels {lab.shape} and mask {mask.shape} differ")
    dy, dx = shift_of(a.check)
    loaded = {k: load_map(v) for k, v in maps.items() if k != "official"}
    shape = next(iter(loaded.values())).shape
    valid = valid_both(a.renders, shape=shape)
    so, sl = overlap(shape, lab.shape, dy, dx)
    use = mask[sl] & valid[so]
    if a.official:
        off = load_map(a.official)
        if off.shape != lab.shape:
            raise ValueError(f"the official map {off.shape} isn't on the labels' grid {lab.shape}")
        use &= off[sl] > 0
    truth = lab[sl][use]
    res = dict(labels=a.labels, mask=a.mask, labels_shape=list(lab.shape), map_shape=list(shape),
               inplane_shift_yx=[dy, dx], pixels=int(use.sum()), ink_fraction=float(truth.mean()) if truth.size else None,
               scores={})
    for name, path in maps.items():
        m = off if name == "official" else loaded[name]
        region = sl if name == "official" else so
        res["scores"][name] = dict(path=path, **auc_f1(m[region][use], truth))
    if "original" in res["scores"] and "snapped" in res["scores"]:
        o, s = res["scores"]["original"], res["scores"]["snapped"]
        if o["auc"] is not None and s["auc"] is not None:
            res["snapped_minus_original"] = dict(auc=s["auc"] - o["auc"], best_f1=s["best_f1"] - o["best_f1"])
    json.dump(res, open(a.out_json, "w"), indent=1)
    for name, sc in res["scores"].items():
        if sc["auc"] is not None:
            print(f"{name}: AUC {sc['auc']:.4f}, best F1 {sc['best_f1']:.4f} (at p >= {sc['threshold_probability']:.3f})")
    if "snapped_minus_original" in res:
        d = res["snapped_minus_original"]
        print(f"snapped - original: AUC {d['auc']:+.4f}, best F1 {d['best_f1']:+.4f} over {res['pixels']:,} pixels")


def cmd_sanity(a):
    ours, off = load_map(a.ours), load_map(a.official)
    dy, dx = shift_of(a.check)
    valid = valid_both(a.renders, shape=ours.shape)
    so, sf = overlap(ours.shape, off.shape, dy, dx)
    use = valid[so] & (off[sf] > 0)
    x = ours[so][use].astype(np.float64)
    y = off[sf][use].astype(np.float64)
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else None
    res = dict(ours=a.ours, official=a.official, inplane_shift_yx=[dy, dx], pixels=int(use.sum()), pearson_r=r,
               mean_abs_difference=float(np.abs(x - y).mean() / 255) if x.size else None)
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"our original map vs the official one: r = {r:.4f} over {res['pixels']:,} pixels, "
          f"mean |difference| {res['mean_abs_difference']:.4f}")


def display(p_u8):
    """The model card's display scaling, (p - 0.25) / 0.5 clipped to 0..1 (amendment 1)."""
    return np.clip((p_u8.astype(np.float32) / 255.0 - 0.25) / 0.5, 0.0, 1.0)


def tile_rows(p_o, p_s, valid, tile, min_valid):
    rows = []
    H, W = valid.shape
    for i0 in range(0, H, tile):
        for j0 in range(0, W, tile):
            i1, j1 = min(H, i0 + tile), min(W, j0 + tile)
            v = valid[i0:i1, j0:j1]
            frac = float(v.mean())
            if frac < min_valid or v.sum() < 100:
                continue
            o = p_o[i0:i1, j0:j1][v].astype(np.float64) / 255
            s = p_s[i0:i1, j0:j1][v].astype(np.float64) / 255
            mo, ms = float(o.mean()), float(s.mean())
            qo, qs = float(np.percentile(o, 99)), float(np.percentile(s, 99))
            rows.append(dict(i0=i0, i1=i1, j0=j0, j1=j1, valid=round(frac, 4), mean_original=mo, mean_snapped=ms,
                             p99_original=qo, p99_snapped=qs, d_mean=ms - mo, d_p99=qs - qo))
    return rows


def add_snap_stats(rows, dense_dir, factor, k, r0v, c0v, roi):
    """Mean |shift| and share of flagged samples of the dense snapped mesh under each tile."""
    import tifffile
    sys.path.insert(0, os.path.join(HERE, ".."))
    from scroll_label_snap.tifxyz import read_tifxyz
    _, vd, _ = read_tifxyz(dense_dir)
    shift = np.abs(tifffile.imread(os.path.join(dense_dir, "snap_shift.tif")).astype(np.float32))
    flag = tifffile.imread(os.path.join(dense_dir, "snap_flag.tif")) > 0
    limit = tifffile.imread(os.path.join(dense_dir, "snap_limit.tif")) > 0
    bad = flag | limit
    Hd, Wd = vd.shape
    for r in rows:
        y0, y1 = r["i0"] + roi[0], r["i1"] + roi[0]       # render pixels
        x0, x1 = r["j0"] + roi[2], r["j1"] + roi[2]
        a0 = int(np.floor((r0v + y0 / k) * factor))
        a1 = int(np.ceil((r0v + (y1 - 1) / k) * factor)) + 1
        b0 = int(np.floor((c0v + x0 / k) * factor))
        b1 = int(np.ceil((c0v + (x1 - 1) / k) * factor)) + 1
        a0, a1, b0, b1 = max(0, a0), min(Hd, a1), max(0, b0), min(Wd, b1)
        v = vd[a0:a1, b0:b1]
        if v.any():
            r["mean_abs_shift"] = float(shift[a0:a1, b0:b1][v].mean())
            r["flagged_share"] = float(bad[a0:a1, b0:b1][v].mean())
        else:
            r["mean_abs_shift"] = r["flagged_share"] = None


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(rho=None, p=None, n=int(ok.sum()))
    rho, p = spearmanr(x[ok], y[ok])
    return dict(rho=float(rho), p=float(p), n=int(ok.sum()))


INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#c3c2b7"
BLUE, RED, MID = "#184f95", "#a8312f", "#f0efec"


def figures(rows, p_o, p_s, valid, out_dir, name, summary, tile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    d99 = np.array([r["d_p99"] for r in rows])
    dmean = np.array([r["d_mean"] for r in rows])

    # distribution of the changes (two small multiples, one measure each)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), dpi=150)
    for ax, vals, label in ((axs[0], d99, "Change in 99th percentile"), (axs[1], dmean, "Change in mean")):
        lim = max(0.02, float(np.percentile(np.abs(vals), 99.5)) if vals.size else 0.02)
        bins = np.linspace(-lim, lim, 41)
        ax.hist(np.clip(vals, -lim, lim), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_xlabel(f"{label} (snapped - original, probability)")
        ax.set_ylabel("Tiles")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"{name}: change in ink probability per 256-pixel tile ({len(rows)} tiles)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_tile_changes.png"))
    plt.close(fig)

    # map of the change in the 99th percentile
    H, W = valid.shape
    grid = np.full((-(-H // tile), -(-W // tile)), np.nan)
    for r in rows:
        grid[r["i0"] // tile, r["j0"] // tile] = r["d_p99"]
    lim = max(0.02, float(np.nanpercentile(np.abs(grid), 99)) if np.isfinite(grid).any() else 0.02)
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(8, 8 * grid.shape[0] / max(grid.shape[1], 1) + 0.8), dpi=150)
    im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Change in 99th percentile of ink probability (snapped - original)")
    cb.outline.set_visible(False)
    ax.set_title(f"{name}: where snapping raises (blue) or lowers (red) the ink signal; white: no data",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_change_map.png"))
    plt.close(fig)

    # the 6 largest drops and the 6 largest gains, original | snapped
    order = np.argsort(d99)
    drops, gains = list(order[:6]), list(order[::-1][:6])
    n = max(len(drops), 1)
    fig, axs = plt.subplots(n, 4, figsize=(8, 2.1 * n + 0.6), dpi=150, squeeze=False)
    for row in range(n):
        for col, (sel, kind) in enumerate(((drops, "drop"), (gains, "gain"))):
            for m, (img, lab) in enumerate(((p_o, "original"), (p_s, "snapped"))):
                ax = axs[row, 2 * col + m]
                ax.set_xticks([])
                ax.set_yticks([])
                if row >= len(sel):
                    ax.axis("off")
                    continue
                r = rows[sel[row]]
                crop = display(img[r["i0"]:r["i1"], r["j0"]:r["j1"]])
                crop[~valid[r["i0"]:r["i1"], r["j0"]:r["j1"]]] = 0
                ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                if m == 0:
                    ax.set_ylabel(f"{kind} {row + 1}\n{r['d_p99']:+.3f}", fontsize=8, color=INK2)
                if row == 0:
                    ax.set_title(f"{'Largest drops' if col == 0 else 'Largest gains'}: {lab}", fontsize=8, color=INK2)
    fig.suptitle(f"{name}: tiles where snapping changed the ink signal most (change in 99th percentile)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_extreme_tiles.png"))
    plt.close(fig)
    summary["extremes"] = dict(drops=[rows[i] for i in drops], gains=[rows[i] for i in gains])


def cmd_tiles(a):
    os.makedirs(a.out_dir, exist_ok=True)
    p_o, p_s = load_map(a.original), load_map(a.snapped)
    if p_o.shape != p_s.shape:
        raise ValueError(f"the maps differ in shape: {p_o.shape} vs {p_s.shape}")
    roi = a.roi or [0, None, 0, None]
    import zarr
    attrs = dict(zarr.open_array(a.renders[0], mode="r").attrs)
    full = valid_both(a.renders)
    roi = [roi[0], roi[1] if roi[1] is not None else full.shape[0], roi[2], roi[3] if roi[3] is not None else full.shape[1]]
    valid = full[roi[0]:roi[1], roi[2]:roi[3]]
    if valid.shape != p_o.shape:
        raise ValueError(f"the maps are {p_o.shape}, the renders' area {valid.shape}; give the run's --roi")
    rows = tile_rows(p_o, p_s, valid, a.tile, a.min_valid)
    factor = int(json.load(open(a.snap_json))["factor"])
    k = int(attrs.get("pixels_per_vertex", 1))
    r0v, c0v = int((attrs.get("rows") or [0])[0]), int((attrs.get("cols") or [0])[0])
    add_snap_stats(rows, a.dense, factor, k, r0v, c0v, roi)
    col = lambda key: [r[key] if r.get(key) is not None else np.nan for r in rows]  # noqa: E731
    d99, dm = np.array(col("d_p99")), np.array(col("d_mean"))
    summary = dict(name=a.name, original=a.original, snapped=a.snapped, roi=roi, tile=a.tile, min_valid=a.min_valid,
                   tiles=len(rows),
                   d_mean_percentiles={str(q): float(np.percentile(dm, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                   d_p99_percentiles={str(q): float(np.percentile(d99, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                   share_tiles_d_p99_below_minus_0_02=float(np.mean(d99 < -0.02)),
                   share_tiles_d_p99_above_0_02=float(np.mean(d99 > 0.02)),
                   whole_map=dict(mean_original=float(p_o[valid].mean() / 255), mean_snapped=float(p_s[valid].mean() / 255),
                                  p99_original=float(np.percentile(p_o[valid], 99) / 255),
                                  p99_snapped=float(np.percentile(p_s[valid], 99) / 255)),
                   spearman=dict(d_mean_vs_abs_shift=spearman(dm, col("mean_abs_shift")),
                                 d_mean_vs_flagged=spearman(dm, col("flagged_share")),
                                 d_p99_vs_abs_shift=spearman(d99, col("mean_abs_shift")),
                                 d_p99_vs_flagged=spearman(d99, col("flagged_share"))))
    figures(rows, p_o, p_s, valid, a.out_dir, a.name, summary, a.tile)
    json.dump(dict(summary, rows=rows), open(os.path.join(a.out_dir, f"{a.name}_tiles.json"), "w"), indent=1)
    s = summary["spearman"]
    print(f"{a.name}: {len(rows)} tiles; change in 99th percentile: median {summary['d_p99_percentiles']['50']:+.4f} "
          f"(5-95%: {summary['d_p99_percentiles']['5']:+.4f} to {summary['d_p99_percentiles']['95']:+.4f}); "
          f"Spearman vs |shift| {s['d_p99_vs_abs_shift']['rho']}, vs flagged {s['d_p99_vs_flagged']['rho']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("out_json")
    p.add_argument("--map", action="append", required=True, metavar="NAME=TIF")
    p.add_argument("--renders", nargs="+", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--official")
    p = sub.add_parser("sanity")
    p.add_argument("out_json")
    p.add_argument("--ours", required=True)
    p.add_argument("--official", required=True)
    p.add_argument("--renders", nargs="+", required=True)
    p.add_argument("--check", required=True)
    p = sub.add_parser("tiles")
    p.add_argument("out_dir")
    p.add_argument("--original", required=True)
    p.add_argument("--snapped", required=True)
    p.add_argument("--renders", nargs=2, required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--snap-json", required=True)
    p.add_argument("--roi", type=int, nargs=4)
    p.add_argument("--name", default="segment")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    {"labels": cmd_labels, "sanity": cmd_sanity, "tiles": cmd_tiles}[a.cmd](a)


if __name__ == "__main__":
    main()
