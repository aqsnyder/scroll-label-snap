"""Ink labels drawn on a finer scan's render of a segment, carried onto the segment's pixel grid on a
coarser scan (using tools/grid_correspondence.py), and ink maps scored against them.

  python tools/fine_labels.py roi OUT_JSON --corr CORR_JSON --supervision TIF [--margin 128]
      The coarse pixels covering the labelled area (supervision mask), with a margin.

  python tools/fine_labels.py transfer OUT_NPZ --corr CORR_JSON --labels TIF --supervision TIF --roi R0 R1 C0 C1
      For each coarse pixel of the roi, the share of its footprint on the fine grid that is
      supervised, and the share of the supervised part that is ink. A pixel is supervised when
      at least half its footprint is, and ink when at least half of its supervised part is.

  python tools/fine_labels.py score OUT_JSON --labels-npz NPZ --map NAME=TIF [...] --renders ZARR [...]
                                  [--reference NAME] [--blocks 128] [--boot 1000] [--save-draws]
      ROC AUC (ties count half) and the best F1 of each map over the supervised pixels valid in all
      renders, and for each map its AUC difference from the reference map with a 95% interval from
      a paired bootstrap over blocks of pixels. --save-draws also writes the bootstrap draws, which
      tools/smooth_snap_readings.py pools over targets.

Pixels are 1/scale per vertex on both grids (20 for scale 0.05), with pixel (r, c) at vertex (r/k, c/k).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ink_scores import auc_f1, load_map  # noqa: E402
from ink_plan import occupancy  # noqa: E402


def corr_params(path):
    c = json.load(open(path))
    return c["fr"], c["orr"], c["fc"], c["oc"], int(c.get("coarse_k", 20)), int(c.get("fine_k", 20))


def coarse_to_fine(r, c, p):
    fr, orr, fc, oc, kc, kf = p
    return kf * (fr * r / kc + orr), kf * (fc * c / kc + oc)


def fine_to_coarse(R, C, p):
    fr, orr, fc, oc, kc, kf = p
    return (R / kf - orr) * kc / fr, (C / kf - oc) * kc / fc


def integral(a):
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64 if a.size >= 2 ** 31 else np.int32)
    S[1:, 1:] = a.astype(S.dtype).cumsum(0).cumsum(1)
    return S


def box_mean(S, y0, y1, x0, x1):
    H, W = S.shape[0] - 1, S.shape[1] - 1
    y0, y1 = np.clip(y0, 0, H), np.clip(y1, 0, H)
    x0, x1 = np.clip(x0, 0, W), np.clip(x1, 0, W)
    area = np.maximum((y1 - y0) * (x1 - x0), 1)
    return (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]) / area


def footprint_shares(lab, sup, roi, p):
    """Supervised share and ink share (of the supervised part) of each coarse pixel's footprint."""
    import tifffile  # noqa: F401  (inputs are read by the caller)
    r0, r1, c0, c1 = roi
    fr, _, fc, _, kc, kf = p
    hr, hc = kf * fr / kc / 2, kf * fc / kc / 2
    rr, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float64), np.arange(c0, c1, dtype=np.float64), indexing="ij")
    R, C = coarse_to_fine(rr, cc, p)
    y0, y1 = np.floor(R - hr + 0.5).astype(int), np.floor(R + hr + 0.5).astype(int)
    x0, x1 = np.floor(C - hc + 0.5).astype(int), np.floor(C + hc + 0.5).astype(int)
    # only the part of the fine grid under the roi
    Y0, Y1 = max(0, int(y0.min())), min(sup.shape[0], int(y1.max()) + 1)
    X0, X1 = max(0, int(x0.min())), min(sup.shape[1], int(x1.max()) + 1)
    s = sup[Y0:Y1, X0:X1] > 0
    Ssup, Sink = integral(s), integral((lab[Y0:Y1, X0:X1] > 0) & s)
    y0, y1, x0, x1 = y0 - Y0, y1 - Y0, x0 - X0, x1 - X0
    sup_share = box_mean(Ssup, y0, y1, x0, x1)
    ink_part = box_mean(Sink, y0, y1, x0, x1)
    ink_share = np.where(sup_share > 0, ink_part / np.maximum(sup_share, 1e-12), 0.0)
    return sup_share.astype(np.float32), ink_share.astype(np.float32)


def cmd_roi(a):
    import tifffile
    p = corr_params(a.corr)
    sup = tifffile.imread(a.supervision) > 0
    ys, xs = np.nonzero(sup)
    if ys.size == 0:
        sys.exit("the supervision mask is empty")
    rr, cc = fine_to_coarse(np.array([ys.min(), ys.max() + 1], float), np.array([xs.min(), xs.max() + 1], float), p)
    roi = [int(np.floor(rr[0])) - a.margin, int(np.ceil(rr[1])) + a.margin,
           int(np.floor(cc[0])) - a.margin, int(np.ceil(cc[1])) + a.margin]
    c = json.load(open(a.corr))
    H, W = ((np.array(c["coarse_grid"]) - 1) * int(c.get("coarse_k", 20)) + 1).tolist()
    roi = [max(0, roi[0]), min(H, roi[1]), max(0, roi[2]), min(W, roi[3])]
    json.dump(dict(roi=roi, corr=a.corr, supervision=a.supervision, margin=a.margin), open(a.out_json, "w"), indent=1)
    print(f"roi rows {roi[0]}-{roi[1]}, cols {roi[2]}-{roi[3]}")


def cmd_transfer(a):
    import tifffile
    p = corr_params(a.corr)
    lab, sup = tifffile.imread(a.labels), tifffile.imread(a.supervision)
    sup_share, ink_share = footprint_shares(lab, sup, a.roi, p)
    supervised = sup_share >= 0.5
    ink = supervised & (ink_share >= 0.5)
    np.savez_compressed(a.out_npz, supervised=supervised, ink=ink, sup_share=sup_share, ink_share=ink_share,
                        roi=np.array(a.roi))
    print(f"{supervised.sum():,} supervised pixels in the roi, {ink.sum() / max(supervised.sum(), 1):.1%} ink")


def bootstrap_auc_diffs(scores_a, scores_b, truth, blocks, n_boot, seed=0):
    """AUC(b) - AUC(a) over each resample of blocks (paired: the same pixels for both). A resample
    with no ink or no background pixel is skipped."""
    ub = np.unique(blocks)
    idx_of = {b: np.flatnonzero(blocks == b) for b in ub}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        sel = np.concatenate([idx_of[b] for b in pick])
        t = truth[sel]
        if t.all() or not t.any():
            continue
        diffs.append(auc_f1(scores_b[sel], t)["auc"] - auc_f1(scores_a[sel], t)["auc"])
    return diffs


def block_bootstrap_auc_diff(scores_a, scores_b, truth, blocks, n_boot, seed=0, keep_draws=False):
    """95% interval of AUC(b) - AUC(a) over resampled blocks (paired: the same pixels for both)."""
    diffs = bootstrap_auc_diffs(scores_a, scores_b, truth, blocks, n_boot, seed)
    if not diffs:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = dict(lo=float(lo), hi=float(hi), resamples=len(diffs))
    if keep_draws:
        out["draws"] = [float(d) for d in diffs]
    return out


def cmd_score(a):
    L = np.load(a.labels_npz)
    supervised, ink = L["supervised"], L["ink"]
    maps = dict(m.split("=", 1) for m in a.map)
    loaded = {k: load_map(v) for k, v in maps.items()}
    for k, m in loaded.items():
        if m.shape != supervised.shape:
            sys.exit(f"map {k} is {m.shape}, the labels' roi is {supervised.shape}")
    use = supervised.copy()
    for r in a.renders:
        occ, _ = occupancy(r)
        if occ.shape != use.shape:
            sys.exit(f"render {r} is {occ.shape}, the labels' roi is {use.shape}")
        use &= occ
    for m in loaded.values():
        use &= m > 0
    truth = ink[use]
    H, W = use.shape
    yy, xx = np.nonzero(use)
    blocks = (yy // a.blocks) * (W // a.blocks + 1) + xx // a.blocks
    res = dict(labels=a.labels_npz, pixels=int(use.sum()), ink_fraction=float(truth.mean()) if truth.size else None,
               blocks=int(len(np.unique(blocks))) if truth.size else 0, block_size=a.blocks, scores={})
    for k, m in loaded.items():
        res["scores"][k] = dict(path=maps[k], **auc_f1(m[use], truth))
    ref = a.reference or next(iter(maps))
    res["reference"] = ref
    for k, m in loaded.items():
        if k == ref or res["scores"][k]["auc"] is None:
            continue
        d = dict(auc=res["scores"][k]["auc"] - res["scores"][ref]["auc"],
                 best_f1=res["scores"][k]["best_f1"] - res["scores"][ref]["best_f1"])
        d["auc_95"] = block_bootstrap_auc_diff(loaded[ref][use], m[use], truth, blocks, a.boot, keep_draws=a.save_draws)
        res["scores"][k]["minus_reference"] = d
    json.dump(res, open(a.out_json, "w"), indent=1)
    for k, sc in res["scores"].items():
        extra = ""
        if "minus_reference" in sc:
            d = sc["minus_reference"]
            ci = d["auc_95"]
            extra = f"; vs {ref}: AUC {d['auc']:+.4f}" + (f" ({ci['lo']:+.4f} to {ci['hi']:+.4f})" if ci else "")
        if sc["auc"] is not None:
            print(f"{k}: AUC {sc['auc']:.4f}, best F1 {sc['best_f1']:.4f}{extra}")
    print(f"over {res['pixels']:,} supervised pixels ({res['ink_fraction']:.1%} ink) in {res['blocks']} blocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roi")
    p.add_argument("out_json")
    p.add_argument("--corr", required=True)
    p.add_argument("--supervision", required=True)
    p.add_argument("--margin", type=int, default=128)
    p = sub.add_parser("transfer")
    p.add_argument("out_npz")
    p.add_argument("--corr", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--supervision", required=True)
    p.add_argument("--roi", type=int, nargs=4, required=True)
    p = sub.add_parser("score")
    p.add_argument("out_json")
    p.add_argument("--labels-npz", required=True)
    p.add_argument("--map", action="append", required=True, metavar="NAME=TIF")
    p.add_argument("--renders", nargs="+", required=True)
    p.add_argument("--reference")
    p.add_argument("--blocks", type=int, default=128)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--save-draws", action="store_true",
                   help="also write each difference's bootstrap draws (for pooling over targets)")
    a = ap.parse_args()
    {"roi": cmd_roi, "transfer": cmd_transfer, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
