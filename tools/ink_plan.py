"""The fixed decisions of the pre-registered ink test (results/ink_test_plan.md), computed the same
way every time, so the run script needs no judgement calls.

  python tools/ink_plan.py render-check CHECK_JSON OUT_JSON [--normals grid --center 0 --step 1]
      From a compare_surface_volumes.py result: does our render follow the official surface
      volume's layer convention (same order, the model's centre layer on the same surface), and is
      the mean correlation at least 0.9? If the order is reversed or the centre is off, it gives the
      --normals and --center to render with instead. verdict: pass, re-render or fail.

  python tools/ink_plan.py crop RENDER_ZARR OUT_JSON [--size 1024] [--middle-rows 96 288]
      The timing crop: the size x size window centred on the median row and median column of the
      render's valid pixels, moved inside the image if needed. Also counts the patches the official
      inference would run (128-pixel patches every 64 pixels that hold any non-zero input) in the
      crop, in the whole render, and in the middle vertex rows.

  python tools/ink_plan.py choose CROP_JSON OUT_JSON --forward TIF LOG --reverse TIF LOG
                                  [--project RENDER_ZARR ...] [--budget-hours 8]
      The layer direction: the one whose ink map has the higher 99th percentile over the crop's
      valid pixels (forward if equal). With --project (the renders of the primary run), also the
      scope: the whole segment if the projected time is within the budget, else the middle rows,
      else stop.

The patch rules copy villa's flat inference (vesuvius/ink_detection/inference/infer.py):
_sliding_positions_1d, center_crop_layer_indices and the raw-empty patch skip.
"""
import argparse
import json
import sys

import numpy as np

PATCH, STRIDE, DEPTH = 128, 64, 17


# --- copies of the official inference's scheduling rules ---------------------------------------
def sliding_positions(length, patch, stride):
    if length <= patch:
        return [0]
    pos = list(range(0, length - patch + 1, max(1, stride)))
    if pos[-1] != length - patch:
        pos.append(length - patch)
    return pos


def center_crop(n_layers, depth):
    idx = np.arange(n_layers)
    if n_layers <= depth:
        return idx
    start = n_layers // 2 - depth // 2
    return idx[start:start + depth]


def count_patches(occ, patch=PATCH, stride=STRIDE, roi=None):
    """Patches the inference forwards: scheduled ones (overlapping roi, if given) with any input."""
    H, W = occ.shape
    S = np.zeros((H + 1, W + 1), np.int64)
    S[1:, 1:] = occ.astype(np.int64).cumsum(0).cumsum(1)
    n = 0
    for y0 in sliding_positions(H, patch, stride):
        vh = min(patch, H - y0)
        if roi is not None and (y0 >= roi[1] or y0 + vh <= roi[0]):
            continue
        for x0 in sliding_positions(W, patch, stride):
            vw = min(patch, W - x0)
            if roi is not None and (x0 >= roi[3] or x0 + vw <= roi[2]):
                continue
            if S[y0 + vh, x0 + vw] - S[y0, x0 + vw] - S[y0 + vh, x0] + S[y0, x0] > 0:
                n += 1
    return n


# --- reading renders ------------------------------------------------------------------------------
def occupancy(render_zarr, layers=None, band=1024):
    """Pixels with any non-zero input, over the given layers (default all), read in row bands."""
    import zarr
    z = zarr.open_array(render_zarr, mode="r")
    L, H, W = z.shape
    sel = np.arange(L) if layers is None else np.asarray(layers)
    occ = np.zeros((H, W), bool)
    for r0 in range(0, H, band):
        blk = np.asarray(z[int(sel.min()):int(sel.max()) + 1, r0:r0 + band])
        occ[r0:r0 + band] = (blk[sel - sel.min()] > 0).any(0)
    return occ, dict(z.attrs)


def model_occupancy(render_zarr):
    import zarr
    L = zarr.open_array(render_zarr, mode="r").shape[0]
    return occupancy(render_zarr, center_crop(L, DEPTH))


# --- render-check ----------------------------------------------------------------------------------
def render_decision(check, normals="grid", center=0.0, step=1.0, min_corr=0.9, tol=0.5):
    best = check["best_layer_alignment"]
    L, L_ref = int(check["ours_shape"][0]), int(check["reference_shape"][0])
    order = 1 if best["order"] == "same" else -1
    off = int(best["ref_layer_of_our_layer_0"])
    c_ref = L_ref // 2                       # the middle of the 17 layers the model reads from it
    i_c = order * (c_ref - off)              # our layer matched to that (our layer i <-> ref off + order * i)
    t_c = center + (i_c - (L - 1) / 2.0) * step   # where that layer sits, along our current normals
    reversed_ = order == -1
    off_centre = abs(t_c - center) > tol
    new_normals = ({"grid": "flipped", "flipped": "grid"}[normals]) if reversed_ else normals
    new_center = float(-t_c if reversed_ else t_c)
    corr = float(best["mean_corr"])
    if reversed_ or off_centre:
        verdict = "re-render"
    else:
        verdict = "pass" if corr >= min_corr else "fail"
    return dict(verdict=verdict, mean_corr=corr, min_corr_required=min_corr, order=best["order"],
                ref_layer_of_our_layer_0=off, our_layers=L, reference_layers=L_ref,
                reference_model_centre_layer=c_ref, our_layer_matching_it=float(i_c),
                its_offset_along_our_normals=float(t_c), current=dict(normals=normals, center=center),
                normals=new_normals if verdict == "re-render" else normals,
                center=new_center if verdict == "re-render" else center,
                inplane_shift_yx=check.get("inplane_shift_yx"))


# --- crop ---------------------------------------------------------------------------------------
def crop_window(occ, size):
    H, W = occ.shape
    rows, cols = np.nonzero(occ)
    if rows.size == 0:
        raise ValueError("the render has no valid pixels")
    cr, cc = int(np.round(np.median(rows))), int(np.round(np.median(cols)))
    h, w = min(size, H), min(size, W)
    r0 = int(np.clip(cr - h // 2, 0, H - h))
    c0 = int(np.clip(cc - w // 2, 0, W - w))
    return [r0, r0 + h, c0, c0 + w], [cr, cc]


def middle_rows_roi(attrs, shape, rows):
    k = int(attrs.get("pixels_per_vertex", 1))
    R0 = int((attrs.get("rows") or [0])[0])
    r0 = max(0, (rows[0] - R0) * k)
    r1 = min(shape[0], (rows[1] - R0) * k + 1)
    return [r0, r1, 0, shape[1]]


def cmd_crop(a):
    occ, attrs = occupancy(a.render)
    mocc, _ = model_occupancy(a.render)
    roi, centre = crop_window(occ, a.size)
    mid = middle_rows_roi(attrs, occ.shape, a.middle_rows)
    r0, r1, c0, c1 = roi
    res = dict(render=a.render, shape=list(occ.shape), valid_fraction=round(float(occ.mean()), 4),
               centre_rule="median row and median column of the valid pixels",
               centre=centre, roi=roi, crop_valid_fraction=round(float(occ[r0:r1, c0:c1].mean()), 4),
               patch=PATCH, stride=STRIDE, depth=DEPTH,
               patches_crop=count_patches(mocc, roi=roi), patches_whole=count_patches(mocc),
               middle_rows=list(a.middle_rows), middle_roi=mid, patches_middle=count_patches(mocc, roi=mid))
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"{a.render}: valid {res['valid_fraction']:.1%}; crop rows {r0}-{r1}, cols {c0}-{c1} "
          f"({res['crop_valid_fraction']:.1%} valid); patches: crop {res['patches_crop']}, "
          f"whole {res['patches_whole']}, middle rows {res['patches_middle']}")


# --- choose -------------------------------------------------------------------------------------
def map_stats(tif, valid):
    import tifffile
    p = tifffile.imread(tif).astype(np.float32) / 255.0
    if p.shape != valid.shape:
        raise ValueError(f"{tif} has shape {p.shape}, the crop is {valid.shape}")
    v = p[valid]
    q = np.percentile(v, [50, 90, 99]) if v.size else [np.nan] * 3
    return dict(p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), mean=float(v.mean()) if v.size else None,
                valid_pixels=int(v.size))


def cmd_choose(a):
    crop = json.load(open(a.crop_json))
    r0, r1, c0, c1 = crop["roi"]
    occ, _ = occupancy(crop["render"])
    valid = occ[r0:r1, c0:c1]
    f = map_stats(a.forward[0], valid)
    r = map_stats(a.reverse[0], valid)
    direction = "forward" if f["p99"] >= r["p99"] else "reverse"
    res = dict(crop=a.crop_json, rule="higher 99th percentile of the ink map over the crop's valid pixels; "
               "forward if equal", forward=f, reverse=r, direction=direction)
    secs = [json.load(open(lg))["seconds"] for lg in (a.forward[1], a.reverse[1])]
    res["crop_seconds"] = dict(forward=secs[0], reverse=secs[1])
    if a.project:
        per_patch = float(np.mean(secs)) / max(1, crop["patches_crop"])
        whole = middle = 0
        for rz in a.project:
            mocc, attrs = model_occupancy(rz)
            whole += count_patches(mocc)
            middle += count_patches(mocc, roi=middle_rows_roi(attrs, mocc.shape, crop["middle_rows"]))
        hw, hm = per_patch * whole / 3600, per_patch * middle / 3600
        if hw <= a.budget_hours:
            scope, roi = "whole", None
        elif hm <= a.budget_hours:
            scope, roi = "middle rows", crop["middle_roi"]
        else:
            scope, roi = "stop", None
        res.update(seconds_per_patch=round(per_patch, 4), renders=a.project, patches_whole=whole,
                   patches_middle=middle, projected_hours_whole=round(hw, 2),
                   projected_hours_middle=round(hm, 2), budget_hours=a.budget_hours, scope=scope, roi=roi,
                   scope_rule="whole if the projection is within the budget, else the middle vertex rows, "
                   "else stop (time per patch from the crop runs, both directions averaged)")
    json.dump(res, open(a.out_json, "w"), indent=1)
    msg = f"direction: {direction} (99th percentile forward {f['p99']:.3f}, reverse {r['p99']:.3f})"
    if a.project:
        msg += (f"; projected {res['projected_hours_whole']:.2f} h for the whole segment, "
                f"{res['projected_hours_middle']:.2f} h for the middle rows -> scope: {res['scope']}")
    print(msg)


def cmd_render_check(a):
    d = render_decision(json.load(open(a.check_json)), a.normals, a.center, a.step)
    d["check"] = a.check_json
    json.dump(d, open(a.out_json, "w"), indent=1)
    print(f"renderer check: {d['verdict']} (mean correlation {d['mean_corr']:.3f}, {d['order']} order, "
          f"the model's centre layer of the official volume is at {d['its_offset_along_our_normals']:+.1f} "
          f"voxels along our normals); render with --normals {d['normals']} --center {d['center']:g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render-check")
    p.add_argument("check_json")
    p.add_argument("out_json")
    p.add_argument("--normals", default="grid", choices=("grid", "flipped"))
    p.add_argument("--center", type=float, default=0.0)
    p.add_argument("--step", type=float, default=1.0)
    p = sub.add_parser("crop")
    p.add_argument("render")
    p.add_argument("out_json")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--middle-rows", type=int, nargs=2, default=[96, 288])
    p = sub.add_parser("choose")
    p.add_argument("crop_json")
    p.add_argument("out_json")
    p.add_argument("--forward", nargs=2, metavar=("TIF", "LOG"), required=True)
    p.add_argument("--reverse", nargs=2, metavar=("TIF", "LOG"), required=True)
    p.add_argument("--project", nargs="+")
    p.add_argument("--budget-hours", type=float, default=8.0)
    a = ap.parse_args()
    {"render-check": cmd_render_check, "crop": cmd_crop, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
