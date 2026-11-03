"""Build blind review set v2: fresh points, checks that hold across the whole window, practice
items with feedback, and worked examples for a reviewer who has never seen these scans.

Differences from make_review.py (v1):
- Points near any item of an earlier review (--exclude-key) are skipped.
- Controls are windows where the two labels agree across the whole 48 x 48 close-up (window
  IoU >= --control-iou), not just at the centre point.
- Catch items push the original label off the surface only where both labels agree across the
  window (so the original is right there) and where the scan confirms the direction: for a push
  into the gap the gap in front must be dark, for a push into the sheet the papyrus behind must
  be bright. Half go each way.
- Practice items (same kind as the catch items, with an explanation) and a worked example
  (good / too far out / too far in) are written to items.json for the page's tutorial.

Usage:
  python tools/make_review_v2.py OUT_DIR KEY_JSON IMAGES_DIR LABELS_DIR SNAPPED_DIR \\
      [--exclude-key results/review_v1_key.json] [--seed 1]
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys

import numpy as np
import tifffile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from scroll_label_snap.core import Params, prepare, warp_labels, sample  # noqa: E402
from eval_snap import band_position  # noqa: E402
from make_review import HALF, GAP, slice2d, to_gray, overlay, pick_spread  # noqa: E402

PUSH_GAP, PUSH_SHEET = 3.0, -2.5


def cut(img, lo, hi, labs, c, axis):
    """Everything an item needs from the volumes: CT close-up, context view, label masks."""
    ct = to_gray(slice2d(img, c, axis, HALF), lo, hi)
    ctx = to_gray(slice2d(img, c, axis, 2 * HALF), lo, hi).reshape(2 * HALF, 2, 2 * HALF, 2).mean((1, 3))
    return dict(ct=ct, ctx=ctx, masks={k: slice2d(v, c, axis, HALF) == 1 for k, v in labs.items()})


def png(panels):
    w = 2 * HALF
    out = np.full((w, len(panels) * w + (len(panels) - 1) * GAP, 3), 255, np.float32)
    for k, p in enumerate(panels):
        out[:, k * (w + GAP):k * (w + GAP) + w] = p
    buf = io.BytesIO()
    Image.fromarray(out.astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def context_panel(cu):
    rgb = np.repeat(cu["ctx"][..., None], 3, -1)
    q0, q1 = HALF // 2, HALF // 2 + HALF - 1
    for k in range(q0, q1 + 1):
        for (y, x) in ((q0, k), (q1, k), (k, q0), (k, q1)):
            rgb[y, x] = [90, 200, 255]
    return rgb


def sprite(cu, a, b):
    """context | CT | A | B, as in v1."""
    ct_rgb = np.repeat(cu["ct"][..., None], 3, -1)
    return png([context_panel(cu), ct_rgb, overlay(cu["ct"], cu["masks"][a]), overlay(cu["ct"], cu["masks"][b])])


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def profiles(ct_s, pts, n, offs):
    return np.stack([sample(ct_s, pts + t * n) for t in offs], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("key_json")
    ap.add_argument("images_dir")
    ap.add_argument("labels_dir")
    ap.add_argument("snapped_dir")
    ap.add_argument("--exclude-key", default=None)
    ap.add_argument("--exclude-radius", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-cube", type=int, default=2, help="big and moderate moves per cube, each")
    ap.add_argument("--controls", type=int, default=15)
    ap.add_argument("--catch", type=int, default=16)
    ap.add_argument("--practice", type=int, default=4)
    ap.add_argument("--control-iou", type=float, default=0.8)
    ap.add_argument("--catch-iou", type=float, default=0.75)
    ap.add_argument("--scan-per-cube", type=int, default=4000,
                    help="centre-unchanged points per cube checked for window agreement")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    excl = json.load(open(a.exclude_key)) if a.exclude_key else {}
    names = sorted(f[:-4] for f in os.listdir(a.snapped_dir) if f.endswith(".tif"))

    comparisons, control_pool, catch_pool, cube_data = [], [], [], {}
    for name in names:
        img = tifffile.imread(os.path.join(a.images_dir, name + ".tif"))
        lab = tifffile.imread(os.path.join(a.labels_dir, name + ".tif"))
        snapped = tifffile.imread(os.path.join(a.snapped_dir, name + ".tif"))
        lo, hi = np.percentile(img, [1, 99.5])
        ct_s, pts, n, _, _, comp = prepare(img, lab, Params())
        cube_data[name] = (pts, n, comp, lo, hi)
        margin = 2 * HALF + 2
        inner = np.all((pts >= margin) & (pts <= np.array(lab.shape) - 1 - margin), axis=1)
        old = np.array([v["point_zyx"] for v in excl.values() if v["cube"] == name]).reshape(-1, 3)
        if len(old):
            dmin = np.min(np.linalg.norm(pts[:, None, :] - old[None], axis=2), axis=1) if len(pts) * len(old) < 5e7 \
                else np.array([np.min(np.linalg.norm(old - x, axis=1)) for x in pts])
            inner &= dmin >= a.exclude_radius
        idx = np.flatnonzero(inner)
        p, nn = pts[idx], n[idx]
        delta = band_position(snapped, p, nn) - band_position(lab, p, nn)
        ok = np.isfinite(delta)
        big = np.flatnonzero(ok & (np.abs(delta) >= 2.0))
        mid = np.flatnonzero(ok & (np.abs(delta) >= 1.0) & (np.abs(delta) < 2.0))
        sel_big = pick_spread(big, p, a.per_cube, rng)
        sel_mid = pick_spread(mid, p, a.per_cube + (a.per_cube - len(sel_big)), rng, taken=sel_big)
        labs = {"original": lab, "snapped": snapped}
        for stratum, sel in (("big move", sel_big), ("moderate move", sel_mid)):
            for i in sel:
                axis = int(np.argmin(np.abs(nn[i])))
                cu = cut(img, lo, hi, labs, p[i], axis)
                comparisons.append(dict(cube=name, stratum=stratum, c=p[i], axis=axis, delta=float(delta[i]),
                                        cut=cu, kinds=("original", "snapped"),
                                        window_iou=iou(cu["masks"]["original"], cu["masks"]["snapped"])))

        # centre-unchanged points: how well do the two labels agree across the whole window,
        # and does the scan show papyrus behind the label and gap in front of it?
        same = np.flatnonzero(ok & (np.abs(delta) <= 0.25))
        same = np.setdiff1d(same, np.array(sel_big + sel_mid, int))
        if len(same) > a.scan_per_cube:
            same = rng.choice(same, a.scan_per_cube, replace=False)
        offs = np.arange(-4.0, 5.01, 0.5, dtype=np.float32)
        prof = profiles(ct_s, p[same], nn[same], offs)
        near = prof[:, np.abs(offs) <= 3.0]
        mid_lvl = float(np.median(near))
        contrast = float(np.percentile(near, 90) - np.percentile(near, 10))
        m = lambda lo_, hi_: prof[:, (offs >= lo_) & (offs <= hi_)].mean(1)
        behind, front = m(-2.5, -0.5), m(2.0, 4.5)
        deep = m(-3.5, -1.5)
        at_edge = (behind > mid_lvl + 0.15 * contrast) & (front < mid_lvl - 0.15 * contrast)
        gap_ok = at_edge                                   # +3 lands in a dark gap
        sheet_ok = at_edge & (deep > mid_lvl + 0.15 * contrast)   # -2.5 lands inside bright papyrus
        for k, i in enumerate(same):
            axis = int(np.argmin(np.abs(nn[i])))
            o = slice2d(lab, p[i], axis, HALF) == 1
            s = slice2d(snapped, p[i], axis, HALF) == 1
            w = iou(o, s)
            if w >= a.control_iou:
                control_pool.append(dict(cube=name, c=p[i], axis=axis, delta=float(delta[i]), window_iou=w))
            if w >= a.catch_iou and (gap_ok[k] or sheet_ok[k]):
                catch_pool.append(dict(cube=name, c=p[i], axis=axis, window_iou=w,
                                       gap=bool(gap_ok[k]), sheet=bool(sheet_ok[k])))
        print(f"{name}: big {len(sel_big)}, moderate {len(sel_mid)}; control pool {len(control_pool)}, "
              f"catch pool {len(catch_pool)}", flush=True)

    def spread_pick(pool, k, per_cube, taken):
        """k entries, at most per_cube per cube, 32+ voxels from each other and from taken."""
        out = []
        for j in rng.permutation(len(pool)):
            e = pool[j]
            same_cube = [t for t in taken + out if t["cube"] == e["cube"]]
            if sum(1 for t in out if t["cube"] == e["cube"]) >= per_cube:
                continue
            if all(np.linalg.norm(e["c"] - t["c"]) >= 32.0 for t in same_cube):
                out.append(e)
                if len(out) == k:
                    break
        return out

    taken = [dict(cube=c["cube"], c=c["c"]) for c in comparisons]
    controls = spread_pick(control_pool, a.controls, 2, taken)
    taken += controls
    example = spread_pick([e for e in catch_pool if e["gap"] and e["sheet"]], 1, 1, taken)
    taken += example
    half = a.catch // 2
    catch_gap = spread_pick([e for e in catch_pool if e["gap"]], half + a.practice // 2, 2, taken)
    taken += catch_gap
    catch_sheet = spread_pick([e for e in catch_pool if e["sheet"]], half + a.practice - a.practice // 2, 2, taken)
    practice = catch_gap[half:] + catch_sheet[half:]
    catch_gap, catch_sheet = catch_gap[:half], catch_sheet[:half]
    print(f"controls {len(controls)}, catch {len(catch_gap)} gap + {len(catch_sheet)} sheet, practice {len(practice)}, "
          f"example {len(example)}", flush=True)

    # second pass: cut the controls, and push the original label for catch, practice and example
    pushed_cache = {}

    def pushed(e, shift):
        """Original label pushed by `shift` along the normal, in a crop around e (enough for the window)."""
        name = e["cube"]
        lab = tifffile.imread(os.path.join(a.labels_dir, name + ".tif"))
        pts, n, comp, _, _ = cube_data[name]
        r = 2 * HALF + 8
        c0 = np.clip(np.rint(e["c"]).astype(int) - r, 0, None)
        c1 = np.minimum(np.rint(e["c"]).astype(int) + r, np.array(lab.shape))
        sub = lab[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
        inside = np.all((pts >= c0 - 6) & (pts < c1 + 6), axis=1)
        out = lab.copy()
        out[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] = warp_labels(
            sub, pts[inside] - c0, n[inside], np.full(inside.sum(), shift, np.float32), comp=comp[inside])
        return lab, out

    def cut_for(e, shift=None):
        name = e["cube"]
        img = tifffile.imread(os.path.join(a.images_dir, name + ".tif"))
        _, _, _, lo, hi = cube_data[name]
        if shift is None:
            lab = tifffile.imread(os.path.join(a.labels_dir, name + ".tif"))
            snapped = tifffile.imread(os.path.join(a.snapped_dir, name + ".tif"))
            return cut(img, lo, hi, {"original": lab, "snapped": snapped}, e["c"], e["axis"])
        lab, pu = pushed(e, shift)
        return cut(img, lo, hi, {"original": lab, f"pushed {shift:+.1f}": pu}, e["c"], e["axis"])

    raw = list(comparisons)
    for e in controls:
        raw.append(dict(e, stratum="no move", cut=cut_for(e), kinds=("original", "snapped")))
    for e, shift in [(e, PUSH_GAP) for e in catch_gap] + [(e, PUSH_SHEET) for e in catch_sheet]:
        cu = cut_for(e, shift)
        tag = f"pushed {shift:+.1f}"
        raw.append(dict(e, stratum="catch", delta=shift, cut=cu, kinds=("original", tag),
                        window_iou=iou(cu["masks"]["original"], cu["masks"][tag])))

    order = rng.permutation(len(raw))
    # exactly half of the comparisons (moves and controls) show the snapped label as A, and
    # exactly half of the catch items show the original as A
    original_first = np.zeros(len(raw), bool)
    for is_catch in (False, True):
        members = [j for j in range(len(raw)) if (raw[j]["stratum"] == "catch") == is_catch]
        original_first[rng.permutation(members)[:len(members) // 2]] = True
    items, key = [], {}
    for rank, j in enumerate(order, 1):
        it = raw[j]
        kinds = list(it["kinds"]) if original_first[j] else list(it["kinds"])[::-1]
        qid = f"q{rank:03d}"
        items.append(dict(id=qid, sprite=sprite(it["cut"], kinds[0], kinds[1])))
        key[qid] = dict(cube=it["cube"], stratum=it["stratum"], A=kinds[0], B=kinds[1],
                        point_zyx=[round(float(v), 1) for v in it["c"]],
                        center_zyx=[int(round(float(v))) for v in it["c"]],
                        slice_axis="zyx"[it["axis"]], delta=round(float(it["delta"]), 2),
                        window_iou=round(float(it["window_iou"]), 3))

    # practice: known answers, shown with an explanation after each pick (not part of the key)
    prac = []
    for k, e in enumerate(practice):
        shift = PUSH_GAP if k < a.practice // 2 else PUSH_SHEET
        cu = cut_for(e, shift)
        tag = f"pushed {shift:+.1f}"
        good_first = bool(rng.random() < 0.5)
        A, B = ("original", tag) if good_first else (tag, "original")
        bad = "B" if good_first else "A"
        why = (f"{bad}'s band floats out in the dark gap, about 3 voxels off the edge. The other one sits on it."
               if shift > 0 else
               f"{bad}'s band is buried inside the bright layer, about 2.5 voxels in. The other one sits on its edge.")
        prac.append(dict(id=f"p{k + 1}", sprite=sprite(cu, A, B), better="A" if good_first else "B", why=why,
                         cube=e["cube"], point_zyx=[round(float(v), 1) for v in e["c"]]))
    prac = [prac[i] for i in rng.permutation(len(prac))]

    ex = None
    if example:
        e = example[0]
        cu_out = cut_for(e, PUSH_GAP)
        cu_in = cut_for(e, PUSH_SHEET)
        ex = dict(sprite=png([np.repeat(cu_out["ct"][..., None], 3, -1),
                              overlay(cu_out["ct"], cu_out["masks"]["original"]),
                              overlay(cu_out["ct"], cu_out["masks"][f"pushed {PUSH_GAP:+.1f}"]),
                              overlay(cu_in["ct"], cu_in["masks"][f"pushed {PUSH_SHEET:+.1f}"])]),
                  panels=["Scan, no band", "Good: on the edge", "Too far out: in the gap", "Too far in: inside the layer"],
                  cube=e["cube"], point_zyx=[round(float(v), 1) for v in e["c"]])

    set_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    with open(os.path.join(a.out_dir, "items.json"), "w") as f:
        json.dump(dict(version=2, set=set_id, panel=2 * HALF, gap=GAP, items=items, practice=prac, example=ex), f)
    with open(a.key_json, "w") as f:
        json.dump(key, f, indent=1)
    counts = {}
    for v in key.values():
        counts[v["stratum"]] = counts.get(v["stratum"], 0) + 1
    print("items:", len(items), counts, "| practice:", len(prac), "| example:", ex is not None, "| set:", set_id)


if __name__ == "__main__":
    main()
