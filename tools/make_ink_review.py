"""Build the blind legibility review of the pre-registered ink test (results/ink_test_plan.md) from
the two w052 ink maps (original and snapped mesh, seed 42).

Usage:
  python tools/make_ink_review.py OUT_DIR KEY_JSON --original TIF --snapped TIF --renders R_ORIG R_SNAP
      [--roi R0 R1 C0 C1] [--practice-map TIF --practice-renders R_ORIG R_SNAP] [--seed 2026]
      [--display {fixed,midway}] [--n-comparisons N] [--n-controls N] [--reuse-places KEY_JSON]
      [--fixed-arm N] [--level-only N] [--weak-catch N] [--weak-sigma S]

Items, as pre-registered (with amendments 1 and 2):
  - 60 comparisons: 512 x 512 crops of the two maps at the same place. Centres are drawn at random
    (seed 2026) from pixels where the higher of the two maps is at least 0.5; if fewer than 60
    places qualify, from the top 1% of pixels instead. Crops lie inside the map, are at least 90%
    valid in both renders, and don't overlap (centres at least 512 pixels apart in rows or columns).
  - 10 controls: the same, from any valid pixel.
  - 8 catch items: one map against itself blurred (Gaussian, sigma 4), at further places drawn as
    for the comparisons; 4 from each map. The sharp one is the right answer.
  - Which map is A is balanced: exactly half of the comparisons, controls and catch items each.
    Items are shuffled.
  - Display: the model card's scaling (p - 0.25) / 0.5 clipped to 0..1, identically for both
    panels; pixels outside the mesh (either render) are black in both.
  - Practice (optional, from another segment's map): sharp against blurred, with feedback.

Options for a follow-up review that separates "dimmer" from "less legible" (all off by default; without them the
output is byte-identical to the pre-registered set):
  --display midway  every comparison, control, catch and practice pair goes through a monotone per-crop LUT onto the
                    midway of its two histograms (over the crop's valid pixels), then the same fixed display. Level and
                    contrast are then the same in both panels and only structure differs.
  --fixed-arm N     N more comparisons (kind "comparison_fixed") on the fixed scale, for a concurrent check.
  --level-only N    N items (kind "level_only"): the original map against itself pushed to the snapped crop's histogram,
                    fixed scale. Structure is identical by construction, so answers measure dimming alone.
  --weak-catch N    N more catch items (kind "catch_weak", half from each map) blurred with --weak-sigma (default 2).
  --n-comparisons, --n-controls  counts (default 60 and 10).
  --reuse-places K  comparisons and controls at the places of an earlier key instead of new draws.

Writes OUT_DIR/items.json and OUT_DIR/img/*.png, which say nothing about which map is which, and
KEY_JSON, the answer key, to keep private until the answers are in. The set id is the first 12
hex digits of the SHA-256 of json.dumps(key, sort_keys=True), as for the earlier reviews.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ink_scores import display, load_map, valid_both  # noqa: E402

CROP, HALF = 512, 256
N_COMP, N_CTRL, N_CATCH, MIN_VALID, SIGMA = 60, 10, 8, 0.9, 4.0


def integral(mask):
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    S[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    return S


def feasible(pool, valid, min_valid=MIN_VALID):
    """Pixels (flat indices) whose 512-pixel crop lies inside the map and is valid enough."""
    H, W = valid.shape
    y, x = np.divmod(np.asarray(pool, np.int64), W)
    inside = (y >= HALF) & (y <= H - HALF) & (x >= HALF) & (x <= W - HALF)
    y, x = y[inside], x[inside]
    S = integral(valid)
    s = S[y + HALF, x + HALF] - S[y - HALF, x + HALF] - S[y + HALF, x - HALF] + S[y - HALF, x - HALF]
    keep = s >= min_valid * CROP * CROP
    return y[keep], x[keep]


def draw(ys, xs, rng, need, taken):
    """Up to `need` centres in random order, none closer than CROP (rows or columns) to another."""
    if need <= 0:
        return []
    out = []
    ty = np.array([t[0] for t in taken], np.int64)
    tx = np.array([t[1] for t in taken], np.int64)
    for i in rng.permutation(len(ys)):
        y, x = int(ys[i]), int(xs[i])
        if ty.size and np.any(np.maximum(np.abs(ty - y), np.abs(tx - x)) < CROP):
            continue
        out.append((y, x))
        ty, tx = np.append(ty, y), np.append(tx, x)
        if len(out) == need:
            break
    return out


def ink_pools(p_o, p_s, valid):
    pmax = np.maximum(p_o, p_s)
    over = np.flatnonzero((valid & (pmax.astype(np.float32) / 255 >= 0.5)).ravel())
    thr = np.percentile(pmax[valid], 99)
    top = np.flatnonzero((valid & (pmax >= thr)).ravel())
    return pmax, over, top, float(thr) / 255


def blurred(p, valid, y, x, sigma=SIGMA):
    """The crop of p around (y, x), blurred with a normalised Gaussian over valid pixels only."""
    from scipy.ndimage import gaussian_filter
    m = int(np.ceil(4 * sigma))
    H, W = p.shape
    y0, y1, x0, x1 = max(0, y - HALF - m), min(H, y + HALF + m), max(0, x - HALF - m), min(W, x + HALF + m)
    v = valid[y0:y1, x0:x1].astype(np.float32)
    f = p[y0:y1, x0:x1].astype(np.float32) * v
    out = gaussian_filter(f, sigma) / np.maximum(gaussian_filter(v, sigma), 1e-6)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out[y - HALF - y0:y + HALF - y0, x - HALF - x0:x + HALF - x0]


def crop(a, y, x):
    return a[y - HALF:y + HALF, x - HALF:x + HALF]


def lut_to(src, ref_sorted):
    """256-entry monotone LUT: each level of uint8 `src` goes to the mean of `ref_sorted` over that level's ranks."""
    h = np.bincount(src, minlength=256)
    cum = np.cumsum(h)
    lo = np.concatenate([[0], cum[:-1]])
    S = np.concatenate([[0.0], np.cumsum(ref_sorted, dtype=np.float64)])
    lut = np.arange(256, dtype=np.float64)
    used = h > 0
    lut[used] = (S[cum[used]] - S[lo[used]]) / h[used]
    return lut


def midway(a, b, valid_crop):
    """Two uint8 crops on the midway of their histograms, as floats on the 0..255 scale. Valid pixels decide the LUTs."""
    va, vb = a[valid_crop], b[valid_crop]
    ref = 0.5 * (np.sort(va).astype(np.float64) + np.sort(vb).astype(np.float64))
    return lut_to(va, ref)[a], lut_to(vb, ref)[b]


def to_histogram_of(a, b, valid_crop):
    """uint8 crop `a` pushed to the histogram of crop `b` by a monotone LUT: the structure of `a` at the level of `b`."""
    return lut_to(a[valid_crop], np.sort(b[valid_crop]).astype(np.float64))[a]


def save_png(path, p_u8, valid_crop):
    from PIL import Image
    d = display(p_u8)
    d[~valid_crop] = 0
    Image.fromarray(np.rint(d * 255).astype(np.uint8), mode="L").save(path, optimize=True)


def build_practice(a, rng, img_dir):
    if not a.practice_map:
        return []
    p = load_map(a.practice_map)
    valid = valid_both(a.practice_renders, shape=p.shape)
    _, over, top, _ = ink_pools(p, p, valid)
    ys, xs = feasible(over if len(over) else top, valid)
    spots = draw(ys, xs, rng, a.practice, [])
    out = []
    for k, (y, x) in enumerate(spots):
        sharp_first = k % 2 == 0
        vc = crop(valid, y, x)
        panels = [crop(p, y, x), blurred(p, valid, y, x)]
        if a.display == "midway":
            panels = list(midway(panels[0], panels[1], vc))
        if not sharp_first:
            panels = panels[::-1]
        ids = []
        for side, img in zip("ab", panels):
            rel = f"img/p{k + 1}{side}.png"
            save_png(os.path.join(img_dir, os.path.basename(rel)), img, vc)
            ids.append(rel)
        better = "A" if sharp_first else "B"
        out.append(dict(id=f"p{k + 1}", a=ids[0], b=ids[1], better=better,
                        why=f"{better} is the sharp one; the other is the same map blurred, which smears the strokes."))
    return [out[i] for i in rng.permutation(len(out))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("key_json")
    ap.add_argument("--original", required=True)
    ap.add_argument("--snapped", required=True)
    ap.add_argument("--renders", nargs=2, required=True)
    ap.add_argument("--roi", type=int, nargs=4)
    ap.add_argument("--practice-map")
    ap.add_argument("--practice-renders", nargs="+")
    ap.add_argument("--practice", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--display", choices=("fixed", "midway"), default="fixed")
    ap.add_argument("--n-comparisons", type=int, default=N_COMP)
    ap.add_argument("--n-controls", type=int, default=N_CTRL)
    ap.add_argument("--reuse-places", metavar="KEY_JSON")
    ap.add_argument("--fixed-arm", type=int, default=0)
    ap.add_argument("--level-only", type=int, default=0)
    ap.add_argument("--weak-catch", type=int, default=0)
    ap.add_argument("--weak-sigma", type=float, default=2.0)
    a = ap.parse_args()
    img_dir = os.path.join(a.out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    p_o, p_s = load_map(a.original), load_map(a.snapped)
    if p_o.shape != p_s.shape:
        sys.exit(f"the two maps differ in shape: {p_o.shape} vs {p_s.shape}")
    full = valid_both(a.renders)
    roi = a.roi or [0, full.shape[0], 0, full.shape[1]]
    valid = full[roi[0]:roi[1], roi[2]:roi[3]]
    if valid.shape != p_o.shape:
        sys.exit(f"the maps are {p_o.shape}, the renders' area {valid.shape}; give the run's --roi")
    pmax, over, top, thr = ink_pools(p_o, p_s, valid)

    # comparisons: from p >= 0.5, or the top 1% if fewer than 60 places qualify
    ys, xs = feasible(over, valid)
    pool_name = "higher map >= 0.5"
    if a.reuse_places:
        old = json.load(open(a.reuse_places))["key"]
        comp = [tuple(v["centre_yx"]) for _, v in sorted(old.items()) if v["kind"] == "comparison"]
        ctrl = [tuple(v["centre_yx"]) for _, v in sorted(old.items()) if v["kind"] == "control"]
        pool_name = f"places of {os.path.basename(a.reuse_places)}"
    else:
        comp = draw(ys, xs, rng, a.n_comparisons, [])
        if len(comp) < a.n_comparisons:
            ys, xs = feasible(top, valid)
            comp = draw(ys, xs, rng, a.n_comparisons, [])
            pool_name = f"top 1% (higher map >= {thr:.3f})"
    taken = list(comp)
    cy, cx = feasible(np.flatnonzero(valid.ravel()), valid)
    if not a.reuse_places:
        ctrl = draw(cy, cx, rng, a.n_controls, taken)
    taken += ctrl
    catch = draw(ys, xs, rng, N_CATCH, taken)
    catch_pool = pool_name
    if len(catch) < N_CATCH:
        ty, tx = feasible(top, valid)
        catch += draw(ty, tx, rng, N_CATCH - len(catch), taken + catch)
        catch_pool += " then top 1%"
    if len(catch) < N_CATCH:                    # last resort: reuse comparison places
        extra = [comp[i] for i in rng.permutation(len(comp))[:N_CATCH - len(catch)]]
        catch += extra
        catch_pool += " then comparison places"
    print(f"pool: {pool_name}, {len(over):,} pixels with the higher map >= 0.5; comparisons {len(comp)}, "
          f"controls {len(ctrl)}, catch {len(catch)}", flush=True)

    raw = [dict(kind="comparison", c=c) for c in comp] + [dict(kind="control", c=c) for c in ctrl]
    catch_maps = (["original"] * (N_CATCH // 2) + ["snapped"] * (N_CATCH - N_CATCH // 2))
    catch_maps = [catch_maps[i] for i in rng.permutation(len(catch_maps))][:len(catch)]
    raw += [dict(kind="catch", c=c, map=m) for c, m in zip(catch, catch_maps)]
    # extra arms of a follow-up review, drawn last so that the default draws don't change
    taken += catch
    extra = [("comparison_fixed", a.fixed_arm), ("level_only", a.level_only), ("catch_weak", a.weak_catch)]
    for kind, n in extra:
        if n <= 0:
            continue
        spots = draw(ys, xs, rng, n, taken)
        if len(spots) < n:
            sys.exit(f"only {len(spots)} free places left for {n} {kind} items: lower the counts")
        taken += spots
        maps_ = [("original", "snapped")[i % 2] for i in range(n)]
        raw += [dict(kind=kind, c=c, map=m) for c, m in zip(spots, maps_)]
    first = np.zeros(len(raw), bool)            # True: original (or the sharp map) is A
    kinds_used = ["comparison", "control", "catch"] + [k for k, n in extra if n > 0]
    for kind in kinds_used:
        members = [j for j, r in enumerate(raw) if r["kind"] == kind]
        if members:
            first[rng.permutation(members)[:len(members) // 2]] = True
    order = rng.permutation(len(raw))

    items, key = [], {}
    for rank, j in enumerate(order, 1):
        r = raw[j]
        y, x = r["c"]
        vc = crop(valid, y, x)
        if r["kind"] in ("catch", "catch_weak"):
            src = p_o if r["map"] == "original" else p_s
            kinds = [f"{r['map']} (sharp)", f"{r['map']} (blurred)"]
            sigma = SIGMA if r["kind"] == "catch" else a.weak_sigma
            panels = [crop(src, y, x), blurred(src, valid, y, x, sigma)]
        elif r["kind"] == "level_only":
            kinds = ["original", "original at the snapped level"]
            panels = [crop(p_o, y, x), to_histogram_of(crop(p_o, y, x), crop(p_s, y, x), vc)]
        else:
            kinds = ["original", "snapped"]
            panels = [crop(p_o, y, x), crop(p_s, y, x)]
        shown = "fixed" if r["kind"] in ("level_only", "comparison_fixed") else a.display
        if shown == "midway":
            panels = list(midway(panels[0], panels[1], vc))
        if not first[j]:
            kinds, panels = kinds[::-1], panels[::-1]
        qid = f"q{rank:03d}"
        paths = []
        for side, img in zip("ab", panels):
            rel = f"img/{qid}{side}.png"
            save_png(os.path.join(a.out_dir, rel), img, vc)
            paths.append(rel)
        items.append(dict(id=qid, a=paths[0], b=paths[1]))
        key[qid] = dict(kind=r["kind"], A=kinds[0], B=kinds[1], centre_yx=[int(y), int(x)],
                        centre_render_yx=[int(y + roi[0]), int(x + roi[2])], crop_valid=round(float(vc.mean()), 4),
                        pmax_centre=round(float(pmax[y, x]) / 255, 4),
                        p_original_mean=round(float(crop(p_o, y, x)[vc].mean()) / 255, 4),
                        p_snapped_mean=round(float(crop(p_s, y, x)[vc].mean()) / 255, 4))
        if a.display != "fixed" or any(n > 0 for _, n in extra):
            key[qid]["display"] = shown
    practice = build_practice(a, np.random.default_rng(a.seed + 1), img_dir)
    set_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    meta = dict(original=a.original, snapped=a.snapped, renders=a.renders, roi=roi, seed=a.seed,
                pool=pool_name, catch_pool=catch_pool, pixels_over_half=int(len(over)), top1_threshold=thr)
    if a.display != "fixed" or any(n > 0 for _, n in extra):
        meta.update(display=a.display, weak_sigma=a.weak_sigma, reuse_places=a.reuse_places)
    with open(os.path.join(a.out_dir, "items.json"), "w") as f:
        json.dump(dict(version="ink-1", set=set_id, crop=CROP, items=items, practice=practice), f, indent=1)
    with open(a.key_json, "w") as f:
        json.dump(dict(key=key, meta=meta), f, indent=1)
    counts = {}
    for v in key.values():
        counts[v["kind"]] = counts.get(v["kind"], 0) + 1
    print(f"items: {len(items)} {counts} | practice: {len(practice)} | set: {set_id}")


if __name__ == "__main__":
    main()
