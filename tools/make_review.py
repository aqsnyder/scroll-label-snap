"""Build a blind A/B review set: CT close-ups with two candidate labels in random order.

For each cube, picks surface points where the snapper moved the label a lot (|shift| >= 2
voxels), moderately (1-2 voxels), and not at all (controls), spread out in space. Adds catch
items where one candidate is the original label deliberately pushed off the surface, so
the review's own reliability can be checked.

Each item is one small PNG sprite (native voxel resolution) with four panels:
  context (2x zoomed out) | CT close-up | candidate A | candidate B
The page upscales it without smoothing.

Usage:
  python tools/make_review.py OUT_DIR KEY_JSON IMAGES_DIR LABELS_DIR SNAPPED_DIR [--seed 0]

Writes OUT_DIR/items.json (ids + sprites only; nothing says which candidate is which) and
KEY_JSON (the answer key: keep it away from the reviewer until the review is done).
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
from scroll_label_snap.core import Params, prepare, warp_labels  # noqa: E402
from eval_snap import band_position  # noqa: E402

HALF = 24                       # close-up is 48 x 48 voxels
GAP = 2                         # pixels between panels
YELLOW = np.array([255, 210, 40], np.float32)
ALPHA = 0.42


def slice2d(vol, c, axis, half):
    """2-D window of `vol` through point c (z, y, x), perpendicular to `axis`."""
    sl = [slice(int(round(v)) - half, int(round(v)) + half) for v in c]
    sl[axis] = int(round(c[axis]))
    return vol[tuple(sl)]


def to_gray(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255


def overlay(gray, mask):
    rgb = np.repeat(gray[..., None], 3, -1)
    rgb[mask] = rgb[mask] * (1 - ALPHA) + YELLOW * ALPHA
    return rgb


def sprite(img, lab_a, lab_b, c, axis, lo, hi):
    ct = to_gray(slice2d(img, c, axis, HALF), lo, hi)
    ctx_full = to_gray(slice2d(img, c, axis, 2 * HALF), lo, hi)
    ctx = ctx_full.reshape(2 * HALF, 2, 2 * HALF, 2).mean((1, 3))      # 2x zoomed out
    ctx_rgb = np.repeat(ctx[..., None], 3, -1)
    q0, q1 = HALF // 2, HALF // 2 + HALF - 1                            # box = close-up area
    for k in range(q0, q1 + 1):
        for (y, x) in ((q0, k), (q1, k), (k, q0), (k, q1)):
            ctx_rgb[y, x] = [90, 200, 255]
    a = overlay(ct, slice2d(lab_a, c, axis, HALF) == 1)
    b = overlay(ct, slice2d(lab_b, c, axis, HALF) == 1)
    ct_rgb = np.repeat(ct[..., None], 3, -1)
    w = 2 * HALF
    out = np.full((w, 4 * w + 3 * GAP, 3), 255, np.float32)
    for k, panel in enumerate((ctx_rgb, ct_rgb, a, b)):
        out[:, k * (w + GAP):k * (w + GAP) + w] = panel
    buf = io.BytesIO()
    Image.fromarray(out.astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pick_spread(cands, pts, k, rng, min_dist=32.0, taken=()):
    """Up to k candidate indices, each at least min_dist voxels from the others."""
    chosen = list(taken)
    out = []
    for i in rng.permutation(cands):
        if all(np.linalg.norm(pts[i] - pts[j]) >= min_dist for j in chosen):
            chosen.append(i)
            out.append(int(i))
            if len(out) == k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("key_json")
    ap.add_argument("images_dir")
    ap.add_argument("labels_dir")
    ap.add_argument("snapped_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catch-cubes", default="sample_00001,sample_00004,sample_00009,sample_00013,sample_00019")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    catch_cubes = a.catch_cubes.split(",")
    names = sorted(f[:-4] for f in os.listdir(a.snapped_dir) if f.endswith(".tif"))

    raw, example = [], None
    for name in names:
        img = tifffile.imread(os.path.join(a.images_dir, name + ".tif"))
        lab = tifffile.imread(os.path.join(a.labels_dir, name + ".tif"))
        snapped = tifffile.imread(os.path.join(a.snapped_dir, name + ".tif"))
        lo, hi = np.percentile(img, [1, 99.5])
        _, pts, n, _, _, _ = prepare(img, lab, Params())
        margin = 2 * HALF + 2
        inner = np.all((pts >= margin) & (pts <= np.array(lab.shape) - 1 - margin), axis=1)
        idx = np.flatnonzero(inner)
        p, nn = pts[idx], n[idx]
        delta = band_position(snapped, p, nn) - band_position(lab, p, nn)
        ok = np.isfinite(delta)
        big = np.flatnonzero(ok & (np.abs(delta) >= 2.0))
        mid = np.flatnonzero(ok & (np.abs(delta) >= 1.0) & (np.abs(delta) < 2.0))
        same = np.flatnonzero(ok & (np.abs(delta) <= 0.25))
        sel_big = pick_spread(big, p, 3, rng)
        sel_mid = pick_spread(mid, p, 3 + (3 - len(sel_big)), rng, taken=sel_big)
        sel_same = pick_spread(same, p, 1, rng, taken=sel_big + sel_mid)
        for stratum, sel in (("big move", sel_big), ("moderate move", sel_mid), ("no move", sel_same)):
            for i in sel:
                axis = int(np.argmin(np.abs(nn[i])))
                raw.append(dict(cube=name, stratum=stratum, c=p[i], axis=axis,
                                delta=float(delta[i]), img=img, lo=lo, hi=hi,
                                labs={"original": lab, "snapped": snapped}))

        if name in catch_cubes:
            # original vs the original pushed off the surface: into the gap or into the sheet
            k = catch_cubes.index(name)
            shift = 3.0 if k % 2 == 0 else -2.5
            pushed = warp_labels(lab, pts, n, np.full(len(pts), shift, np.float32))
            sel = pick_spread(same if len(same) else np.arange(len(p)), p, 3, rng,
                              taken=sel_big + sel_mid + sel_same)
            tag = f"pushed {shift:+.1f}"
            for t, i in enumerate(sel):
                axis = int(np.argmin(np.abs(nn[i])))
                item = dict(cube=name, stratum="catch", c=p[i], axis=axis, delta=shift, img=img,
                            lo=lo, hi=hi, labs={"original": lab, tag: pushed})
                if example is None and shift > 0 and t == len(sel) - 1:
                    example = item          # one gap-ward catch used as the worked example
                else:
                    raw.append(item)
        print(f"{name}: big {len(sel_big)}, moderate {len(sel_mid)}, controls {len(sel_same)}", flush=True)

    order = rng.permutation(len(raw))
    # Exactly half of the comparisons show the snapped label as A, and exactly half of the
    # catch items show the original as A, so a reviewer who always picks one side scores 50%.
    original_first = np.zeros(len(raw), bool)
    for is_catch in (False, True):
        members = [j for j in range(len(raw)) if (raw[j]["stratum"] == "catch") == is_catch]
        original_first[rng.permutation(members)[:len(members) // 2]] = True
    items, key = [], {}
    for rank, j in enumerate(order, 1):
        it = raw[j]
        kinds = list(it["labs"].keys())          # ["original", "snapped" | "pushed ..."]
        if not original_first[j]:
            kinds = kinds[::-1]
        qid = f"q{rank:03d}"
        items.append(dict(id=qid, sprite=sprite(it["img"], it["labs"][kinds[0]], it["labs"][kinds[1]],
                                                it["c"], it["axis"], it["lo"], it["hi"])))
        key[qid] = dict(cube=it["cube"], stratum=it["stratum"], A=kinds[0], B=kinds[1],
                        point_zyx=[round(float(v), 1) for v in it["c"]],
                        center_zyx=[int(round(float(v))) for v in it["c"]],   # voxel the slices are cut through
                        slice_axis="zyx"[it["axis"]], delta=round(it["delta"], 2))
    ex = None
    if example is not None:
        ex = dict(sprite=sprite(example["img"], example["labs"]["original"],
                                list(example["labs"].values())[1], example["c"], example["axis"],
                                example["lo"], example["hi"]), better="A")
    # Identifies this exact set; the page stamps it on every answer and the scorer checks it.
    set_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    with open(os.path.join(a.out_dir, "items.json"), "w") as f:
        json.dump(dict(version=1, set=set_id, panel=2 * HALF, gap=GAP, items=items, example=ex), f)
    with open(a.key_json, "w") as f:
        json.dump(key, f, indent=1)
    counts = {}
    for v in key.values():
        counts[v["stratum"]] = counts.get(v["stratum"], 0) + 1
    print("items:", len(items), counts, "| example:", ex is not None, "| set:", set_id)


if __name__ == "__main__":
    main()
