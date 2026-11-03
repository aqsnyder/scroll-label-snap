"""One voxel either side of the recto face: is it gap outside and papyrus inside? A direct check of
where a mesh sits, from surface volumes rendered with tools/render_surface_volume.py (suggested in
the Vesuvius Challenge Discord: from a perfect snap, a layer about 1 voxel off the surface should
lie entirely in the gap).

Usage:
  python tools/face_step_check.py OUT_JSON --render NAME=ZARR [--render NAME=ZARR ...]
         --face C [--normal-sign S] [--step 1.0] [--band 1024]

  --face C         where the mesh puts the recto face, in voxels along the snapper's normal (the
                   segment's convention_offset in snap_segment.json); the original mesh and the
                   snapped one keep the same convention
  --normal-sign S  the snapper's normal_sign (snap_segment.json); with the render's --normals it
                   says which way the render's layers run relative to the snapper's normal
  --step D         look D voxels outside and inside the face (default 1)

For every pixel valid in all renders, the scan at face + D (should be gap) and face - D (should be
papyrus), read by linear interpolation between the render's layers. The gap / papyrus threshold is
Otsu's, over those values pooled from all renders, so every mesh is judged by the same threshold.
Reports, per render: the share of pixels with gap outside, with papyrus inside, with both, and
with the scan brighter inside than outside, and the mean of each side.
"""
import argparse
import json
import sys

import numpy as np


def otsu(values, bins=256):
    h, edges = np.histogram(values, bins=bins, range=(0, 256))
    h = h.astype(np.float64)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    return float(edges[k + 1])


def layer_weights(offsets, t):
    """Two layers and their weights for the offset t (along the render's normal)."""
    o = np.asarray(offsets, np.float64)
    if not (o.min() <= t <= o.max()):
        raise ValueError(f"offset {t} is outside the render's layers ({o.min()} .. {o.max()})")
    i = int(np.searchsorted(o, t, side="right") - 1)
    i = min(i, len(o) - 2)
    w = (t - o[i]) / (o[i + 1] - o[i])
    return i, 1.0 - w, i + 1, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_json")
    ap.add_argument("--render", action="append", required=True, metavar="NAME=ZARR")
    ap.add_argument("--face", type=float, required=True)
    ap.add_argument("--normal-sign", type=int, default=1, choices=(-1, 1))
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--band", type=int, default=1024)
    a = ap.parse_args()
    import zarr
    renders = dict(r.split("=", 1) for r in a.render)
    Z = {k: zarr.open_array(v, mode="r") for k, v in renders.items()}
    shapes = {z.shape for z in Z.values()}
    if len(shapes) != 1:
        sys.exit(f"the renders differ in shape: {shapes}")
    L, H, W = shapes.pop()
    plan = {}
    for k, z in Z.items():
        at = dict(z.attrs)
        # the render's layer offsets run along its own normal: grid normals, or flipped; the
        # snapper's normal is the grid normal times normal_sign
        along = (1 if at.get("normals", "grid") == "grid" else -1) * a.normal_sign
        out_t, in_t = along * (a.face + a.step), along * (a.face - a.step)
        plan[k] = dict(outside=layer_weights(at["layer_offsets"], out_t), inside=layer_weights(at["layer_offsets"], in_t),
                       render_offset_outside=out_t, render_offset_inside=in_t)

    def read(k, r0, r1):
        z = Z[k]
        i0, w0, i1, w1 = plan[k]["outside"]
        j0, v0, j1, v1 = plan[k]["inside"]
        lo, hi = min(i0, j0), max(i1, j1) + 1
        blk = np.asarray(z[lo:hi, r0:r1]).astype(np.float32)
        out = w0 * blk[i0 - lo] + w1 * blk[i1 - lo]
        ins = v0 * blk[j0 - lo] + v1 * blk[j1 - lo]
        return out, ins, (blk > 0).any(0)

    # pass 1: the common valid area and the pooled values for the threshold (a sample)
    rng = np.random.default_rng(0)
    pool = []
    for r0 in range(0, H, a.band):
        r1 = min(H, r0 + a.band)
        parts = {k: read(k, r0, r1) for k in Z}
        valid = np.logical_and.reduce([p[2] for p in parts.values()])
        idx = np.flatnonzero(valid.ravel())
        if idx.size:
            take = rng.choice(idx, min(idx.size, 20000), replace=False)
            for o, i, _ in parts.values():
                pool += [o.ravel()[take], i.ravel()[take]]
    thr = otsu(np.concatenate(pool))
    # pass 2: the shares
    acc = {k: dict(n=0, gap_out=0, pap_in=0, both=0, brighter_in=0, sum_out=0.0, sum_in=0.0) for k in Z}
    for r0 in range(0, H, a.band):
        r1 = min(H, r0 + a.band)
        parts = {k: read(k, r0, r1) for k in Z}
        valid = np.logical_and.reduce([p[2] for p in parts.values()])
        for k, (o, i, _) in parts.items():
            o, i = o[valid], i[valid]
            s = acc[k]
            s["n"] += int(o.size)
            g, p = o < thr, i >= thr
            s["gap_out"] += int(g.sum())
            s["pap_in"] += int(p.sum())
            s["both"] += int((g & p).sum())
            s["brighter_in"] += int((i > o).sum())
            s["sum_out"] += float(o.sum())
            s["sum_in"] += float(i.sum())
    res = dict(face=a.face, step=a.step, normal_sign=a.normal_sign, threshold=thr, shape=[L, H, W], renders={})
    for k, s in acc.items():
        n = max(s["n"], 1)
        res["renders"][k] = dict(path=renders[k], pixels=s["n"], gap_outside=s["gap_out"] / n,
                                 papyrus_inside=s["pap_in"] / n, both=s["both"] / n, brighter_inside=s["brighter_in"] / n,
                                 mean_outside=s["sum_out"] / n, mean_inside=s["sum_in"] / n,
                                 render_offsets=dict(outside=plan[k]["render_offset_outside"],
                                                     inside=plan[k]["render_offset_inside"]))
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"threshold {thr:.0f} (Otsu); {a.step:g} voxel outside / inside the face at {a.face:+g}:")
    for k, r in res["renders"].items():
        print(f"  {k}: gap outside {r['gap_outside']:.1%}, papyrus inside {r['papyrus_inside']:.1%}, "
              f"both {r['both']:.1%}, brighter inside {r['brighter_inside']:.1%} "
              f"(means {r['mean_inside']:.0f} inside, {r['mean_outside']:.0f} outside) over {r['pixels']:,} pixels")


if __name__ == "__main__":
    main()
