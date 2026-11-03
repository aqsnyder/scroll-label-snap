"""Helpers for the depth diagnostic (results/depth_sweep_plan.md): how strongly does the ink map
depend on where the surface sits, and does the direction of the snapper's move matter?

  python tools/depth_sweep.py mirror OUT_TIFXYZ --original TIFXYZ --snapped TIFXYZ
      The snapper's move the other way: 2 x original - snapped at every vertex valid in both (a
      drop-in output shares the original's grid). Other vertices are left out.

  python tools/depth_sweep.py negate OUT_NPZ IN_NPZ
      A depth file from tools/depth_transfer.py with the depth negated (the same move the other
      way along the target's normals); everything else is copied.

  python tools/depth_sweep.py compare OUT_JSON --reference NAME=TIF --map NAME=TIF [--map ...]
                                       --renders ZARR [ZARR ...] [--tile 256] [--min-valid 0.5]
      Each map against the reference, over the pixels valid in every render: whether it is the
      same map, the Pearson r, the whole-map mean and 99th percentile, and the ink test's tile
      measures (the change in each 256-pixel tile's mean and 99th percentile; the share of tiles
      losing or gaining more than 0.02). A map whose file is missing is reported as missing.

A uniform shift needs no new render: the model reads the middle 17 of a render's 21 layers
(layers 2-18), and villa's --layer-start 2+d --layer-end 19+d reads the stack d layers further
along the normal instead.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from ink_scores import load_map, tile_rows, valid_both  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402


def mirrored(P_orig, v_orig, P_snap, v_snap):
    valid = v_orig & v_snap
    M = np.where(valid[..., None], 2.0 * P_orig.astype(np.float64) - P_snap.astype(np.float64), -1.0)
    return M.astype(np.float32), valid


def cmd_mirror(a):
    Po, vo, meta = read_tifxyz(a.original)
    Ps, vs, _ = read_tifxyz(a.snapped)
    if Po.shape != Ps.shape:
        sys.exit(f"the grids differ: {Po.shape[:2]} and {Ps.shape[:2]} (use the drop-in output)")
    M, valid = mirrored(Po, vo, Ps, vs)
    d = np.linalg.norm((Ps - Po)[valid].astype(np.float64), axis=-1)
    write_tifxyz(a.out_tifxyz, M, valid, meta)
    print(f"{int(valid.sum()):,} vertices; the snapper moved them {d.mean():.2f} voxels on average "
          f"(median {np.median(d):.2f}); the mirror moves them as far the other way -> {a.out_tifxyz}")


def cmd_negate(a):
    z = np.load(a.in_npz, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["depth"] = -out["depth"]
    if "stats" in out:
        st = json.loads(str(out["stats"]))
        st["negated_from"] = a.in_npz
        out["stats"] = np.array(json.dumps(st))
    np.savez_compressed(a.out_npz, **out)
    d = out["depth"][np.isfinite(out["depth"])]
    print(f"negated {d.size:,} depths (now mean {d.mean():+.2f}, mean |depth| {np.abs(d).mean():.2f}) -> {a.out_npz}")


def pearson(x, y, band=1 << 22):
    n = sx = sy = sxx = syy = sxy = 0.0
    for i in range(0, x.size, band):
        a, b = x[i:i + band].astype(np.float64), y[i:i + band].astype(np.float64)
        n += a.size
        sx += a.sum()
        sy += b.sum()
        sxx += (a * a).sum()
        syy += (b * b).sum()
        sxy += (a * b).sum()
    if n == 0:
        return None
    cov = sxy - sx * sy / n
    var = (sxx - sx * sx / n) * (syy - sy * sy / n)
    return float(cov / np.sqrt(var)) if var > 0 else None


def compare_one(ref, m, valid, tile, min_valid):
    r, x = ref[valid], m[valid]
    rows = tile_rows(ref, m, valid, tile, min_valid)
    d99 = np.array([t["d_p99"] for t in rows])
    dm = np.array([t["d_mean"] for t in rows])
    out = dict(identical=bool(np.array_equal(ref, m)), max_abs_difference=int(np.abs(ref.astype(np.int16) - m.astype(np.int16)).max()),
               pearson_r=pearson(r, x),
               whole_map=dict(mean_reference=float(r.mean() / 255), mean=float(x.mean() / 255),
                              p99_reference=float(np.percentile(r, 99) / 255), p99=float(np.percentile(x, 99) / 255)),
               tiles=len(rows))
    if rows:
        out.update(d_p99_median=float(np.median(d99)), d_mean_median=float(np.median(dm)),
                   abs_d_p99_median=float(np.median(np.abs(d99))),
                   share_tiles_d_p99_below_minus_0_02=float(np.mean(d99 < -0.02)),
                   share_tiles_d_p99_above_0_02=float(np.mean(d99 > 0.02)),
                   d_p99_percentiles={str(q): float(np.percentile(d99, q)) for q in (5, 25, 50, 75, 95)})
    return out


def cmd_compare(a):
    rname, rpath = a.reference.split("=", 1)
    ref = load_map(rpath)
    valid = valid_both(a.renders, shape=ref.shape)
    res = dict(reference=dict(name=rname, path=rpath), renders=a.renders, pixels=int(valid.sum()), tile=a.tile,
               min_valid=a.min_valid, maps={})
    for spec in a.map:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            res["maps"][name] = dict(path=path, missing=True)
            print(f"{name}: missing ({path})")
            continue
        m = load_map(path)
        if m.shape != ref.shape:
            sys.exit(f"map {name} is {m.shape}, the reference {ref.shape}")
        c = dict(path=path, **compare_one(ref, m, valid, a.tile, a.min_valid))
        res["maps"][name] = c
        if c["identical"]:
            print(f"{name}: identical to {rname}")
        else:
            print(f"{name}: r {c['pearson_r']:.4f}; mean {c['whole_map']['mean']:.4f} (reference {c['whole_map']['mean_reference']:.4f}); "
                  f"tiles: median change in the 99th percentile {c.get('d_p99_median', float('nan')):+.4f}, "
                  f"{c.get('share_tiles_d_p99_below_minus_0_02', float('nan')):.0%} lose and "
                  f"{c.get('share_tiles_d_p99_above_0_02', float('nan')):.0%} gain more than 0.02")
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mirror")
    p.add_argument("out_tifxyz")
    p.add_argument("--original", required=True)
    p.add_argument("--snapped", required=True)
    p = sub.add_parser("negate")
    p.add_argument("out_npz")
    p.add_argument("in_npz")
    p = sub.add_parser("compare")
    p.add_argument("out_json")
    p.add_argument("--reference", required=True)
    p.add_argument("--map", action="append", required=True)
    p.add_argument("--renders", nargs="+", required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    dict(mirror=cmd_mirror, negate=cmd_negate, compare=cmd_compare)[a.cmd](a)


if __name__ == "__main__":
    main()
