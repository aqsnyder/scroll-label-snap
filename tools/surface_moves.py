"""How a snapped surface moved its original: size, net shift and roughness of the moves, from the two
tifxyz files alone (no CT). The snapped surface must share the original's vertex grid, as a drop-in
output or the release of results/finesnap_ink_plan.md does.

  python tools/surface_moves.py OUT_JSON --original TIFXYZ --snapped TIFXYZ [--compare TIFXYZ]
                                [--codes NAME=TIF ...] [--blocks NAME]

Moves are taken along the original's vertex normals, as tools/depth_sweep.py does. Reported:
- the grid and the vertices with a position in each file;
- the moves' mean (a net shift of the whole surface), standard deviation, and the median, 90th and
  99th percentile of their size, with the shares over 5 and 10 voxels;
- the correlation between neighbouring vertices' moves (1 is smooth, 0 is rough at the vertex scale);
- the moves' tangential part, which should be about 0 for moves along the normal;
- their large-scale part (a Gaussian of 1 vertex) and the rest;
- with --compare, another snap of the same original: its moves and their correlation with these;
- with --codes, per-vertex code images such as finesnap's abstain.tif: how many of the original's
  vertices carry each code, and the median move size under each;
- with --blocks NAME, the connected blocks of the vertices coded 1 in that code image (held back).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from depth_sweep import lag1, low_pass, move_field  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402


def moves(P, vp, S, vs):
    d, N, valid = move_field(P, vp, S, vs)
    if not valid.any():
        return None, d, N, valid
    a = np.abs(d[valid])
    lp = low_pass(d, valid, 1.0)
    tang = np.linalg.norm(((S.astype(np.float64) - P.astype(np.float64)) - d[..., None] * N)[valid], axis=-1)
    st = dict(vertices=int(valid.sum()), mean=float(d[valid].mean()), sd=float(d[valid].std()),
              mean_abs=float(a.mean()), median_abs=float(np.median(a)),
              p90_abs=float(np.percentile(a, 90)), p99_abs=float(np.percentile(a, 99)),
              share_abs_over_5=float((a > 5).mean()), share_abs_over_10=float((a > 10).mean()),
              neighbour_r=lag1(d, valid), tangential_mean=float(tang.mean()),
              smoothed_1_vertex=dict(sd=float(lp[valid].std()), neighbour_r=lag1(lp, valid)),
              rest_sd=float((d - lp)[valid].std()))
    return st, d, N, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_json")
    ap.add_argument("--original", required=True)
    ap.add_argument("--snapped", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--codes", action="append", default=[], metavar="NAME=TIF")
    ap.add_argument("--blocks")
    a = ap.parse_args()
    P, vp, _ = read_tifxyz(a.original)
    S, vs, _ = read_tifxyz(a.snapped)
    if P.shape != S.shape:
        sys.exit(f"the grids differ: {P.shape[:2]} and {S.shape[:2]}")
    res = dict(original=a.original, snapped=a.snapped, grid=list(P.shape[:2]), original_vertices=int(vp.sum()),
               snapped_vertices=int(vs.sum()), snapped_where_original_has_none=int((vs & ~vp).sum()))
    st, d, _, valid = moves(P, vp, S, vs)
    res["moves"] = st
    if a.compare:
        C, vc, _ = read_tifxyz(a.compare)
        if C.shape != P.shape:
            sys.exit(f"--compare's grid differs: {C.shape[:2]}")
        sc, dc, _, valid_c = moves(P, vp, C, vc)
        both = valid & valid_c
        if sc and both.sum() > 2:
            sc["correlation_with_snapped"] = float(np.corrcoef(d[both], dc[both])[0, 1])
            sc["correlation_smoothed_1_vertex"] = float(np.corrcoef(low_pass(d, valid, 1.0)[both],
                                                                    low_pass(dc, valid_c, 1.0)[both])[0, 1])
        res["compare"] = dict(path=a.compare, moves=sc)
    if a.codes:
        import tifffile
        res["codes"] = {}
        for spec in a.codes:
            name, path = spec.split("=", 1)
            code = tifffile.imread(path)
            if code.shape != vp.shape:
                sys.exit(f"{name} is {code.shape}, the grid {vp.shape}")
            vals, counts = np.unique(code[vp], return_counts=True)
            entry = dict(path=path, counts={str(int(k)): int(n) for k, n in zip(vals, counts)}, median_abs_move={})
            for k in vals:
                sel = valid & (code == k)
                if sel.any():
                    entry["median_abs_move"][str(int(k))] = float(np.median(np.abs(d[sel])))
            res["codes"][name] = entry
            if name == a.blocks:
                from scipy import ndimage as ndi
                lab, n = ndi.label(vp & (code == 1))
                sizes = np.bincount(lab.ravel())[1:]
                res["blocks"] = dict(code=name, n=int(n), largest=int(sizes.max()) if n else 0,
                                     sizes=sorted((int(s) for s in sizes), reverse=True)[:20])
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1)
    if st:
        print(f"{st['vertices']:,} vertices moved: mean {st['mean']:+.2f}, sd {st['sd']:.2f}, median |.| {st['median_abs']:.2f}, "
              f"p90 {st['p90_abs']:.2f}, over 10 voxels {st['share_abs_over_10']:.1%}, neighbour r {st['neighbour_r']:.2f} -> {a.out_json}")


if __name__ == "__main__":
    main()
