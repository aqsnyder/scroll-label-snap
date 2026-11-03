"""Checks behind the README's note on drift draws in the label drift test.

  python tools/drift_draws.py seeds OLD_EVAL_JSON IN_DIR [--scheme half|index] [--cubes NAME ...]
      For each cube in an older evaluation, rebuild the drifted labels with the seed that the
      scheme gives cube k (0-based, in name order) and check that the drifted labels' score
      ("within 1 voxel", no correction) equals the old value exactly. 'half' is k // 2, which
      is what results/eval_v2.json used (two processes, each counting from 0); 'index' is k,
      as tools/eval_folder.py does.

  python tools/drift_draws.py diagnose IN_DIR CUBE --seed S [--param NAME=VALUE ...]
      Run the snapper on one cube's drifted labels and describe the trusted points that end
      more than 2 voxels off: where the drift was, which way they moved, how clustered they are.

IN_DIR holds images/ and labels/ as for eval_folder.py.
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from scroll_label_snap.core import Params, prepare, snap, warp_labels  # noqa: E402
from eval_snap import band_position, drift_field, trusted_samples  # noqa: E402


def load(in_dir, name, p):
    img = tifffile.imread(os.path.join(in_dir, "images", name + ".tif"))
    lab = tifffile.imread(os.path.join(in_dir, "labels", name + ".tif"))
    prep0 = prepare(img, lab, p)
    pts0, n0 = prep0[1], prep0[2]
    lo, hi = 10, np.array(lab.shape) - 11
    inner = np.all((pts0 >= lo) & (pts0 <= hi), axis=1)
    tr = trusted_samples(lab, prep0, p)[inner]
    return img, lab, prep0, inner, tr


def cmd_seeds(a):
    p = Params()
    with open(a.old_eval) as f:
        cubes = sorted(json.load(f)["cubes"], key=lambda c: c["cube"])
    bad = 0
    for k, c in enumerate(cubes):
        if a.cubes and c["cube"] not in a.cubes:
            continue
        seed = k // 2 if a.scheme == "half" else k
        img, lab, prep0, inner, tr = load(a.in_dir, c["cube"], p)
        pts0, n0 = prep0[1], prep0[2]
        o = band_position(lab, pts0[inner], n0[inner])
        lab_d = warp_labels(lab, pts0, n0, drift_field(pts0, 1.5, 40.0, np.random.default_rng(seed)))
        e = np.abs(band_position(lab_d, pts0[inner], n0[inner]) - o)[tr]
        w = float(np.mean(e[np.isfinite(e)] <= 1.0))
        old = c["drift_test"]["drifted (no correction)"]["within_1"]
        same = w == old
        bad += not same
        print(f"{c['cube']} seed {seed}: {'same' if same else 'DIFFERENT'} ({w:.4f} vs {old:.4f})", flush=True)
    print("all the same" if not bad else f"{bad} differ")


def cmd_diagnose(a):
    p = Params()
    for kv in a.param:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        p = dataclasses.replace(p, **{k: (v == "True") if isinstance(cur, bool) else type(cur)(float(v))})
    img, lab, prep0, inner, tr = load(a.in_dir, a.cube, p)
    pts0, n0 = prep0[1], prep0[2]
    o = band_position(lab, pts0[inner], n0[inner])
    d = drift_field(pts0, 1.5, 40.0, np.random.default_rng(a.seed))
    lab_d = warp_labels(lab, pts0, n0, d)
    res = snap(img, lab_d, p, prepared=prepare(img, lab_d, p))
    q = band_position(res.labels, pts0[inner], n0[inner])
    qd = band_position(lab_d, pts0[inner], n0[inner])
    err, dpos = (q - o)[tr], (qd - o)[tr]
    ok = np.isfinite(err) & np.isfinite(dpos)
    err, dpos, P = err[ok], dpos[ok], pts0[inner][tr][ok]
    far = np.abs(err) > 2
    print(f"{a.cube}, seed {a.seed}, params {a.param or 'default'}: {ok.sum()} trusted points, "
          f"within 1 voxel {np.mean(np.abs(err) <= 1):.3f}, over 2 voxels {far.mean():.3f}")
    if not far.any():
        return
    print(f"  drift there: median |drift| {np.median(np.abs(dpos[far])):.2f} (elsewhere {np.median(np.abs(dpos[~far])):.2f})")
    print(f"  error with the drift's sign: {np.mean(np.sign(err[far]) == np.sign(dpos[far])):.2f}; "
          f"signed error 10/50/90th percentile: {np.round(np.percentile(err[far], [10, 50, 90]), 2).tolist()} "
          f"(+ is toward the gap)")
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(P[far]).query_pairs(6.0, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(far.sum(), far.sum()))
    nc, lab_c = connected_components(A, directed=False)
    sizes = np.sort(np.bincount(lab_c))[::-1]
    print(f"  {far.sum()} such points in {nc} patches (points within 6 voxels joined); "
          f"the 3 largest hold {sizes[:3].sum() / far.sum():.0%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("old_eval")
    s.add_argument("in_dir")
    s.add_argument("--scheme", choices=("half", "index"), default="half")
    s.add_argument("--cubes", nargs="*", help="only these cubes (default: all)")
    g = sub.add_parser("diagnose")
    g.add_argument("in_dir")
    g.add_argument("cube")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--param", action="append", default=[])
    a = ap.parse_args()
    cmd_seeds(a) if a.cmd == "seeds" else cmd_diagnose(a)


if __name__ == "__main__":
    main()
