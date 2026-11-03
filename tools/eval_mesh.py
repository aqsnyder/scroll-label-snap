"""Evaluate mesh snapping on patches fetched by tools/fetch_segment.py. The mesh analogue of
tools/eval_snap.py.

Usage:
  python tools/eval_mesh.py OUT_JSON PATCH_DIR [PATCH_DIR ...] [--seed 0]

Per patch:
  1. Drift test. Push the mesh off the surface with a known smooth random drift (std 1.5
     voxels, correlation length 40 voxels, zero median so the mesh's own convention does not
     change), applied to the vertices along their normals. Snap the drifted mesh, and run the
     port of VC3D's alpha-composite refiner on it. Measure how far each result lands from the
     original mesh, along the original normals, at trusted samples: where the original mesh
     sits on a strong, unambiguous papyrus edge (same rule as for labels). The refiner works to
     its own convention, so it is also scored after removing its median offset on the
     original mesh ("offset-corrected"), and with its opacity thresholds tuned on the same
     patches (a grid search; its most generous setting).
  2. Consistency at all samples: does snapping the drifted mesh land where snapping the
     original lands?
  3. How much snapping moves the original mesh.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import Params, alphacomp_shifts  # noqa: E402
from scroll_label_snap.mesh import patch_orientation, prepare_mesh, snap_mesh, trusted_dense  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz  # noqa: E402


def drift_field(rng, amp=1.5, corr=40.0, n_modes=64):
    """Smooth random scalar field of 3D position (std ~amp, correlation length ~corr voxels),
    the same construction as tools/eval_snap.py."""
    k = rng.normal(size=(n_modes, 3))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    k *= (2 * np.pi / corr) * rng.uniform(0.6, 1.4, size=(n_modes, 1))
    phi = rng.uniform(0, 2 * np.pi, n_modes)
    return lambda pts: amp * np.sqrt(2.0 / n_modes) * np.cos(pts @ k.T + phi).sum(1)


def stats(err):
    err = np.abs(err[np.isfinite(err)])
    if not len(err):
        return dict(mean=float("nan"), median=float("nan"), within_1=float("nan"), over_2=float("nan"), n=0)
    return dict(mean=float(err.mean()), median=float(np.median(err)), within_1=float(np.mean(err <= 1.0)),
                over_2=float(np.mean(err > 2.0)), n=int(len(err)))


def on_grid(prep, values):
    out = np.full(prep.vd.shape + np.shape(values)[1:], np.nan, np.float32)
    out[prep.vd] = values
    return out


BASELINE_GRID = [(118, 165)] + [(lo, hi) for lo in (50, 60, 70, 80, 90) for hi in (90, 100, 110, 125, 140)
                                 if hi >= lo + 20]


def evaluate(patch_dir, p, seed, drift_std=1.5, drift_corr=40.0):
    t0 = time.time()
    pj = json.load(open(os.path.join(patch_dir, "patch.json")))
    off, ls = np.asarray(pj["offset_zyx"], np.float32), float(pj.get("scale_to_level", 1.0))
    P, valid, _ = read_tifxyz(os.path.join(patch_dir, "segment.tifxyz"))
    ct = tifffile.imread(os.path.join(patch_dir, "volume.tif"))
    rng = np.random.default_rng(seed)

    prep0 = prepare_mesh(ct, P, valid, p, off, ls, normal_sign=patch_orientation(patch_dir))
    # drift the vertices along their (oriented) normals; zero median over the dense samples
    field = drift_field(rng, drift_std, drift_corr)
    Q = P.astype(np.float32) * ls - off
    d_dense = field(prep0.Pd[prep0.vd].astype(np.float64))
    d_v = np.clip(field(Q.reshape(-1, 3).astype(np.float64)).reshape(valid.shape) - np.median(d_dense), -3.5, 3.5)
    Pd = P + (d_v[..., None] * prep0.N) / ls
    Pd[~prep0.ok] = -1
    prep_d = prepare_mesh(ct, Pd, prep0.ok, p, off, ls, factor=prep0.factor, normal_sign=prep0.normal_sign,
                          ct_s=prep0.ct_s)

    X0, N0 = prep0.Pd, prep0.Nd                                   # original dense samples (crop coords)
    res0 = snap_mesh(ct, P, valid, p, off, ls, prep=prep0)
    res_d = snap_mesh(ct, Pd, prep0.ok, p, off, ls, prep=prep_d)
    as_crop = lambda r: r.dense_P * ls - off                      # noqa: E731
    base0 = alphacomp_shifts(ct, X0[prep0.vd], N0[prep0.vd])
    base_d = alphacomp_shifts(ct, prep_d.Pd[prep_d.vd], prep_d.Nd[prep_d.vd])
    base_off = float(np.median(base0))
    pos = {
        "drifted": prep_d.Pd,
        "snap(original)": as_crop(res0),
        "snap(drifted)": as_crop(res_d),
        "baseline(original)": X0 + on_grid(prep0, base0)[..., None] * N0,
        "baseline(drifted)": prep_d.Pd + on_grid(prep_d, base_d)[..., None] * prep_d.Nd,
        "baseline offset-corrected(drifted)": prep_d.Pd + (on_grid(prep_d, base_d) - base_off)[..., None] * prep_d.Nd,
    }
    both = prep0.vd & prep_d.vd

    def err(k, ref=X0):
        e = np.einsum("...k,...k->...", pos[k] - ref, N0)
        e[~both] = np.nan
        return e

    tr = np.zeros(prep0.vd.shape, bool)
    tr[prep0.vd] = trusted_dense(prep0, p)
    tr &= both
    # the refiner with other opacity thresholds (VC3D's defaults are 118 and 165 of 255)
    grid = {}
    for lo, hi in BASELINE_GRID:
        kw = dict(low=lo / 255, high=hi / 255)
        b0 = on_grid(prep0, alphacomp_shifts(ct, X0[prep0.vd], N0[prep0.vd], **kw))
        bd = on_grid(prep_d, alphacomp_shifts(ct, prep_d.Pd[prep_d.vd], prep_d.Nd[prep_d.vd], **kw))
        m0 = float(np.nanmedian(b0[prep0.vd]))
        e = np.einsum("...k,...k->...", prep_d.Pd + (bd - m0)[..., None] * prep_d.Nd - X0, N0)
        grid[f"{lo}-{hi}"] = dict(offset_corrected=stats(e[tr]), median_offset=m0)
    drift_res = {
        "drifted (no correction)": stats(err("drifted")[tr]),
        "snapper": stats(err("snap(drifted)")[tr]),
        "baseline": stats(err("baseline(drifted)")[tr]),
        "baseline, offset-corrected": stats(err("baseline offset-corrected(drifted)")[tr]),
        "snapper consistency": stats(err("snap(drifted)", pos["snap(original)"])[both]),
        "baseline consistency": stats(err("baseline(drifted)", pos["baseline(original)"])[both]),
    }
    moved = res0.dense_shift[prep0.vd] * ls
    name = os.path.basename(os.path.normpath(patch_dir))
    return dict(patch=name, seconds=round(time.time() - t0, 1), n_dense=int(prep0.vd.sum()),
                n_trusted=int(tr.sum()), factor=prep0.factor, vertex_spacing=prep0.spacing / ls,
                convention_offset=res0.info["convention_offset"], normal_sign=prep0.normal_sign,
                orientation_agree=prep0.orientation_agree, baseline_median_offset=base_off,
                drift=dict(std=drift_std, corr=drift_corr), drift_test=drift_res, baseline_grid=grid,
                original_mesh_moves=dict(mean_abs=float(np.abs(moved).mean()),
                                         frac_moved_1vox=float(np.mean(np.abs(moved) >= 1.0)),
                                         median_signed=float(np.median(moved)),
                                         frac_flagged=float(res0.dense_flag[prep0.vd].mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift-std", type=float, default=1.5)
    ap.add_argument("--drift-corr", type=float, default=40.0)
    a = ap.parse_args()
    p = Params()
    results = []
    for k, d in enumerate(a.patches):
        r = evaluate(d, p, a.seed + k, a.drift_std, a.drift_corr)
        results.append(r)
        t = r["drift_test"]
        print(f"{r['patch']}: {r['n_dense']} samples, {r['n_trusted']} trusted, edge at {r['convention_offset']:+.2f}, "
              f"{r['seconds']} s", flush=True)
        for m, s in t.items():
            print(f"   {m:28s} mean {s['mean']:.2f}  within 1: {s['within_1']:.0%}  over 2: {s['over_2']:.0%}", flush=True)
    keys = list(results[0]["drift_test"])
    pooled = {}
    for m in keys:
        rs = [r["drift_test"][m] for r in results if r["drift_test"][m]["n"] > 0]
        w = [r["n"] for r in rs]
        pooled[m] = {s: float(np.average([r[s] for r in rs], weights=w)) if rs else float("nan")
                     for s in ("mean", "within_1", "over_2")}
        pooled[m]["n"] = int(sum(w))
    # best opacity thresholds for the refiner, chosen on these same patches (its most generous setting)
    def pooled_grid(key):
        rs = [r["baseline_grid"][key]["offset_corrected"] for r in results if r["baseline_grid"][key]["offset_corrected"]["n"]]
        w = [r["n"] for r in rs]
        return {s: float(np.average([r[s] for r in rs], weights=w)) for s in ("mean", "within_1", "over_2")} | {"n": int(sum(w))}
    gp = {key: pooled_grid(key) for key in results[0]["baseline_grid"]}
    best = min(gp, key=lambda key: gp[key]["mean"])
    pooled[f"baseline, thresholds {best} (best on these patches), offset-corrected"] = gp[best]
    json.dump(dict(params=p.asdict(), seed=a.seed, drift=dict(std=a.drift_std, corr=a.drift_corr), pooled=pooled,
                   baseline_grid_pooled=gp, baseline_best=best, patches=results),
              open(a.out_json, "w"), indent=1)
    print("pooled:")
    for m, s in pooled.items():
        print(f"   {m:28s} mean {s['mean']:.2f}  within 1: {s['within_1']:.0%}  over 2: {s['over_2']:.0%}  (n={s['n']})")
    print("refiner thresholds (offset-corrected mean error):",
          ", ".join(f"{key}: {v['mean']:.2f}" for key, v in sorted(gp.items(), key=lambda kv: kv[1]["mean"])[:6]))


if __name__ == "__main__":
    main()
