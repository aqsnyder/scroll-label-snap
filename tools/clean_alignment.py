"""Add the fair refiner's row to the clean-label table of an existing evaluation.

eval_snap.py now scores the existing refiner on the clean labels twice: with VC3D's default
thresholds, and with thresholds from the cube's own gap and papyrus levels and its median offset
removed ("baseline adaptive(original)"). Evaluations made before that change only have the first.
This script computes the missing row for each cube of such an evaluation, without rerunning the
drift test, and writes a copy of the evaluation with the row added.

As a check it also recomputes the original-label and default-refiner rows, and stops if they
differ from the evaluation, so the new row comes from the same surface samples and normals.

Usage: python tools/clean_alignment.py IN_DIR EVAL_JSON OUT_JSON [--workers 12]

IN_DIR holds images/ and labels/ as for eval_folder.py. Each cube's result is cached in
OUT_JSON + ".parts/", so an interrupted run resumes.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "baseline adaptive(original)"
TOL = dict(mean_step=0.05, frac_positive=0.002, coverage=0.002)


def run_one(img_path, lab_path, out_json):
    os.environ["SNAP_THREADS"] = "1"
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, HERE)
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    from scroll_label_snap.core import Params, prepare, alphacomp_shifts, warp_labels
    from eval_snap import baseline, band_position, otsu_class_means, step_alignment

    t0 = time.time()
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    prep0 = prepare(img, lab, Params())
    pts0, n0 = prep0[1], prep0[2]
    lo, hi = 10, np.array(lab.shape) - 11
    inner = np.all((pts0 >= lo) & (pts0 <= hi), axis=1)
    # exactly as in eval_snap.evaluate
    ac_vol = ndi.gaussian_filter(img.astype(np.float32) / 255.0, 1.4)
    gap_level, pap_level = otsu_class_means(img)
    s0 = alphacomp_shifts(img, pts0, n0, low=gap_level / 255, high=pap_level / 255, vol=ac_vol)
    versions = {"original": lab,
                "baseline(original)": baseline(img, lab, prep0),
                KEY: warp_labels(lab, pts0, n0, s0 - float(np.median(s0)))}
    ct_e = ndi.gaussian_filter(img.astype(np.float32), 0.5)
    pi, ni = pts0[inner], n0[inner]
    r = {k: step_alignment(ct_e, pi, ni, band_position(v, pi, ni)) for k, v in versions.items()}
    r["seconds"] = round(time.time() - t0, 1)
    with open(out_json + ".tmp", "w") as f:
        json.dump(r, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("eval_json")
    ap.add_argument("out_json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    with open(a.eval_json) as f:
        ev = json.load(f)
    parts = a.out_json + ".parts"
    os.makedirs(parts, exist_ok=True)
    jobs = {}
    for c in ev["cubes"]:
        n = c["cube"]
        img = os.path.join(a.in_dir, "images", n + ".tif")
        lab = os.path.join(a.in_dir, "labels", n + ".tif")
        if not (os.path.exists(img) and os.path.exists(lab)):
            sys.exit(f"missing {img} or {lab}")
        jobs[n] = (img, lab, os.path.join(parts, n + ".json"))
    todo = [n for n, j in jobs.items() if not os.path.exists(j[2])]
    print(f"{len(jobs)} cubes, {len(todo)} to compute, {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, *jobs[n]): n for n in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            print(f"[{k}/{len(todo)}] {futs[fu]}: {r['seconds']} s, papyrus on the correct side: "
                  f"original {r['original']['frac_positive']:.0%}, refiner (own levels, offset removed) "
                  f"{r[KEY]['frac_positive']:.0%} | elapsed {(time.time() - t0) / 60:.0f} min", flush=True)

    worst = {k: 0.0 for k in TOL}
    for c in ev["cubes"]:
        with open(jobs[c["cube"]][2]) as f:
            r = json.load(f)
        for k in ("original", "baseline(original)"):
            for s, tol in TOL.items():
                d = abs(r[k][s] - c["edge_alignment"][k][s])
                worst[s] = max(worst[s], d)
                if d > tol:
                    sys.exit(f"{c['cube']}: recomputed {k} {s} = {r[k][s]:.4f}, but the evaluation has "
                             f"{c['edge_alignment'][k][s]:.4f}; not the same inputs, nothing written")
        c["edge_alignment"][KEY] = r[KEY]
    ev.setdefault("notes", []).append(f"edge_alignment[{KEY!r}] added by tools/clean_alignment.py; the "
                                      f"recomputed original and default-refiner rows matched within {worst}")
    with open(a.out_json, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"rows reproduced (largest differences {worst}); wrote {a.out_json}")


if __name__ == "__main__":
    main()
