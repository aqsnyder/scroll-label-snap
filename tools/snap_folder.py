"""Snap every cube in a folder laid out like the surface-label dataset, in parallel.

  IN_DIR/images/sample_NNNNN.tif + IN_DIR/labels/sample_NNNNN.tif
  -> OUT_DIR/labels/sample_NNNNN.tif, OUT_DIR/confidence/sample_NNNNN.tif,
     OUT_DIR/flags/sample_NNNNN.tif (labels on a face of the opposite polarity),
     OUT_DIR/summary.csv

Usage (Windows or Linux):
  python tools/snap_folder.py IN_DIR OUT_DIR [--workers N] [--limit K]

Each worker needs about 3 GB of RAM. Cubes already present in OUT_DIR are skipped,
so an interrupted run can be resumed.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run_one(img_path, lab_path, out_dir):
    os.environ["SNAP_THREADS"] = "1"          # one process per core already
    import numpy as np
    import tifffile
    from scroll_label_snap.core import Params, snap, shift_volume

    name = os.path.basename(img_path).rsplit(".", 1)[0]
    t0 = time.time()
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    res = snap(img, lab, Params())
    conf = shift_volume(lab.shape, res.labels == 1, res.pts, (res.confidence * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(out_dir, "labels", name + ".tif"), res.labels, compression="zlib")
    tifffile.imwrite(os.path.join(out_dir, "confidence", name + ".tif"), conf, compression="zlib")
    flags = shift_volume(lab.shape, res.labels == 1, res.pts, (res.reversed_face * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(out_dir, "flags", name + ".tif"), flags, compression="zlib")
    i = res.info
    return dict(cube=name, samples=i["n_samples"], sheets=i["n_sheets"],
                mean_abs_shift=round(i["mean_abs_shift"], 3),
                frac_moved_1vox=round(i["frac_moved_1vox"], 3),
                frac_flagged=round(i["frac_reversed_face"], 3),
                mean_confidence=round(float(np.mean(res.confidence[res.movable])), 3)
                if res.movable.any() else 0.0,
                seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first K cubes")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "confidence"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "flags"), exist_ok=True)
    names = sorted(f for f in os.listdir(os.path.join(a.in_dir, "images")) if f.endswith(".tif"))
    names = [f for f in names if os.path.exists(os.path.join(a.in_dir, "labels", f))]
    todo = [f for f in names if not os.path.exists(os.path.join(a.out_dir, "labels", f))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(names)} cube pairs found, {len(todo)} to process, {a.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, os.path.join(a.in_dir, "images", f),
                          os.path.join(a.in_dir, "labels", f), a.out_dir): f for f in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                rows.append(r)
                print(f"[{k}/{len(todo)}] {r['cube']}: moved >=1 vox {r['frac_moved_1vox']:.0%}, "
                      f"mean confidence {r['mean_confidence']:.2f}, {r['seconds']} s", flush=True)
            except Exception as e:  # keep going; report at the end
                print(f"[{k}/{len(todo)}] {futs[fut]}: FAILED: {e}", flush=True)

    csv_path = os.path.join(a.out_dir, "summary.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cube", "samples", "sheets", "mean_abs_shift",
                                          "frac_moved_1vox", "frac_flagged", "mean_confidence", "seconds"])
        if new:
            w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["cube"]))
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
