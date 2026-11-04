"""Snap every cube in a folder laid out like the surface-label dataset, in parallel.

  IN_DIR/images/sample_NNNNN.tif + IN_DIR/labels/sample_NNNNN.tif
  -> OUT_DIR/labels/sample_NNNNN.tif, OUT_DIR/confidence/sample_NNNNN.tif,
     OUT_DIR/flags/sample_NNNNN.tif (labels on a face of the opposite polarity),
     OUT_DIR/summary.csv

Usage (Windows or Linux):
  python tools/snap_folder.py IN_DIR OUT_DIR [--workers N] [--limit K] [--param NAME=VALUE ...]

Each worker needs about 3 GB of RAM. Cubes with all three outputs in OUT_DIR are skipped, so an
interrupted run can be resumed: each file is written under a temporary name and renamed into place,
and each finished cube leaves its summary row in OUT_DIR/rows/ before its labels appear. The exit
code is non-zero if any cube failed. A summary.csv held open by another program does not count as
a failure: the rows are safe in rows/, and the next run writes it. An OUT_DIR that is IN_DIR, or
whose labels have no confidence or flags beside them, is refused: its labels would be overwritten.

--param overrides a snapper parameter (see scroll_label_snap/core.py Params), e.g. keep_topology=true.
The overrides are kept in OUT_DIR/params.json, and a rerun refuses to mix cubes made with other ones.
A run without --param writes no such file.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUTPUTS = ("labels", "confidence", "flags")
COLUMNS = ["cube", "samples", "sheets", "mean_abs_shift", "frac_moved_1vox", "frac_flagged", "mean_confidence", "seconds"]


def write_tif(path, data):
    """A file in place is a complete file: write it under a temporary name, then rename it."""
    import tifffile
    tifffile.imwrite(path + ".tmp", data, compression="zlib")
    try:
        os.replace(path + ".tmp", path)
    except OSError:                           # another program holds the file
        os.remove(path + ".tmp")
        raise


def run_one(img_path, lab_path, out_dir, overrides=None):
    os.environ["SNAP_THREADS"] = "1"          # one process per core already
    import numpy as np
    import tifffile
    from scroll_label_snap.core import Params, snap, shift_volumes

    name = os.path.basename(img_path).rsplit(".", 1)[0]
    t0 = time.time()
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    res = snap(img, lab, Params(**(overrides or {})))
    conf, flags = shift_volumes(lab.shape, res.labels == 1, res.pts,
                                [(res.confidence * 255).astype(np.uint8), (res.reversed_face * 255).astype(np.uint8)])
    write_tif(os.path.join(out_dir, "confidence", name + ".tif"), conf)
    write_tif(os.path.join(out_dir, "flags", name + ".tif"), flags)
    i = res.info
    row = dict(cube=name, samples=i["n_samples"], sheets=i["n_sheets"],
               mean_abs_shift=round(i["mean_abs_shift"], 3),
               frac_moved_1vox=round(i["frac_moved_1vox"], 3),
               frac_flagged=round(i["frac_reversed_face"], 3),
               mean_confidence=round(float(np.mean(res.confidence[res.movable])), 3)
               if res.movable.any() else 0.0,
               seconds=round(time.time() - t0, 1))
    # the row goes first and the labels last, so a cube with labels always has its row, even if
    # this run's main process is gone by the time the cube finishes
    row_path = os.path.join(out_dir, "rows", name + ".json")
    os.makedirs(os.path.dirname(row_path), exist_ok=True)
    with open(row_path + ".tmp", "w") as f:
        json.dump(row, f)
    os.replace(row_path + ".tmp", row_path)
    write_tif(os.path.join(out_dir, "labels", name + ".tif"), res.labels)
    return row


def read_rows(out_dir):
    """Summary rows so far, by cube: those in summary.csv, then the ones the workers left in rows/. A summary.csv that
    can't be read, saved by a spreadsheet in another form, say, is rebuilt from rows/."""
    rows = {}
    csv_path = os.path.join(out_dir, "summary.csv")
    row_dir = os.path.join(out_dir, "rows")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    rows[r["cube"]] = {k: r.get(k) for k in COLUMNS}
        except (KeyError, OSError, UnicodeError, csv.Error) as e:
            rows = {}
            print(f"warning: can't read {csv_path} ({type(e).__name__}: {e}), so it is rebuilt from {row_dir}", flush=True)
    for name in os.listdir(row_dir):
        if name.endswith(".json"):
            with open(os.path.join(row_dir, name)) as f:
                r = json.load(f)
            rows[r["cube"]] = r
    return rows


def write_summary(out_dir, rows):
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows[c] for c in sorted(rows))
    try:
        os.replace(csv_path + ".tmp", csv_path)
    except OSError:
        os.remove(csv_path + ".tmp")
        raise
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first K cubes")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a snapper parameter (see scroll_label_snap/core.py Params), e.g. keep_topology=true")
    a = ap.parse_args()

    from scroll_label_snap.core import parse_params
    overrides = parse_params(a.param)

    in_labels, out_labels = os.path.join(a.in_dir, "labels"), os.path.join(a.out_dir, "labels")
    if not (os.path.isdir(os.path.join(a.in_dir, "images")) and os.path.isdir(in_labels)):
        sys.exit(f"{a.in_dir} does not hold images/ and labels/")
    if os.path.isdir(out_labels) and os.path.samefile(in_labels, out_labels):
        sys.exit(f"{a.out_dir} is also the input: its labels would be overwritten; use a new folder")
    for sub in OUTPUTS + ("rows",):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    done = os.listdir(out_labels)
    if done and not os.listdir(os.path.join(a.out_dir, "confidence")) and not os.listdir(os.path.join(a.out_dir, "flags")):
        sys.exit(f"{out_labels} holds labels with no confidence or flags beside them, so they are not from this tool, "
                 "and they would be overwritten; use a new folder")
    params_path = os.path.join(a.out_dir, "params.json")
    given = json.load(open(params_path)) if os.path.exists(params_path) else {}
    if given != overrides and done:
        sys.exit(f"{a.out_dir} holds cubes made with other --param settings ({given}); use a new folder")
    names = sorted(f for f in os.listdir(os.path.join(a.in_dir, "images")) if f.endswith(".tif"))
    names = [f for f in names if os.path.exists(os.path.join(a.in_dir, "labels", f))]
    todo = [f for f in names if not all(os.path.exists(os.path.join(a.out_dir, sub, f)) for sub in OUTPUTS)]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(names)} cube pairs found, {len(todo)} to process, {a.workers} workers", flush=True)
    if todo and given != overrides:           # no labels yet, so whatever an earlier try left there made nothing
        if overrides:
            json.dump(overrides, open(params_path, "w"), indent=1)
        else:
            os.remove(params_path)

    rows = read_rows(a.out_dir)
    failed = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, os.path.join(a.in_dir, "images", f),
                          os.path.join(a.in_dir, "labels", f), a.out_dir, overrides): f for f in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append(futs[fut])
                print(f"[{k}/{len(todo)}] {futs[fut]}: FAILED: {e}", flush=True)
                continue
            rows[r["cube"]] = r
            try:
                write_summary(a.out_dir, rows)
            except OSError:         # held open by another program; the write below says so, once
                pass
            print(f"[{k}/{len(todo)}] {r['cube']}: moved >=1 vox {r['frac_moved_1vox']:.0%}, "
                  f"mean confidence {r['mean_confidence']:.2f}, {r['seconds']} s", flush=True)

    try:
        print("wrote", write_summary(a.out_dir, rows))
    except OSError as e:
        print(f"warning: can't write summary.csv ({e}). The cubes are finished and their rows are in rows/; "
              "close the file and run the same command again to write it", flush=True)
    if failed:
        sys.exit(f"{len(failed)} of {len(todo)} cubes failed: {', '.join(sorted(failed))}; run the same command again to retry them")


if __name__ == "__main__":
    main()
