"""Run the evaluation from tools/eval_snap.py on every cube pair in a folder, in parallel.

Usage: python tools/eval_folder.py IN_DIR OUT_DIR [--workers 12] [--limit N]

IN_DIR holds images/ and labels/ with matching file names. Each cube's result goes to
OUT_DIR/cubes/<name>.json as soon as it finishes, so an interrupted run resumes where it
stopped. Progress goes to OUT_DIR/progress.log. At the end, OUT_DIR/eval.json combines all
cubes in the format tools/summarize_eval.py reads. Cube k (in name order) uses drift seed k,
as in eval_snap.py. Each worker needs about 3 GB of RAM.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(img_path, lab_path, out_json, seed):
    os.environ["SNAP_THREADS"] = "1"          # one process per core already
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, HERE)
    from scroll_label_snap.core import Params
    from eval_snap import evaluate
    r = evaluate(img_path, lab_path, Params(), seed)
    with open(out_json + ".tmp", "w") as f:
        json.dump(r, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, ".."))
    from scroll_label_snap.core import Params
    img_dir, lab_dir = os.path.join(a.in_dir, "images"), os.path.join(a.in_dir, "labels")
    cube_dir = os.path.join(a.out_dir, "cubes")
    os.makedirs(cube_dir, exist_ok=True)
    logf = os.path.join(a.out_dir, "progress.log")

    def log(msg):
        print(msg, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    names = sorted(f[:-4] for f in os.listdir(img_dir)
                   if f.endswith(".tif") and os.path.exists(os.path.join(lab_dir, f)))
    if a.limit:
        names = names[:a.limit]
    todo = [(k, n) for k, n in enumerate(names) if not os.path.exists(os.path.join(cube_dir, n + ".json"))]
    log(f"{len(names)} cube pairs, {len(todo)} to evaluate, {a.workers} workers")
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, os.path.join(img_dir, n + ".tif"), os.path.join(lab_dir, n + ".tif"),
                          os.path.join(cube_dir, n + ".json"), k): n for k, n in todo}
        for fut in as_completed(futs):
            n = futs[fut]
            done += 1
            try:
                r = fut.result()
                d = r["drift_test"]
                log(f"[{done}/{len(todo)}] {n}: {r['seconds']:.0f} s, trusted {r['n_trusted']}, within 1 voxel: "
                    f"no correction {d['drifted (no correction)']['within_1']:.0%}, snapper {d['snapper']['within_1']:.0%}"
                    f" | elapsed {(time.time() - t0) / 60:.0f} min")
            except Exception as e:                       # keep going; the cube can be rerun later
                log(f"[{done}/{len(todo)}] {n}: FAILED {e!r}")
    cubes = [json.load(open(os.path.join(cube_dir, n + ".json"))) for n in names
             if os.path.exists(os.path.join(cube_dir, n + ".json"))]
    with open(os.path.join(a.out_dir, "eval.json"), "w") as f:
        json.dump(dict(params=Params().asdict(), cubes=cubes), f, indent=2)
    log(f"wrote eval.json with {len(cubes)} of {len(names)} cubes")


if __name__ == "__main__":
    main()
