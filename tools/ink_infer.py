"""Run the official flat ink inference (vesuvius.ink_detection.inference.infer from ScrollPrize/villa)
once, and keep a run log: the exact command, the villa commit, versions, the wall time and a summary
of the output. A finished run (its log says done and its output exists) is skipped, so a script that
calls this can be started again.

Run with the ink environment's Python, villa's vesuvius/src on PYTHONPATH:

  python -P tools/ink_infer.py LOG_JSON [--villa-commit SHA] -- INPUT_ZARR CHECKPOINT OUTPUT_TIF [infer options]

Everything after "--" goes to the inference unchanged. Use one direction per run (forward or
reverse), so each run's time is measured on its own.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone


def output_paths(infer_args):
    out = infer_args[2]
    if "--direction" in infer_args and infer_args[infer_args.index("--direction") + 1] == "both":
        stem, ext = os.path.splitext(out)
        return [out, f"{stem}_reverse{ext}"]
    return [out]


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("log_json")
    ap.add_argument("--villa-commit", default="")
    a = ap.parse_args(sys.argv[1:cut])
    infer_args = sys.argv[cut + 1:]
    if len(infer_args) < 3:
        sys.exit("give INPUT_ZARR CHECKPOINT OUTPUT_TIF after --")
    outs = output_paths(infer_args)
    if os.path.exists(a.log_json):
        old = json.load(open(a.log_json))
        if old.get("status") == "done" and old.get("argv") == infer_args and all(os.path.exists(p) for p in outs):
            print(f"already done ({old.get('seconds')} s): {', '.join(outs)}", flush=True)
            return
    import numpy as np
    import torch
    import zarr
    from vesuvius.ink_detection.inference import infer
    log = dict(status="running", argv=infer_args, villa_commit=a.villa_commit, infer_module=infer.__file__,
               python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
               torch_threads=torch.get_num_threads(), numpy=np.__version__, zarr=zarr.__version__,
               started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(os.path.abspath(a.log_json)), exist_ok=True)
    json.dump(log, open(a.log_json, "w"), indent=1)
    t0 = time.time()
    rc = infer.main(infer_args)
    log["seconds"] = round(time.time() - t0, 1)
    log["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    import tifffile
    log["outputs"] = []
    for p in outs:
        im = tifffile.imread(p)
        nz = im[im > 0]
        log["outputs"].append(dict(path=p, shape=list(im.shape), dtype=str(im.dtype),
                                   nonzero_fraction=round(float((im > 0).mean()), 4),
                                   nonzero_mean=round(float(nz.mean()), 2) if nz.size else None))
    log["status"] = "done" if rc in (0, None) else f"exit {rc}"
    json.dump(log, open(a.log_json, "w"), indent=1)
    print(f"{log['status']} in {log['seconds']} s -> {', '.join(outs)}", flush=True)
    if log["status"] != "done":
        sys.exit(1)


if __name__ == "__main__":
    main()
