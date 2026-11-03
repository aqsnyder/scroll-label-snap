"""Check the ink-detection environment before the pre-registered ink test (results/ink_test_plan.md).

Run with the ink environment's Python (3.14, tools/requirements-ink.txt) and villa's vesuvius/src on
PYTHONPATH:

  python -P tools/ink_selfcheck.py OUT_JSON --checkpoint CKPT [--checkpoint CKPT ...] [--villa-commit SHA]
                                   [--work DIR]

It
  1. records the versions (Python, torch and its thread count, numpy, zarr, timm, the villa commit);
  2. imports the official flat inference, vesuvius.ink_detection.inference.infer;
  3. reads each checkpoint's embedded training config and lists the training segments. It stops
     (exit code 3) if w052, the test's held-out segment, is among them;
  4. runs the official inference end to end on a small random surface volume with the test's
     options, and checks the output: the right shape, and zero where the input is empty.

Exit code 0 when everything passes. Nothing here depends on the scan or the meshes.
"""
import argparse
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback


def walk_strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from walk_strings(k)
            yield from walk_strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from walk_strings(v)


def training_segments(config):
    """Segment names in the config's training datasets (both the names and the physical keys)."""
    names, physical = set(), set()
    for d in config.get("datasets", []) or []:
        for s in d.get("segments", []) or []:
            names.add(str(s))
        for v in (d.get("sampling_physical_segment_keys") or {}).values():
            physical.add(str(v))
    return sorted(names), sorted(physical)


def mentions(strings, segment):
    pat = re.compile(rf"(?<![0-9a-z]){re.escape(segment)}(?![0-9])", re.I)
    return sorted({s for s in strings if pat.search(s)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_json")
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--villa-commit", default="")
    ap.add_argument("--held-out", default="w052")
    ap.add_argument("--work")
    a = ap.parse_args()
    rep = dict(ok=False, held_out=a.held_out, villa_commit=a.villa_commit, steps={})

    def save():
        with open(a.out_json, "w") as f:
            json.dump(rep, f, indent=1)

    # 1. versions
    import numpy as np
    import torch
    import zarr
    rep["versions"] = dict(python=sys.version.split()[0], platform=platform.platform(),
                           machine=platform.machine(), processor=platform.processor(),
                           cpu_count=os.cpu_count(), torch=torch.__version__,
                           torch_threads=torch.get_num_threads(), numpy=np.__version__,
                           zarr=zarr.__version__, cuda_available=bool(torch.cuda.is_available()))
    try:
        import timm
        rep["versions"]["timm"] = timm.__version__
    except Exception as e:  # the import of the inference below reports it properly
        rep["versions"]["timm"] = f"not importable: {e}"
    print("versions:", json.dumps(rep["versions"]), flush=True)

    # 2. the official inference imports
    try:
        from vesuvius.ink_detection.inference import infer
        rep["steps"]["import_infer"] = dict(ok=True, file=infer.__file__)
    except Exception:
        rep["steps"]["import_infer"] = dict(ok=False, error=traceback.format_exc())
        save()
        print(rep["steps"]["import_infer"]["error"], file=sys.stderr)
        sys.exit("the official inference doesn't import; see " + a.out_json)
    print("imported", infer.__file__, flush=True)

    # 3. training segments, from the config inside each checkpoint
    rep["checkpoints"] = []
    for ck in a.checkpoint:
        c = dict(path=ck, bytes=os.path.getsize(ck))
        try:  # plain data only, the safe way; the official loader itself uses weights_only=False
            payload = torch.load(ck, map_location="cpu", weights_only=True)
            c["loads_with_weights_only"] = True
        except Exception as e:
            c["loads_with_weights_only"] = False
            c["weights_only_error"] = str(e)[:300]
            from vesuvius.ink_detection.models.checkpoint import load_checkpoint
            payload = load_checkpoint(ck)
        cfg = payload.get("config")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        c["state_keys"] = sorted(k for k in payload if k != "config")
        c["model_type"] = cfg.get("model_type")
        c["patch_size"] = cfg.get("patch_size")
        c["normalization"] = (cfg.get("image_normalization") or {}).get("mode")
        names, physical = training_segments(cfg)
        c["training_segments"], c["training_physical_segments"] = names, physical
        c["config_keys"] = sorted(cfg)
        ds_strings = list(walk_strings(cfg.get("datasets", [])))
        c["held_out_mentions"] = mentions(ds_strings, a.held_out)
        c["held_out_mentions_anywhere"] = mentions(list(walk_strings(cfg)), a.held_out)
        if not cfg.get("datasets"):
            c["warning"] = "no 'datasets' in the embedded config, so the training segments can't be checked"
            print("WARNING:", c["warning"], flush=True)
        rep["checkpoints"].append(c)
        print(f"{ck}: {c['model_type']}, patch {c['patch_size']}, {c['normalization']}; "
              f"{len(names)} training segments: {', '.join(names)}", flush=True)
        if c["held_out_mentions"]:
            save()
            print(f"STOP: {a.held_out} appears in this checkpoint's training data: {c['held_out_mentions']}",
                  file=sys.stderr)
            sys.exit(3)

    # 4. end to end on a small random surface volume (left two thirds random, right third empty)
    work = a.work or tempfile.mkdtemp(prefix="ink_selfcheck_")
    os.makedirs(work, exist_ok=True)
    zpath = os.path.join(work, "random_surface_volume.zarr")
    out_tif = os.path.join(work, "random_prediction.tif")
    rng = np.random.default_rng(0)
    vol = np.zeros((21, 384, 384), np.uint8)
    vol[:, :, :256] = rng.integers(20, 200, size=(21, 384, 256), dtype=np.uint8)
    kw = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    z = zarr.open_array(zpath, mode="w", shape=vol.shape, chunks=(21, 128, 128), dtype="u1",
                        fill_value=0, **kw)
    z[:] = vol
    argv = [zpath, a.checkpoint[0], out_tif, "--device", "cpu", "--no-compile", "--overlap", "0.5",
            "--blend-mode", "hann", "--batch-size", "4", "--loader-threads", "2"]
    t0 = time.time()
    try:
        infer.main(argv)
        import tifffile
        p = tifffile.imread(out_tif)
        empty = p[:, 320:]          # only the last column of patches covers this, and it is empty
        filled = p[:, :256]
        covered = float((filled > 0).mean())
        ok = p.shape == (384, 384) and p.dtype == np.uint8 and int(empty.max()) == 0 and covered > 0.99
        rep["steps"]["end_to_end"] = dict(ok=bool(ok), seconds=round(time.time() - t0, 1), argv=argv,
                                          shape=list(p.shape), dtype=str(p.dtype),
                                          filled_covered=round(covered, 4), filled_min=int(filled.min()),
                                          filled_max=int(filled.max()), filled_mean=round(float(filled.mean()), 2),
                                          empty_max=int(empty.max()))
    except Exception:
        rep["steps"]["end_to_end"] = dict(ok=False, error=traceback.format_exc(), argv=argv)
    e2e = rep["steps"]["end_to_end"]
    print("end to end:", json.dumps({k: v for k, v in e2e.items() if k != "argv"}), flush=True)
    rep["ok"] = bool(e2e.get("ok"))
    save()
    if not rep["ok"]:
        sys.exit("the end-to-end check failed; see " + a.out_json)
    print("self-check passed ->", a.out_json, flush=True)


if __name__ == "__main__":
    main()
