"""Download parts of the Vesuvius Challenge open-data bucket by prefix, and turn label zarrs into TIFFs.

Usage:
  python tools/fetch_open_data.py OUT_DIR PREFIX [PREFIX ...] [--skip SUBSTRING ...] [--labels-to-tif]

Each PREFIX is a key prefix in the bucket (for example PHerc0841/segments/20260220213127-w00/mesh/).
Every object under it is listed (S3 ListObjectsV2) and downloaded into OUT_DIR under the same key,
skipping keys that contain any --skip substring and files already downloaded with the right size.
OUT_DIR/listing.json records every key and size seen, downloaded or not.

--labels-to-tif: for every downloaded zarr whose path contains "ink-labels", write each of its
levels as OUT_DIR/<zarr path>.level<L>.tif (uint8) and print its shape. Needs zarr 3 for zarr v3
stores (the ink environment has it).
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix, session):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = session.get(BUCKET + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.iter(NS + "Contents"):
            keys.append((c.find(NS + "Key").text, int(c.find(NS + "Size").text)))
        if root.findtext(NS + "IsTruncated") == "true":
            token = root.findtext(NS + "NextContinuationToken")
        else:
            return keys


def download(key, size, out_dir, session, tries=4):
    path = os.path.join(out_dir, *key.split("/"))
    if os.path.exists(path) and os.path.getsize(path) == size:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in range(tries):
        try:
            with session.get(f"{BUCKET}/{key}", stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, path)
            return True
        except Exception as e:  # retry with backoff; give up after the last try
            if k == tries - 1:
                raise
            print(f"  retry {key}: {e}", flush=True)
            time.sleep(2 ** k)


def labels_to_tif(out_dir):
    import numpy as np
    import tifffile
    import zarr
    done = []
    for dirpath, dirnames, _ in os.walk(out_dir):
        for d in list(dirnames):
            if d.endswith(".zarr") and "ink-labels" in dirpath.replace("\\", "/"):
                zpath = os.path.join(dirpath, d)
                root = zarr.open(zpath, mode="r")
                arrays = [("", root)] if hasattr(root, "shape") else sorted(root.arrays(), key=lambda kv: kv[0])
                for name, arr in arrays:
                    a = np.asarray(arr[:])
                    tif = f"{zpath}.level{name or 0}.tif"
                    tifffile.imwrite(tif, a.astype(np.uint8) if a.dtype != np.uint8 else a, compression="zlib")
                    vals = np.unique(a[:: max(1, a.shape[0] // 512)])[:8]
                    print(f"{tif}: shape {a.shape}, dtype {a.dtype}, values {vals.tolist()}", flush=True)
                    done.append(dict(tif=tif, shape=list(a.shape), dtype=str(a.dtype)))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("prefix", nargs="+")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--labels-to-tif", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "scroll-label-snap fetch_open_data"
    listing, n_new, n_bytes = {}, 0, 0
    for prefix in a.prefix:
        keys = list_keys(prefix, s)
        listing[prefix] = [dict(key=k, size=z) for k, z in keys]
        take = [(k, z) for k, z in keys if not any(x in k for x in a.skip)]
        print(f"{prefix}: {len(keys)} objects, {len(take)} to fetch ({sum(z for _, z in take) / 1e6:.1f} MB)", flush=True)
        for k, z in take:
            if download(k, z, a.out_dir, s):
                n_new += 1
                n_bytes += z
    json.dump(listing, open(os.path.join(a.out_dir, "listing.json"), "w"), indent=1)
    print(f"downloaded {n_new} files, {n_bytes / 1e6:.1f} MB -> {a.out_dir}", flush=True)
    if a.labels_to_tif:
        out = labels_to_tif(a.out_dir)
        json.dump(out, open(os.path.join(a.out_dir, "labels_tif.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
