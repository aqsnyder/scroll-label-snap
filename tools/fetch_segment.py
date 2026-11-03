"""Download a traced segment (tifxyz) from the Vesuvius Challenge open-data bucket, plus crops of
its scroll volume around a few patches of it, for developing and testing mesh snapping.

Usage:
  python tools/fetch_segment.py OUT_DIR --segment TIFXYZ_URL --volume OME_ZARR_URL
         [--patches 4] [--patch 24] [--margin 24] [--level 0] [--seed 0] [--max-side 900]

  TIFXYZ_URL    https URL of a .tifxyz folder (meta.json, x.tif, y.tif, z.tif, optional mask.tif)
  OME_ZARR_URL  https URL of the scroll volume the segment was traced on (OME-Zarr, zarr v2)

Writes:
  OUT_DIR/segment.tifxyz/         the whole segment, as downloaded
  OUT_DIR/patch_K/volume.tif      a crop of the volume around patch K (z, y, x order)
  OUT_DIR/patch_K/segment.tifxyz/ the patch's grid window, in the segment's own coordinates
  OUT_DIR/patch_K/patch.json      crop offset (z, y, x) and size, grid window, level, sources, and
                                  which side of the mesh is recto (recto_sign: +1 along the grid's
                                  own normal, -1 against it), from the whole segment's curvature

Patches are windows of --patch x --patch grid vertices with every vertex valid, picked at random
with --seed and kept apart. Needs tifffile, zarr<3, fsspec and aiohttp (plus requests for
some fsspec versions).
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.mesh import axis_orientation  # noqa: E402
from scroll_label_snap.tifxyz import read_tifxyz, write_tifxyz  # noqa: E402

TIFXYZ_FILES = ("meta.json", "x.tif", "y.tif", "z.tif", "mask.tif")


def http_get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def list_s3_folder(url):
    """Direct children of an S3 'folder' given as https://bucket.s3.amazonaws.com/prefix/."""
    u = urllib.parse.urlparse(url)
    prefix = u.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "delimiter": "/"})
    xml = http_get(f"{u.scheme}://{u.netloc}/?{q}")
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(xml)
    keys = [e.text for e in root.findall(".//s3:Contents/s3:Key", ns)]
    return [k[len(prefix):] for k in keys if k[len(prefix):] and "/" not in k[len(prefix):]]


def download_tifxyz(url, out):
    os.makedirs(out, exist_ok=True)
    base = url.rstrip("/") + "/"
    try:
        names = [n for n in list_s3_folder(base) if n in TIFXYZ_FILES]
    except Exception:                                    # not S3, or listing refused: try the usual names
        names = list(TIFXYZ_FILES)
    got = []
    for name in names:
        dst = os.path.join(out, name)
        if not os.path.exists(dst):
            try:
                data = http_get(base + name)
            except Exception:
                if name == "mask.tif":
                    continue
                raise
            with open(dst, "wb") as f:
                f.write(data)
        got.append(name)
    for need in ("x.tif", "y.tif", "z.tif"):
        if need not in got:
            sys.exit(f"{need} not found under {base}")
    return got


def pick_windows(valid, size, count, rng):
    """Top-left corners of fully valid size x size windows, spread apart."""
    c = np.pad(np.cumsum(np.cumsum(valid.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    full = (c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]) == size * size
    cand = np.argwhere(full)
    out = []
    for k in rng.permutation(len(cand)):
        i, j = cand[k]
        if all(max(abs(i - a), abs(j - b)) >= size for a, b in out):
            out.append((int(i), int(j)))
            if len(out) == count:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--patch", type=int, default=24, help="patch size in grid vertices")
    ap.add_argument("--margin", type=int, default=24, help="voxels around the patch's bounding box")
    ap.add_argument("--level", type=int, default=0, help="OME-Zarr level to crop (0 = full resolution)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=900, help="skip windows whose crop is longer than this")
    a = ap.parse_args()
    import fsspec
    import zarr

    seg_dir = os.path.join(a.out_dir, "segment.tifxyz")
    files = download_tifxyz(a.segment, seg_dir)
    print(f"segment: {', '.join(files)}", flush=True)
    P, valid, meta = read_tifxyz(seg_dir)
    print(f"grid {P.shape[1]} x {P.shape[0]}, {valid.mean():.0%} valid, scale {meta.get('scale')}", flush=True)
    # which way the recto faces, from the whole segment's geometry (recto faces the scroll axis)
    r_sign, r_agree, r_axis = axis_orientation(P, valid)
    print(f"scroll axis near y {r_axis[0]:.0f}, x {r_axis[1]:.0f}: the recto side is "
          f"{'along' if r_sign > 0 else 'against'} the grid normal at {r_agree:.0%} of vertices", flush=True)
    orient = dict(recto_agree=r_agree, scroll_axis_yx=list(r_axis))
    if r_agree >= 0.9:
        orient["recto_sign"] = r_sign

    vol = zarr.open_array(fsspec.get_mapper(a.volume.rstrip("/") + f"/{a.level}"), mode="r")
    print(f"volume level {a.level}: shape {vol.shape}, chunks {vol.chunks}, dtype {vol.dtype}", flush=True)
    f = 2 ** a.level
    rng = np.random.default_rng(a.seed)
    windows = pick_windows(valid, a.patch, a.patches * 4, rng)
    done = 0
    for (i, j) in windows:
        sl = (slice(i, i + a.patch), slice(j, j + a.patch))
        Q = P[sl].reshape(-1, 3) / f
        lo = np.floor(Q.min(0)).astype(int) - a.margin
        hi = np.ceil(Q.max(0)).astype(int) + a.margin + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(vol.shape[-3:]))
        if np.any(hi - lo > a.max_side) or np.any(hi <= lo):
            continue
        pdir = os.path.join(a.out_dir, f"patch_{done}")
        pj = os.path.join(pdir, "patch.json")
        if os.path.exists(pj) and os.path.exists(os.path.join(pdir, "volume.tif")):
            old = json.load(open(pj))
            if old.get("grid_window") == dict(row=i, col=j, size=a.patch) and old.get("offset_zyx") == lo.tolist():
                print(f"patch_{done}: already downloaded", flush=True)
                done += 1
                if done == a.patches:
                    break
                continue
        crop = np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        if (crop == 0).mean() > 0.5:                         # mostly outside the scan mask
            continue
        os.makedirs(pdir, exist_ok=True)
        tifffile.imwrite(os.path.join(pdir, "volume.tif"), crop, compression="zlib")
        m = dict(meta, note=f"grid window rows {i}-{i + a.patch - 1}, cols {j}-{j + a.patch - 1} of {a.segment}")
        write_tifxyz(os.path.join(pdir, "segment.tifxyz"), P[sl], valid[sl], m,
                     uuid=f"{meta.get('uuid', 'segment')}_patch{done}")
        json.dump(dict(offset_zyx=lo.tolist(), shape_zyx=list(crop.shape), level=a.level, scale_to_level=1 / f,
                       grid_window=dict(row=i, col=j, size=a.patch), dtype=str(crop.dtype),
                       segment=a.segment, volume=a.volume, **orient), open(pj, "w"), indent=1)
        print(f"patch_{done}: grid rows {i}-{i + a.patch - 1}, cols {j}-{j + a.patch - 1}; crop {crop.shape} "
              f"at {lo.tolist()}, {crop.nbytes / 1e6:.0f} MB", flush=True)
        done += 1
        if done == a.patches:
            break
    if done < a.patches:
        print(f"only found {done} usable patches", flush=True)
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
