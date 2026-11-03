"""Check a rendered surface volume against a reference one (e.g. the organisers' official render of
the same mesh): find the in-plane shift and the layer alignment (offset and order) that match best,
and report how well they match.

Usage:
  python tools/compare_surface_volumes.py OUT_JSON OURS_ZARR REFERENCE_ZARR [--window 2048] [--cache DIR]

Either zarr can be local or an https URL, a plain array or a group with level "0", with the layer
axis first or last (the shortest axis is taken as the layer axis, as the ink inference does). Only a
window around the centre is read. Layers are compared by Pearson correlation over pixels where both
volumes are non-zero.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def open_any(path, cache=None):
    import fsspec
    import zarr
    if path.startswith(("http://", "https://")) and cache:
        from snap_segment import _chunk_cache_class, cache_dir_for
        store = _chunk_cache_class()(fsspec.get_mapper(path.rstrip("/")), cache_dir_for(cache, path.rstrip("/")))
    else:
        store = fsspec.get_mapper(path.rstrip("/"))
    root = zarr.open(store, mode="r")
    arr = root if hasattr(root, "shape") else root["0"]
    depth_first = int(np.argmin(arr.shape)) == 0
    return arr, depth_first


def read_window(arr, depth_first, r0, r1, c0, c1):
    if depth_first:
        return np.asarray(arr[:, r0:r1, c0:c1]).astype(np.float32)
    return np.moveaxis(np.asarray(arr[r0:r1, c0:c1, :]), -1, 0).astype(np.float32)


def phase_shift(a, b):
    """(dy, dx) such that b[y + dy, x + dx] matches a[y, x] best (phase correlation)."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fb * np.conj(fa) / (np.abs(fb * np.conj(fa)) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    H, W = a.shape
    return int(dy - H if dy > H // 2 else dy), int(dx - W if dx > W // 2 else dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--cache")
    ap.add_argument("--samples", type=int, default=400000)
    a = ap.parse_args()
    A, a_first = open_any(a.ours, a.cache)
    B, b_first = open_any(a.reference, a.cache)
    sa = A.shape if a_first else (A.shape[2], A.shape[0], A.shape[1])
    sb = B.shape if b_first else (B.shape[2], B.shape[0], B.shape[1])
    H, W = min(sa[1], sb[1]), min(sa[2], sb[2])
    w = min(a.window, H, W)
    r0, c0 = (H - w) // 2, (W - w) // 2
    va = read_window(A, a_first, r0, r0 + w, c0, c0 + w)
    vb = read_window(B, b_first, r0, r0 + w, c0, c0 + w)
    # in-plane shift from the mean over layers (robust to the layer alignment)
    dy, dx = phase_shift(va.mean(0), vb.mean(0))
    ys, xs = slice(max(0, -dy), w - max(0, dy)), slice(max(0, -dx), w - max(0, dx))
    yb, xb = slice(max(0, dy), w - max(0, -dy)), slice(max(0, dx), w - max(0, -dx))
    pa, pb = va[:, ys, xs], vb[:, yb, xb]
    both = (pa > 0).all(0) & (pb > 0).all(0)
    idx = np.flatnonzero(both.ravel())
    if len(idx) < 1000:
        sys.exit(f"only {len(idx)} pixels are non-zero in both windows; nothing to compare")
    idx = np.random.default_rng(0).choice(idx, min(a.samples, len(idx)), replace=False)
    xa = pa.reshape(pa.shape[0], -1)[:, idx]
    xb = pb.reshape(pb.shape[0], -1)[:, idx]
    za = (xa - xa.mean(1, keepdims=True)) / (xa.std(1, keepdims=True) + 1e-9)
    zb = (xb - xb.mean(1, keepdims=True)) / (xb.std(1, keepdims=True) + 1e-9)
    C = za @ zb.T / za.shape[1]                       # corr(ours layer i, reference layer j)
    best = None
    for order in (1, -1):
        for off in range(-(sb[0] - 1), sa[0] + sb[0]):
            pairs = [(i, off + order * i) for i in range(sa[0]) if 0 <= off + order * i < sb[0]]
            if len(pairs) < max(3, min(sa[0], sb[0]) // 2):
                continue
            m = float(np.mean([C[i, j] for i, j in pairs]))
            if best is None or m > best["mean_corr"]:
                best = dict(order="same" if order == 1 else "reversed", ref_layer_of_our_layer_0=off,
                            pairs=len(pairs), mean_corr=m, min_corr=float(min(C[i, j] for i, j in pairs)))
    res = dict(ours=a.ours, reference=a.reference, ours_shape=list(sa), reference_shape=list(sb),
               window=[r0, r0 + w, c0, c0 + w], inplane_shift_yx=[dy, dx], pixels_compared=int(len(idx)),
               best_layer_alignment=best, corr_ours_by_reference=np.round(C, 3).tolist())
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"shapes: ours {sa}, reference {sb}; in-plane shift (rows, cols) {dy:+d}, {dx:+d}; "
          f"best layer match: {best['order']} order, our layer 0 = reference layer {best['ref_layer_of_our_layer_0']}, "
          f"mean correlation {best['mean_corr']:.3f} (lowest {best['min_corr']:.3f}) over {best['pairs']} layers")


if __name__ == "__main__":
    main()
