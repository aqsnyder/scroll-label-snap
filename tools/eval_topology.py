"""Topology of snapped labels against the originals: Betti numbers, components, fragments and merger events.

Usage:
  python tools/eval_topology.py ORIG SNAPPED OUT_DIR [--workers N] [--limit K] [--min-size 1000]

ORIG and SNAPPED are label folders with matching file names (or folders holding labels/, as tools/snap_folder.py
writes them), or a pair of TIFFs. Each cube's result goes to OUT_DIR/cubes/<name>.json as soon as it finishes, so an
interrupted run resumes where it stopped. At the end OUT_DIR/topology.json holds all cubes and OUT_DIR/topology.csv
one row per cube. No CT is read. Each worker needs about 1 GB of RAM on a 320 cubed cube.

Label 1 is the foreground; label 2 and everything outside the cube count as background. Per cube, for the original
and the snapped band:
  - b0, b1, b2 (components, tunnels, cavities) and the Euler number, at both connectivity pairs: "26" is a 26-connected
    foreground with a 6-connected background (voxels as closed cubes), "6" is the reverse (voxels as vertices).
    b0 and b2 come from labelling the foreground and the padded complement, b1 from b0 + b2 - Euler, and the Euler
    number from counting cubical cells directly.
  - components of at least --min-size voxels, and fragments, the smaller ones, with their voxels (26-connected).
  - merger events: sheets of the original that end up in one component. Every 8th voxel of the original band looks
    for its nearest snapped component within 7 voxels. An original component of at least --min-size voxels goes to the
    snapped component holding most of those voxels, if that is at least half of them and it has at least --min-size
    voxels itself. A snapped component that receives k >= 2 sheets is k - 1 events. This is a rule, not a tracking of
    voxels: a small sheet whose voxels end up nearest a neighbouring sheet counts even if the two stay separate. On the
    published labels of the 20 development cubes it equals per-voxel tracking cube by cube (50 events).
  - for merged components, the contact: each of their voxels near a meeting of two sheets takes the nearest original
    component, and 26-adjacent voxels that differ are contact voxels. Reported in total and per pair of sheets.

The raw numbers mislead without controls, so run them next to the output being judged. On the 20 development cubes:
a rigid roll of the band (np.roll) gives no change at all, and a uniform 1-voxel shift along the normals, through
warp_labels, already makes some tunnels and mergers (b1 at 26-connectivity summed over the cubes 215 for +1 and 464 for
-1, merger events 4 and 5, against 0 and 0 for the original). The published shifts give 7,027 and 50, and the same
shifts negated give 7,300 and 24. So tunnels come from warp_labels with any non-rigid move of this size, not only from
where the band is sent. Betti numbers equal to the original's say nothing about whether the band is in the right place,
and originals are not ground truth: 152 of the 892 have tunnels of their own at 26-connectivity.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage as ndi
import tifffile

S26 = np.ones((3, 3, 3), bool)
S6 = ndi.generate_binary_structure(3, 1)
GE = (1, 10, 100, 1000, 10000, 100000)     # component sizes counted in the JSON
REACH = 7.0                                # how far a voxel looks for its nearest component (voxels)
BLOCK = 8                                  # block size for the search for places where sheets meet


def euler_closed_cubes(b):
    """Euler number of the union of closed unit cubes (foreground 26-connected, background 6)."""
    p = np.pad(b, 1)
    f0 = p[1:, :, :] | p[:-1, :, :]          # faces perpendicular to axis 0
    f1 = p[:, 1:, :] | p[:, :-1, :]          # faces perpendicular to axis 1
    f2 = p[:, :, 1:] | p[:, :, :-1]          # faces perpendicular to axis 2
    faces = int(f0.sum()) + int(f1.sum()) + int(f2.sum())
    e2 = f0[:, 1:, :] | f0[:, :-1, :]        # edges along axis 2
    e1 = f0[:, :, 1:] | f0[:, :, :-1]        # edges along axis 1
    e0 = f1[:, :, 1:] | f1[:, :, :-1]        # edges along axis 0
    edges = int(e2.sum()) + int(e1.sum()) + int(e0.sum())
    verts = int((e2[:, :, 1:] | e2[:, :, :-1]).sum())
    return verts - edges + faces - int(b.sum())


def euler_voxel_vertices(b):
    """Euler number of the complex with voxels as vertices (foreground 6-connected, background 26)."""
    a0 = b[1:, :, :] & b[:-1, :, :]
    a1 = b[:, 1:, :] & b[:, :-1, :]
    a2 = b[:, :, 1:] & b[:, :, :-1]
    edges = int(a0.sum()) + int(a1.sum()) + int(a2.sum())
    s01 = a0[:, 1:, :] & a0[:, :-1, :]
    s02 = a0[:, :, 1:] & a0[:, :, :-1]
    s12 = a1[:, :, 1:] & a1[:, :, :-1]
    squares = int(s01.sum()) + int(s02.sum()) + int(s12.sum())
    return int(b.sum()) - edges + squares - int((s01[:, :, 1:] & s01[:, :, :-1]).sum())


def betti(b, conn=26, n_fg=None):
    """dict(b0, b1, b2, euler) of a boolean volume; the outside counts as background. conn is the foreground's
    connectivity, 26 or 6. n_fg: the foreground's component count at that connectivity, if already known."""
    fg, bg = (S26, S6) if conn == 26 else (S6, S26)
    b0 = ndi.label(b, structure=fg)[1] if n_fg is None else n_fg
    b2 = ndi.label(np.pad(~b, 1, constant_values=True), structure=bg)[1] - 1
    chi = euler_closed_cubes(b) if conn == 26 else euler_voxel_vertices(b)
    return dict(b0=int(b0), b1=int(b0 + b2 - chi), b2=int(b2), euler=int(chi))


class Nearest:
    """Nearest nonzero voxel of a label volume within a radius, by walking a shell of offsets outward."""

    def __init__(self, lab, radius=REACH):
        r = int(np.ceil(radius))
        self.pad = r + 1
        self.flat = np.pad(lab.astype(np.int32), self.pad).ravel()
        s = np.array(lab.shape) + 2 * self.pad
        self.strides = np.array([s[1] * s[2], s[2], 1], np.int64)
        g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1].reshape(3, -1).T
        d2 = (g ** 2).sum(1)
        g, d2 = g[d2 <= radius * radius], d2[d2 <= radius * radius]
        g = g[np.lexsort((g[:, 2], g[:, 1], g[:, 0], d2))]      # nearest first, in a fixed order
        self.steps = (g * self.strides).sum(1)

    def query(self, coords):
        """Label of the nearest nonzero voxel for each of coords (N, 3); 0 if there is none in reach."""
        out = np.zeros(len(coords), np.int32)
        at = ((coords.astype(np.int64) + self.pad) * self.strides).sum(1)
        rem = np.arange(len(coords))
        for step in self.steps:
            if rem.size == 0:
                break
            v = self.flat[at[rem] + step]
            hit = v > 0
            out[rem[hit]] = v[hit]
            rem = rem[~hit]
        return out


def every_8th(mask):
    """mask, keeping only voxels with all coordinates even."""
    k = mask.copy()
    k[1::2, :, :] = False
    k[:, 1::2, :] = False
    k[:, :, 1::2] = False
    return k


def merger_events(l_orig, size_orig, l_snap, size_snap, min_size, share=0.5):
    """The merger rule in this file's docstring. Labels are 1-based; sizes are indexed by label - 1.
    Returns the number of events and {snapped component: [original components]}."""
    k = every_8th(l_orig > 0)
    found = Nearest(l_snap).query(np.argwhere(k))
    ok = found > 0
    key, count = np.unique(l_orig[k][ok].astype(np.int64) * (1 << 24) + found[ok], return_counts=True)
    best, total = {}, {}
    for v, c in zip(key.tolist(), count.tolist()):
        a, x = v >> 24, v & ((1 << 24) - 1)
        total[a] = total.get(a, 0) + c
        if a not in best or c > best[a][0]:
            best[a] = (c, x)
    groups = {}
    for a, (c, x) in best.items():
        if size_orig[a - 1] >= min_size and size_snap[x - 1] >= min_size and c >= share * total[a]:
            groups.setdefault(x, []).append(a)
    merged = {x: sorted(m) for x, m in groups.items() if len(m) >= 2}
    return sum(len(m) - 1 for m in merged.values()), merged


def contacts(l_orig, l_snap, merged):
    """Contact voxels of merged components, as in this file's docstring.
    Returns (total contact voxels, [dict(pair=[A, B], contact_voxels=n)])."""
    sheet_of = {a: x for x, m in merged.items() for a in m}
    nearest = Nearest(l_orig)
    in_merged = np.isin(l_snap, np.array(list(merged), np.int32))

    # only look where a sparse set of merged voxels has two nearest sheets in one block, or next to such a block
    sparse = np.argwhere(every_8th(in_merged))
    near = nearest.query(sparse)
    grid = (np.array(l_snap.shape) + BLOCK - 1) // BLOCK
    block = np.ravel_multi_index((sparse // BLOCK).T, grid)
    seen = np.unique(np.stack([block, near], 1)[near > 0], axis=0)
    ids, n_sheets = np.unique(seen[:, 0], return_counts=True)
    cand = np.zeros(tuple(grid), bool)
    cand[np.unravel_index(ids[n_sheets >= 2], grid)] = True
    cand = ndi.binary_dilation(cand, structure=S26)
    cand = cand.repeat(BLOCK, 0).repeat(BLOCK, 1).repeat(BLOCK, 2)[:l_snap.shape[0], :l_snap.shape[1], :l_snap.shape[2]]

    coords = np.argwhere(in_merged & cand)
    who = nearest.query(coords)
    assigned = np.zeros(l_snap.shape, np.int32)
    assigned[tuple(coords.T)] = who
    padded = np.pad(assigned, 1)
    flat = padded.ravel()
    strides = np.array([padded.shape[1] * padded.shape[2], padded.shape[2], 1], np.int64)
    at = ((coords.astype(np.int64) + 1) * strides).sum(1)
    us, partners = [np.zeros(0, np.int64)], [np.zeros((0, 3), np.int64)]
    for o in np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T:
        if tuple(o) > (0, 0, 0):                            # each 26-adjacent pair once
            v = flat[at + int((o * strides).sum())]
            k = np.flatnonzero((v > 0) & (v != who) & (who > 0))
            us.append(k)
            partners.append(coords[k] + o)
    u, w = np.concatenate(us), np.concatenate(partners)
    p, q = who[u], assigned[tuple(w.T)]
    same = np.array([sheet_of.get(int(i), -1) == sheet_of.get(int(j), -2) for i, j in zip(p, q)], bool)
    u, w, p, q = coords[u[same]], w[same], p[same], q[same]
    contact = np.zeros(l_snap.shape, bool)
    contact[tuple(u.T)] = True
    contact[tuple(w.T)] = True
    pairs = []
    for i, j in sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(p, q)}):
        sel = ((p == i) & (q == j)) | ((p == j) & (q == i))
        pairs.append(dict(pair=[i, j], contact_voxels=len(np.unique(np.r_[u[sel], w[sel]], axis=0))))
    return int(contact.sum()), pairs


def measure(orig_path, snap_path, name=None, min_size=1000):
    name = name or os.path.basename(orig_path).rsplit(".", 1)[0]
    return measure_labels(tifffile.imread(orig_path), tifffile.imread(snap_path), name, min_size)


def measure_labels(lab_o, lab_s, name, min_size=1000):
    t0 = time.time()
    if lab_o.shape != lab_s.shape:
        raise ValueError(f"the original has shape {lab_o.shape} but the snapped labels have shape {lab_s.shape}")
    b_o, b_s = lab_o == 1, lab_s == 1
    l_o, n_o = ndi.label(b_o, structure=S26)
    l_s, n_s = ndi.label(b_s, structure=S26)
    size_o = np.bincount(l_o.ravel(), minlength=n_o + 1)[1:]
    size_s = np.bincount(l_s.ravel(), minlength=n_s + 1)[1:]
    events, merged = merger_events(l_o, size_o, l_s, size_s, min_size) if n_o and n_s else (0, {})
    n_contact, pairs = contacts(l_o, l_s, merged) if merged else (0, [])
    return dict(
        cube=name, shape=[int(v) for v in lab_o.shape], annotated=float((lab_o != 2).mean()),
        label2_same=bool(np.array_equal(lab_o == 2, lab_s == 2)),
        voxels=dict(original=int(b_o.sum()), snapped=int(b_s.sum())),
        betti=dict(original={"26": betti(b_o, 26, n_o), "6": betti(b_o, 6)},
                   snapped={"26": betti(b_s, 26, n_s), "6": betti(b_s, 6)}),
        min_size=min_size,
        components=dict(original=int((size_o >= min_size).sum()), snapped=int((size_s >= min_size).sum())),
        fragments=dict(original=int((size_o < min_size).sum()), snapped=int((size_s < min_size).sum())),
        fragment_voxels=dict(original=int(size_o[size_o < min_size].sum()), snapped=int(size_s[size_s < min_size].sum())),
        components_ge={str(t): [int((size_o >= t).sum()), int((size_s >= t).sum())] for t in GE},
        merger_events=int(events),
        merged=[dict(snapped_component=int(x), sheets=m, sizes=[int(size_o[a - 1]) for a in m], size=int(size_s[x - 1]))
                for x, m in sorted(merged.items())],
        contact_voxels=int(n_contact), contact_pairs=pairs, seconds=round(time.time() - t0, 1))


def run_one(orig_path, snap_path, out_json, name, min_size):
    r = measure(orig_path, snap_path, name, min_size)
    with open(out_json + ".tmp", "w") as f:
        json.dump(r, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    return r


def csv_row(r):
    """One flat row per cube: the Betti numbers, then the size and merger counts."""
    row = dict(cube=r["cube"], label2_same=r["label2_same"],
               voxels_original=r["voxels"]["original"], voxels_snapped=r["voxels"]["snapped"])
    for conn in ("26", "6"):
        for which in ("original", "snapped"):
            for k, v in r["betti"][which][conn].items():
                row[f"{k}_{conn}_{which}"] = v
    for k in ("components", "fragments", "fragment_voxels"):
        for which in ("original", "snapped"):
            row[f"{k}_{which}"] = r[k][which]
    row.update(merger_events=r["merger_events"], contact_voxels=r["contact_voxels"], seconds=r["seconds"])
    return row


def label_dir(d):
    return os.path.join(d, "labels") if os.path.isdir(os.path.join(d, "labels")) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig", help="original labels: a folder, or a TIFF")
    ap.add_argument("snapped", help="snapped labels: a folder, or a TIFF")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first K cubes")
    ap.add_argument("--min-size", type=int, default=1000, help="smaller components are fragments, and take no part in merger events")
    a = ap.parse_args()

    if os.path.isfile(a.orig) != os.path.isfile(a.snapped):
        sys.exit("give two folders or two TIFFs")
    if os.path.isfile(a.orig):
        jobs = [(os.path.basename(a.orig).rsplit(".", 1)[0], a.orig, a.snapped)]
    else:
        od, sd = label_dir(a.orig), label_dir(a.snapped)
        names = sorted(f for f in os.listdir(od) if f.endswith(".tif") and os.path.exists(os.path.join(sd, f)))
        jobs = [(f[:-4], os.path.join(od, f), os.path.join(sd, f)) for f in names]
        if a.limit:
            jobs = jobs[:a.limit]
    cube_dir = os.path.join(a.out_dir, "cubes")
    os.makedirs(cube_dir, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(cube_dir, j[0] + ".json"))]
    print(f"{len(jobs)} cube pairs, {len(todo)} to measure, {a.workers} workers", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, o, s, os.path.join(cube_dir, n + ".json"), n, a.min_size): n for n, o, s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                b = r["betti"]["snapped"]["26"]
                print(f"[{k}/{len(todo)}] {r['cube']}: b0 {b['b0']}, b1 {b['b1']}, b2 {b['b2']}, "
                      f"{r['merger_events']} merger events, {r['seconds']} s", flush=True)
            except Exception as e:  # keep going; report at the end
                failed.append(futs[fut])
                print(f"[{k}/{len(todo)}] {futs[fut]}: FAILED: {type(e).__name__}: {e}", flush=True)

    cubes = [json.load(open(os.path.join(cube_dir, j[0] + ".json"))) for j in jobs
             if os.path.exists(os.path.join(cube_dir, j[0] + ".json"))]
    with open(os.path.join(a.out_dir, "topology.json"), "w") as f:
        json.dump(cubes, f, indent=1)
    if cubes:
        rows = [csv_row(r) for r in cubes]
        with open(os.path.join(a.out_dir, "topology.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote topology.json and topology.csv with {len(cubes)} of {len(jobs)} cubes -> {a.out_dir}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} cubes failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
