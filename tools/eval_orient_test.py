"""The pre-registered test of Params.orient_cube: three arms on each cube's clean labels and on three drifts.

results/orient_test_plan.md says what is measured and what decides. This tool does the measuring and applies that rule.

Usage:
  python tools/eval_orient_test.py IN_DIR OUT_DIR [--workers N] [--cubes NAME ...] [--only NAME ...]

IN_DIR holds images/ and labels/. The cubes are the plan's 100, or --cubes (k below is a cube's position in that list);
--only runs part of the list with the same k, and the decision then counts as incomplete. Each cube's result goes to
OUT_DIR/cubes/<name>.json as soon as it finishes, so an interrupted run resumes where it stopped. At the end
OUT_DIR/orient_test.json holds the decision and the numbers behind it, and OUT_DIR/orient_test_cubes.csv one row per cube.
On one development cube (sample_00018) a worker took about 6 minutes and 1.3 GB.

Per cube, on one fixed set of points, the clean labels' samples at least 10 voxels from every face:
  - each arm's band position (tools/eval_snap.py band_position) along the same lines, on the clean labels and on
    drifted labels at seeds k, 100+k and 200+k. The drifted labels are built once, along the default arm's normals, as
    in tools/eval_snap.py, so all arms start from the same input.
  - consistency: within 1 voxel of the same arm's clean result, at all points.
  - drift: within 1 voxel of the original label, at the points trusted under the default's and under orient_cube's normals.
  - orientation change: the share of points whose nearest sample in the drifted run, within 6 voxels, has a reversed normal.
  - axis share, sign margin and guard triggers of the orient_cube step on all four inputs.
  - the fibre signature, which uses no labels and no CT step sign, on the samples where the two orientations disagree on
    the clean labels. It assumes the scroll axis is scan z.
  - Betti numbers and merger events (tools/eval_topology.py) and contested samples on the clean output of each arm.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import tifffile  # noqa: E402

from eval_snap import band_position, drift_field, trusted_samples  # noqa: E402
from eval_topology import S26, measure_labels  # noqa: E402
from scroll_label_snap.core import (Params, neighbour_graph, orient_normals, sample, solve_surface,  # noqa: E402
                                    surface_points, warp_labels)

PLAN = os.path.join(HERE, "..", "results", "orient_test_plan.md")
ARMS = ("default", "orient_cube", "orient_cube_keep")
SEED_OFFSETS = (0, 100, 200)
GUARD_SHARE = 0.7            # as in core.orient_normals
MIN_DISPUTED = 1000          # disputed samples with fibre data for a cube to count in the fibre guard
STRETCH = (161, 280)         # the id stretch reported separately


# --------------------------------------------------------------------------- which cubes

def plan_lists(path=PLAN):
    """The id lists in the plan's fenced block: {label: [names]}. A list is a label line ending in ':' and rows of ids."""
    with open(path, encoding="utf-8") as f:
        block = re.search(r"```text\n(.*?)```", f.read(), flags=re.S).group(1)
    out = {}
    for part in block.strip().split("\n\n"):
        label, rows = part.split("\n", 1)
        out[label.split(":")[0].strip()] = re.findall(r"sample_\d{5}", rows)
    return out


def draw_cubes(names, excluded, seed=20261103, count=100):
    pool = sorted(set(names) - set(excluded))
    return sorted(np.random.default_rng(seed).choice(pool, count, replace=False).tolist())


# --------------------------------------------------------------------------- orientation

def votes_along(pts, n, ct_s):
    """Each sample's CT vote along n, as in core.orient_normals: positive when the intensity drops toward +n."""
    tt = np.arange(-1.5, 1.501, 0.5, dtype=np.float32)
    prof = np.stack([sample(ct_s, pts + t * n) for t in tt], 1)
    d = np.diff(prof, axis=1)
    return -d[np.arange(len(pts)), np.argmax(np.abs(d), 1)]


def orientation_numbers(pts, n, comp, ct_s):
    """What the orient_cube step sees, from oriented normals and their sheets. axis_share: the normals' squared projection on
    their dominant axis. sign_margin: how decisive the axis's sign is, (votes for - votes against) / all, with sheets weighted
    by |vote|. across_axis: samples on sheets with |mean along the axis| under 0.3, which keep their vote. against_vote:
    samples these normals turn against their own sheet's vote."""
    c = int(comp.max()) + 1
    vote = np.bincount(comp, weights=votes_along(pts, n, ct_s), minlength=c)
    voted = n.astype(np.float64) * np.where(vote[comp] < 0, -1.0, 1.0)[:, None]
    w, v = np.linalg.eigh(voted.T @ voted)
    along = np.bincount(comp, weights=voted @ v[:, -1], minlength=c) / np.bincount(comp, minlength=c)
    weight = np.abs(vote)
    return dict(axis_share=float(w[-1] / w.sum()), guard_triggered=bool(w[-1] < GUARD_SHARE * w.sum()),
                sign_margin=float(abs((np.sign(along) * weight).sum()) / max(weight.sum(), 1e-9)),
                across_axis=float(np.mean(np.abs(along[comp]) < 0.3)), against_vote=float(np.mean(vote[comp] < 0)))


def flip_share(pts0, n0, pts1, n1, r=6.0, thr=-0.5):
    """Share of points (pts0, n0) whose nearest sample of another run, within r voxels, has a reversed normal."""
    d, idx = cKDTree(pts1).query(pts0, k=1)
    return float(np.mean((d <= r) & (np.einsum("ij,ij->i", n0, n1[idx]) < thr)))


# --------------------------------------------------------------------------- fibre signature

def bright_runs(Q, R, ct_s):
    """Along each normal, the bright run the sample sits in: its ends (voxels along R), its length, and whether it looks like
    one whole sheet (6-16 voxels long, with both ends in range)."""
    tt = np.arange(-12, 12.01, 0.5, dtype=np.float32)
    pr = np.stack([sample(ct_s, Q + t * R) for t in tt], 1)
    hi = pr > float(np.median(pr))
    k0 = len(tt) // 2
    start = np.full(len(Q), -1)
    for off in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
        m = (start < 0) & hi[:, k0 + off]
        start[m] = k0 + off
    up, lo = np.full(len(Q), -1), np.full(len(Q), -1)
    for r in np.flatnonzero(start >= 0):
        b = c = start[r]
        while b + 1 < len(tt) and hi[r, b + 1]:
            b += 1
        while c - 1 >= 0 and hi[r, c - 1]:
            c -= 1
        up[r], lo[r] = b, c
    length = (up - lo) * 0.5
    good = (start >= 0) & (length >= 6) & (length <= 16) & (up < len(tt) - 1) & (lo > 0)
    return tt[np.clip(up, 0, None)], tt[np.clip(lo, 0, None)], good


def fibre_signature(img, pts, n, disputed, depth=2.0):
    """The signature under normals n: S = A(`depth` voxels inside the +n face) - A(the same inside the -n face), with
    A = (Ea - Ew) / (Ea + Ew), the squared CT gradient along the scroll axis (scan z) and around the scroll, summed over
    the samples. Positive means these normals agree with the fibre texture. The placebo is the same with those two directions
    turned 45 degrees and should be near 0. t is over blocks of 40 voxels with at least 30 samples each. Covers the samples
    at least 16 voxels from every face, with normals not along the axis, in a bright run like one whole sheet."""
    img = img.astype(np.float32)
    ct_s = ndi.gaussian_filter(img, 1.0)
    inside = np.all((pts >= 16) & (pts <= np.array(img.shape) - 17), axis=1)
    sel = np.flatnonzero(inside & (np.abs(n[:, 0]) < 0.7))
    Q, R = pts[sel], n[sel]
    up, lo, good = bright_runs(Q, R, ct_s)
    grad = [ndi.gaussian_filter(img, 1.0, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    z = np.array([1.0, 0, 0], np.float32)
    ua = z - (R @ z)[:, None] * R
    ua /= np.maximum(np.linalg.norm(ua, axis=1, keepdims=True), 1e-9)
    uw = np.cross(R, ua)
    dirs = dict(a=ua, w=uw, p=(ua + uw) / np.sqrt(2), m=(ua - uw) / np.sqrt(2))
    E = {}
    for side, base, sgn in (("plus", up, -1.0), ("minus", lo, 1.0)):
        g = np.stack([sample(v, Q + (base + sgn * depth)[:, None] * R) for v in grad], 1)
        for d, u in dirs.items():
            E[side, d] = np.einsum("ij,ij->i", g, u) ** 2

    def signature(mask, x="a", y="w"):
        def A(side):
            ex, ey = E[side, x][mask].sum(), E[side, y][mask].sum()
            return (ex - ey) / (ex + ey + 1e-9)
        return float(A("plus") - A("minus"))

    block = (Q // 40).astype(np.int64)
    block = (block[:, 0] * 100 + block[:, 1]) * 100 + block[:, 2]

    def t_over_blocks(mask):
        vals = [signature(mask & (block == b)) for b in np.unique(block[mask]) if (mask & (block == b)).sum() >= 30]
        v = np.array(vals)
        return (float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-9)) if len(v) >= 4 else None), int(len(v))

    d = good & disputed[sel]
    t, blocks = t_over_blocks(good)
    r = dict(n=int(good.sum()), S=signature(good) if good.any() else None, placebo=signature(good, "p", "m") if good.any() else None,
             t=t, blocks=blocks, disputed_n=int(d.sum()), disputed_S=None, disputed_t=None)
    if d.any():
        r["disputed_S"] = signature(d)
        r["disputed_t"] = t_over_blocks(d)[0]
    return r


# --------------------------------------------------------------------------- contested samples

def sample_sheets(pts, sheets):
    """Input sheet of each sample: the sheet at its rounded position, else the nearest band voxel."""
    q = np.clip(np.rint(pts).astype(int), 0, np.array(sheets.shape) - 1)
    s = sheets[q[:, 0], q[:, 1], q[:, 2]].astype(np.int64)
    if (s == 0).any():
        c = np.argwhere(sheets > 0)
        nearest = cKDTree(c).query(pts[s == 0])[1]
        s[s == 0] = sheets[c[nearest, 0], c[nearest, 1], c[nearest, 2]]
    return s


def contested_samples(pts, n, shift, sheet, near=3.0):
    """Samples whose moved position comes within `near` voxels of a moved sample of another input sheet, and at least 0.5
    voxel closer to it than before the move: two labels going to one face."""
    moved = pts + np.asarray(shift, np.float32)[:, None] * n
    out = np.zeros(len(pts), bool)
    pairs = cKDTree(moved).query_pairs(near, output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        a, b = a[sheet[a] != sheet[b]], b[sheet[a] != sheet[b]]
        closer = np.linalg.norm(moved[a] - moved[b], axis=1) < np.linalg.norm(pts[a] - pts[b], axis=1) - 0.5
        out[a[closer]] = True
        out[b[closer]] = True
    return out


# --------------------------------------------------------------------------- one cube

def errors(e):
    e = e[np.isfinite(e)]
    if not len(e):
        return dict(within_1=None, over_2=None, mean=None, n=0)
    return dict(within_1=float(np.mean(e <= 1)), over_2=float(np.mean(e > 2)), mean=float(np.mean(e)), n=int(len(e)))


def run_arms(lab_in, ct_s, pts, raw, i, j, p):
    """Solve and warp all arms on one input: normals, shifts and labels per arm, with one solve per distinct orientation."""
    n_def, comp = orient_normals(pts, raw, i, j, ct_s)
    n_ori, _ = orient_normals(pts, raw, i, j, ct_s, orient_cube=True)
    same = np.array_equal(n_def, n_ori)
    s_def = solve_surface(lab_in.shape, (ct_s, pts, n_def, i, j, comp), p)
    s_ori = s_def if same else solve_surface(lab_in.shape, (ct_s, pts, n_ori, i, j, comp), p)
    plain_def = warp_labels(lab_in, pts, n_def, s_def[0], comp=comp, smooth=p.smooth_warp)
    plain_ori = plain_def if same else warp_labels(lab_in, pts, n_ori, s_ori[0], comp=comp, smooth=p.smooth_warp)
    kept = warp_labels(lab_in, pts, n_ori, s_ori[0], comp=comp, smooth=p.smooth_warp, keep_topology=True)
    return dict(comp=comp, normals=dict(default=n_def, orient_cube=n_ori, orient_cube_keep=n_ori),
                shift=dict(default=s_def[0], orient_cube=s_ori[0], orient_cube_keep=s_ori[0]),
                labels=dict(default=plain_def, orient_cube=plain_ori, orient_cube_keep=kept),
                c_off=dict(default=s_def[4]["convention_offset"], orient_cube=s_ori[4]["convention_offset"]))


def measure_cube(img_path, lab_path, k):
    t0 = time.time()
    p = Params()
    img = tifffile.imread(img_path)
    lab = tifffile.imread(lab_path)
    name = os.path.basename(img_path).rsplit(".", 1)[0]
    ct_s = ndi.gaussian_filter(img.astype(np.float32), p.sigma_ct)
    pts, raw, _ = surface_points(lab, p)
    if len(pts) == 0:
        raise ValueError("no samples in this cube")
    i, j = neighbour_graph(pts, raw, p)

    clean = run_arms(lab, ct_s, pts, raw, i, j, p)
    n_def, n_ori, comp = clean["normals"]["default"], clean["normals"]["orient_cube"], clean["comp"]
    inner = np.all((pts >= 10) & (pts <= np.array(lab.shape) - 11), axis=1)
    pi, ni = pts[inner], n_def[inner]                      # one set of points and lines for all arms and inputs
    o = band_position(lab, pi, ni)
    trusted = trusted_samples(lab, (ct_s, pts, n_def, i, j, comp), p)
    if not np.array_equal(n_def, n_ori):
        trusted = trusted & trusted_samples(lab, (ct_s, pts, n_ori, i, j, comp), p)
    fixed = trusted[inner]
    q_clean = {arm: band_position(clean["labels"][arm], pi, ni) for arm in ARMS}
    disputed = np.einsum("ij,ij->i", n_def, n_ori) < 0

    sheets = ndi.label(lab == 1, structure=S26)[0]
    pid = sample_sheets(pts, sheets)
    del sheets
    arms = {}
    for arm in ARMS:
        m = measure_labels(lab, clean["labels"][arm], name)
        contested = contested_samples(pts, clean["normals"][arm], clean["shift"][arm], pid)
        arms[arm] = dict(betti=m["betti"]["snapped"], betti_equal=m["betti"]["snapped"] == m["betti"]["original"],
                         merger_events=m["merger_events"], merged=m["merged"], components=m["components"],
                         fragments=m["fragments"]["snapped"], contested=float(contested.mean()),
                         orientation=orientation_numbers(pts, clean["normals"][arm], comp, ct_s))
    contested = contested_samples(pts, n_ori, clean["shift"]["orient_cube"], pid)[inner]
    keep_moved = np.abs(q_clean["orient_cube_keep"] - q_clean["orient_cube"]) > 1
    arms["orient_cube_keep"].update(band_moved_over_1_contested=float(np.mean(keep_moved[contested])) if contested.any() else None,
                                    band_moved_over_1_elsewhere=float(np.mean(keep_moved[~contested])) if (~contested).any() else None)
    del clean

    inputs = dict(clean=orientation_numbers(pts, n_ori, comp, ct_s))
    seeds = {}
    for off in SEED_OFFSETS:
        seed = off + k
        drift = drift_field(pts, 1.5, 40.0, np.random.default_rng(seed))
        lab_d = warp_labels(lab, pts, n_def, drift)
        pts_d, raw_d, _ = surface_points(lab_d, p)
        i_d, j_d = neighbour_graph(pts_d, raw_d, p)
        run = run_arms(lab_d, ct_s, pts_d, raw_d, i_d, j_d, p)
        row = {}
        for arm in ARMS:
            q = band_position(run["labels"][arm], pi, ni)
            n_arm = n_def if arm == "default" else n_ori
            row[arm] = dict(consistency=errors(np.abs(q - q_clean[arm])), drift=errors(np.abs(q - o)[fixed]),
                            orientation_change=flip_share(pi, n_arm[inner], pts_d, run["normals"][arm]))
        row["c_off"] = run["c_off"]
        seeds[str(seed)] = row
        inputs[str(seed)] = orientation_numbers(pts_d, run["normals"]["orient_cube"], run["comp"], ct_s)

    shares = [v["axis_share"] for v in inputs.values()]
    margins = [v["sign_margin"] for v in inputs.values()]
    apart = np.abs(q_clean["default"] - q_clean["orient_cube"])[disputed[inner]]
    apart = apart[np.isfinite(apart)]
    return dict(cube=name, id=int(re.search(r"(\d+)$", name).group(1)), k=k, n_samples=int(len(pts)), n_points=int(inner.sum()),
                n_fixed=int(fixed.sum()), axis_share=inputs["clean"]["axis_share"], axis_share_min=min(shares),
                sign_margin=inputs["clean"]["sign_margin"], sign_margin_min=min(margins),
                guard_triggers=sum(v["guard_triggered"] for v in inputs.values()), inputs=inputs, disputed=float(disputed.mean()),
                disputed_points_differ_over_1=float(np.mean(apart > 1)) if len(apart) else None,
                fibre=fibre_signature(img, pts, n_ori, disputed), arms=arms, seeds=seeds, seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------------- the plan's rule

def bootstrap(d, seed=0, n=5000):
    """Mean and 95% interval by resampling cubes."""
    d = np.asarray(d, float)
    b = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return dict(mean=float(d.mean()), low=float(np.percentile(b, 2.5)), high=float(np.percentile(b, 97.5)), cubes=int(len(d)))


def cube_mean(c, arm, key):
    """A cube's mean within-1 share over the seeds, in points; None if a seed has no such points."""
    v = [c["seeds"][s][arm][key]["within_1"] for s in sorted(c["seeds"])]
    return None if any(x is None for x in v) else 100 * float(np.mean(v))


def paired(cubes, arm, key, base="default"):
    """arm minus base per cube, in points, by cube."""
    d = [(cube_mean(c, arm, key), cube_mean(c, base, key)) for c in cubes]
    d = [a - b for a, b in d if a is not None and b is not None]
    if not d:
        return dict(mean=None, low=None, high=None, cubes=0, better=0, worse=0)
    return dict(bootstrap(d), better=sum(x > 0.5 for x in d), worse=sum(x < -0.5 for x in d))


def arm_mean(cubes, arm, key):
    v = [cube_mean(c, arm, key) for c in cubes]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def orientation_change(cubes, arm):
    """Mean over the seeds, per cube, in percent."""
    return [100 * float(np.mean([c["seeds"][s][arm]["orientation_change"] for s in c["seeds"]])) for c in cubes]


def numbers(cubes):
    """The primary measure and the guards' numbers on a set of cubes."""
    change = orientation_change(cubes, "orient_cube")
    qual = [c for c in cubes if c["fibre"]["disputed_n"] >= MIN_DISPUTED]
    pos = sum(c["fibre"]["disputed_S"] > 0 for c in qual)
    return dict(cubes=len(cubes),
                consistency=dict(default=arm_mean(cubes, "default", "consistency"),
                                 orient_cube=arm_mean(cubes, "orient_cube", "consistency")),
                consistency_diff=paired(cubes, "orient_cube", "consistency"),
                drift=dict(default=arm_mean(cubes, "default", "drift"), orient_cube=arm_mean(cubes, "orient_cube", "drift")),
                drift_diff=paired(cubes, "orient_cube", "drift"),
                orientation_change=dict(orient_cube=float(np.mean(change)) if change else None,
                                        orient_cube_max=max(change) if change else None,
                                        default=float(np.mean(orientation_change(cubes, "default"))) if cubes else None),
                fibre=dict(cubes_with_data=len(qual), positive=int(pos), share=pos / len(qual) if qual else None),
                guard_triggers=sum(c["guard_triggers"] > 0 for c in cubes),
                sign_margin_min=min((c["sign_margin_min"] for c in cubes), default=None),
                axis_share_min=min((c["axis_share_min"] for c in cubes), default=None))


def mean_of(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def decide(cubes, planned):
    """The plan's rule. The decision uses all the cubes; the id stretch is reported beside it."""
    all_ = numbers(cubes)
    low = all_["consistency_diff"]["low"]
    drift_low = all_["drift_diff"]["low"]
    change = all_["orientation_change"]["orient_cube"]
    fibre = all_["fibre"]["share"]
    checks = dict(primary=low is not None and low > 0, drift=drift_low is not None and drift_low > -1,
                  orientation_change=change is not None and change <= 1.0, fibre=fibre is not None and fibre >= 0.75)
    keep = [c["arms"]["orient_cube_keep"] for c in cubes]
    third = dict(betti_equal=sum(a["betti_equal"] for a in keep), merger_events_zero=sum(a["merger_events"] == 0 for a in keep),
                 cubes=len(keep), passed=len(keep) == planned and all(a["betti_equal"] and a["merger_events"] == 0 for a in keep),
                 consistency=arm_mean(cubes, "orient_cube_keep", "consistency"),
                 consistency_minus_orient_cube=paired(cubes, "orient_cube_keep", "consistency", "orient_cube"),
                 drift=arm_mean(cubes, "orient_cube_keep", "drift"),
                 drift_minus_orient_cube=paired(cubes, "orient_cube_keep", "drift", "orient_cube"),
                 contested=float(np.mean([a["contested"] for a in keep])) if keep else None,
                 band_moved_over_1_contested=mean_of(a.get("band_moved_over_1_contested") for a in keep),
                 band_moved_over_1_elsewhere=mean_of(a.get("band_moved_over_1_elsewhere") for a in keep))
    a, b = STRETCH
    return dict(complete=len(cubes) == planned, planned=planned, checks=checks,
                adopt=len(cubes) == planned and all(checks.values()), all=all_,
                stretch_161_280=numbers([c for c in cubes if a <= c["id"] <= b]),
                other_ids=numbers([c for c in cubes if not a <= c["id"] <= b]), third_arm=third)


def csv_row(c):
    r = dict(cube=c["cube"], k=c["k"], in_161_280=STRETCH[0] <= c["id"] <= STRETCH[1], samples=c["n_samples"], points=c["n_points"],
             fixed_points=c["n_fixed"], axis_share=c["axis_share"], axis_share_min=c["axis_share_min"],
             sign_margin=c["sign_margin"], sign_margin_min=c["sign_margin_min"], guard_triggers=c["guard_triggers"],
             against_vote=c["arms"]["orient_cube"]["orientation"]["against_vote"], disputed=c["disputed"],
             fibre_disputed_n=c["fibre"]["disputed_n"], fibre_disputed_S=c["fibre"]["disputed_S"],
             fibre_S=c["fibre"]["S"], fibre_placebo=c["fibre"]["placebo"])
    change = {arm: 100 * float(np.mean([c["seeds"][s][arm]["orientation_change"] for s in c["seeds"]])) for arm in ARMS}
    for arm in ARMS:
        r.update({f"consistency_{arm}": cube_mean(c, arm, "consistency"), f"drift_{arm}": cube_mean(c, arm, "drift"),
                  f"orientation_change_{arm}": change[arm], f"betti_equal_{arm}": c["arms"][arm]["betti_equal"],
                  f"merger_events_{arm}": c["arms"][arm]["merger_events"], f"contested_{arm}": c["arms"][arm]["contested"]})
    keep = c["arms"]["orient_cube_keep"]
    r.update(keep_band_moved_over_1_contested=keep["band_moved_over_1_contested"],
             keep_band_moved_over_1_elsewhere=keep["band_moved_over_1_elsewhere"])
    return r


def run_one(img_path, lab_path, out_json, k):
    os.environ["SNAP_THREADS"] = "1"          # one process per core already
    r = measure_cube(img_path, lab_path, k)
    with open(out_json + ".tmp", "w") as f:
        json.dump(r, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--cubes", nargs="+", help="the list of cubes; default: the 100 in results/orient_test_plan.md")
    ap.add_argument("--only", nargs="+", help="run only these cubes of the list")
    a = ap.parse_args()

    cubes = a.cubes or plan_lists()["drawn"]
    cube_dir = os.path.join(a.out_dir, "cubes")
    os.makedirs(cube_dir, exist_ok=True)
    jobs = [(k, n) for k, n in enumerate(cubes) if not a.only or n in a.only]
    todo = [(k, n) for k, n in jobs if not os.path.exists(os.path.join(cube_dir, n + ".json"))]
    print(f"{len(cubes)} cubes in the list, {len(jobs)} chosen, {len(todo)} to run, {a.workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, os.path.join(a.in_dir, "images", n + ".tif"), os.path.join(a.in_dir, "labels", n + ".tif"),
                          os.path.join(cube_dir, n + ".json"), k): n for k, n in todo}
        for count, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                print(f"[{count}/{len(todo)}] {r['cube']}: axis share {r['axis_share']:.3f}, sign margin {r['sign_margin']:.3f}, "
                      f"disputed {r['disputed']:.1%}, consistency {cube_mean(r, 'default', 'consistency'):.1f} -> "
                      f"{cube_mean(r, 'orient_cube', 'consistency'):.1f}, {r['seconds']} s", flush=True)
            except Exception as e:  # keep going; report at the end
                failed.append(futs[fut])
                print(f"[{count}/{len(todo)}] {futs[fut]}: FAILED: {type(e).__name__}: {e}", flush=True)

    done = [json.load(open(os.path.join(cube_dir, n + ".json"))) for _, n in jobs
            if os.path.exists(os.path.join(cube_dir, n + ".json"))]
    result = decide(done, len(cubes))
    with open(os.path.join(a.out_dir, "orient_test.json"), "w") as f:
        json.dump(dict(result, params=Params().asdict()), f, indent=1)
    if done:
        rows = [csv_row(c) for c in done]
        with open(os.path.join(a.out_dir, "orient_test_cubes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(done)} of {len(cubes)} cubes; checks {result['checks']}; adopt: {result['adopt']}; "
          f"third arm passed: {result['third_arm']['passed']} -> {a.out_dir}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} cubes failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
