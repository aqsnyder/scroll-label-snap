"""The readings of the smooth-snap test (results/smooth_snap_plan.md), from tools/fine_labels.py
score files written with --save-draws.

  python tools/smooth_snap_readings.py MANIFEST_JSON OUT_MD

MANIFEST_JSON: {"groups": {GROUP: {"42": [SCORE_JSON, ...], "43": [SCORE_JSON, ...]}, ...},
                "decide": [GROUP, MAP]}
Each score file is one target (or pair) at one seed; within a group all files share a reference map
and the other maps' names. For every group, seed and map against the reference:
  - per file: the AUC difference, the bootstrap's 95% interval, and that interval widened 1.8 times
    about the estimate (a rolled-label null ran about 1.8 times the bootstrap's standard error on
    these targets, results/review_2026_11.md);
  - pooled: the mean weighted by scored pixels, with a 95% interval from the files' bootstrap draws
    combined draw by draw with the same weights, widened the same way.
Readings, on the widened intervals: at one seed, a gain if it lies above 0, a loss if below, and no
detectable difference otherwise. Over the two seeds: "gain at both seeds" or "loss at both seeds" if
both seeds agree and both estimates are larger than the difference between them, "no detectable
difference at both seeds" if neither seed shows one, and "seeds disagree" otherwise. The largest
loss not excluded is minus the lower of the two widened lower ends. Writes OUT_MD and OUT_MD's
.json beside it.
"""
import json
import os
import sys

import numpy as np

WIDEN, MARGIN = 1.8, 0.05
SEEDS = ("42", "43")


def widen(est, lo, hi, k=WIDEN):
    return [est - k * (est - lo), est + k * (hi - est)]


def seed_call(lo, hi):
    return "gain" if lo > 0 else "loss" if hi < 0 else "no detectable difference"


def across_seeds(e1, w1, e2, w2):
    c1, c2 = seed_call(*w1), seed_call(*w2)
    gap = abs(e1 - e2)
    if c1 == c2 == "gain" and min(e1, e2) > gap:
        return "gain at both seeds"
    if c1 == c2 == "loss" and max(e1, e2) < -gap:
        return "loss at both seeds"
    if c1 == c2 == "no detectable difference":
        return "no detectable difference at both seeds"
    return "seeds disagree"


def contrasts(score):
    """The maps of one score file other than its reference: {map: (difference, lo, hi, draws)}."""
    out = {}
    for name, sc in score["scores"].items():
        d = sc.get("minus_reference")
        if not d or not d.get("auc_95"):
            continue
        ci = d["auc_95"]
        if "draws" not in ci:
            sys.exit(f"{score.get('_path', '?')}: no bootstrap draws for {name} (score with --save-draws)")
        out[name] = (d["auc"], ci["lo"], ci["hi"], np.asarray(ci["draws"], np.float64))
    return out


def pool(files):
    """Per-file and pooled rows for each map of one group at one seed."""
    scores = []
    for f in files:
        s = json.load(open(f))
        s["_path"] = f
        scores.append(s)
    refs = sorted({s["reference"] for s in scores})
    if len(refs) != 1:
        sys.exit(f"the files don't share a reference map: {refs}")
    ref = refs[0]
    per = [contrasts(s) for s in scores]
    names = set(per[0])
    if any(set(p) != names for p in per):
        sys.exit("the files don't hold the same maps")
    w = np.array([s["pixels"] for s in scores], np.float64)
    w = w / w.sum()
    rows = {}
    for m in sorted(names):
        n_draws = min(len(p[m][3]) for p in per)
        draws = sum(wi * p[m][3][:n_draws] for wi, p in zip(w, per))
        est = float(sum(wi * p[m][0] for wi, p in zip(w, per)))
        lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
        files_out = [dict(file=s["_path"], pixels=s["pixels"], diff=p[m][0], ci=[p[m][1], p[m][2]],
                          widened=widen(p[m][0], p[m][1], p[m][2])) for s, p in zip(scores, per)]
        rows[m] = dict(reference=ref, pooled=dict(diff=est, ci=[lo, hi], widened=widen(est, lo, hi), draws=int(n_draws)),
                       files=files_out)
    return rows


def readings(manifest):
    out = dict(widen=WIDEN, margin=MARGIN, groups={})
    for g, by_seed in manifest["groups"].items():
        if set(by_seed) != set(SEEDS):
            sys.exit(f"group {g}: need seeds {SEEDS}, got {sorted(by_seed)}")
        if len(by_seed["42"]) != len(by_seed["43"]):
            sys.exit(f"group {g}: the seeds have different numbers of files")
        seeds = {k: pool(by_seed[k]) for k in SEEDS}
        res = {}
        for m in seeds["42"]:
            a, b = seeds["42"][m]["pooled"], seeds["43"][m]["pooled"]
            worst = min(a["widened"][0], b["widened"][0])
            per_file = []
            for fa, fb in zip(seeds["42"][m]["files"], seeds["43"][m]["files"]):
                per_file.append(dict(files=[fa["file"], fb["file"]], diffs=[fa["diff"], fb["diff"]],
                                     reading=across_seeds(fa["diff"], fa["widened"], fb["diff"], fb["widened"])))
            res[m] = dict(reference=seeds["42"][m]["reference"], seed42=seeds["42"][m], seed43=seeds["43"][m],
                          reading=across_seeds(a["diff"], a["widened"], b["diff"], b["widened"]),
                          largest_loss_not_excluded=max(0.0, -worst), no_loss_over_margin=worst > -MARGIN,
                          per_file=per_file)
        out["groups"][g] = res
    g, m = manifest.get("decide", [None, None])
    if g is not None:
        r = out["groups"][g][m]
        out["decision"] = dict(group=g, map=m, reading=r["reading"], no_loss_over_margin=r["no_loss_over_margin"],
                               largest_loss_not_excluded=r["largest_loss_not_excluded"])
    return out


def fmt(p):
    lo, hi = p["ci"]
    wl, wh = p["widened"]
    return f"{p['diff']:+.3f} ({lo:+.3f} to {hi:+.3f}; widened {wl:+.3f} to {wh:+.3f})"


def report(res):
    lines = ["# Smooth-snap test: readings", "",
             f"Intervals: the block bootstrap's 95% interval, then that interval widened {res['widen']} times about "
             "the estimate, which the readings use (results/smooth_snap_plan.md).", ""]
    if "decision" in res:
        d = res["decision"]
        extra = (f" A pooled loss of more than {res['margin']:.2f} is excluded at both seeds."
                 if d["no_loss_over_margin"] else
                 f" A pooled loss of up to {d['largest_loss_not_excluded']:.3f} isn't excluded.")
        lines += [f"**Decision ({d['map']} against the reference, group {d['group']}): {d['reading']}.**{extra}", ""]
    lines += ["| Group | Map − reference | Pooled, seed 42 | Pooled, seed 43 | Reading | Largest loss not excluded |",
              "|---|---|---|---|---|---|"]
    for g, res_g in res["groups"].items():
        for m, r in res_g.items():
            lines.append(f"| {g} | {m} − {r['reference']} | {fmt(r['seed42']['pooled'])} | {fmt(r['seed43']['pooled'])} "
                         f"| {r['reading']} | {r['largest_loss_not_excluded']:.3f} |")
    lines += ["", "Per target or pair:", "", "| Group | Map − reference | File, seed 42 | Seed 42 | Seed 43 | Reading |",
              "|---|---|---|---|---|---|"]
    for g, res_g in res["groups"].items():
        for m, r in res_g.items():
            for fa, fb, pf in zip(r["seed42"]["files"], r["seed43"]["files"], r["per_file"]):
                lines.append(f"| {g} | {m} − {r['reference']} | {os.path.basename(fa['file'])} | {fmt(fa)} | {fmt(fb)} "
                             f"| {pf['reading']} |")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res = readings(json.load(open(sys.argv[1])))
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(report(res))
    with open(sys.argv[2].rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(report(res))


if __name__ == "__main__":
    main()
