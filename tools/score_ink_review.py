"""Unblind and score the ink legibility review (results/ink_test_plan.md, primary outcome).

Usage: python tools/score_ink_review.py KEY_JSON OUT_MD ANSWERS_JSON [ANSWERS_JSON ...]
  KEY_JSON:     from make_ink_review.py ({"key": {item: ...}, "meta": ...})
  ANSWERS_JSON: one per reviewer, {"reviewer": NAME, "answers": {item: {"choice": "A"|"B"|"same"|"unsure",
                "ms": ..., "set": ...}}}. Answers stamped with another set id are ignored.

The decision, as pre-registered: the decisive answers (A or B) on the 60 comparisons, pooled over
reviewers, with a two-sided exact sign test. Snapping helps if p < 0.05 and most decisive answers
prefer the snapped map, hurts if p < 0.05 and most prefer the original, and otherwise there is no
detectable difference. Per-reviewer results, catch items (a reviewer with fewer than 6 of 8 right
is reported, and still counts), controls, side bias and answer times are reported too.

A follow-up set whose comparisons are shown at matched level (make_ink_review.py --display midway)
is read as results/ink_review_matched_plan.md fixes instead: the snapped share of decisive answers
on those comparisons, pooled over reviewers, with a 95% interval clustered by item (a delta-method
standard error of the ratio), then matched_reading. Every arm gets the same summary; only the
matched comparisons decide. The report of a set without matched comparisons is unchanged.
"""
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_review import sign_test, wilson  # noqa: E402

FIXED_SHARE = 0.29       # the first review's snapped share of decisive answers, 31 of 106, rounded
TARGET = {"comparison": "snapped", "comparison_fixed": "snapped", "control": "snapped",
          "level_only": "original at the snapped level"}
READINGS = ("less legible at matched level", "more legible at matched level", "dimmer, not shown less legible",
            "inconclusive")


def set_id(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def hit(rec, u):
    """Whether an unblinded decisive answer picks the arm's target: the snapped map, the dimmed copy of a
    level-only item, or the sharp panel of a catch item."""
    if rec["kind"] in ("catch", "catch_weak"):
        return u.endswith("(sharp)")
    return u == TARGET.get(rec["kind"], "snapped")


def cluster_share(counts, n_items, z=1.96):
    """Pooled share sum(hits) / sum(decisive) over an arm, with a 95% interval from a delta-method
    standard error clustered by item. counts: (hits, decisive) per answered item; n_items: every item
    of the arm (an unanswered item adds nothing to the sums, only to the item count)."""
    k, n = sum(h for h, _ in counts), sum(d for _, d in counts)
    if n == 0 or n_items < 2:
        return dict(items=n_items, hits=k, decisive=n, share=None, se=None, ci=None)
    s = k / n
    se = math.sqrt(sum((h - s * d) ** 2 for h, d in counts) * n_items / (n_items - 1)) / n
    return dict(items=n_items, hits=k, decisive=n, share=s, se=se, ci=[s - z * se, s + z * se])


def matched_reading(lo, hi, ref=FIXED_SHARE):
    """The pre-registered reading of the matched comparisons from their interval (lo, hi)."""
    if hi < 0.5:
        return READINGS[0]
    if lo > 0.5:
        return READINGS[1]
    if lo > ref:
        return READINGS[2]
    return READINGS[3]


def is_matched(key):
    comp = [v for v in key.values() if v["kind"] == "comparison"]
    return bool(comp) and all(v.get("display") == "midway" for v in comp)


def unblind(rec, choice):
    if choice in ("same", "unsure"):
        return choice
    return rec["A"] if choice == "A" else rec["B"]


def score(key, reviewers):
    sid = set_id(key)
    out = dict(set=sid, reviewers=[], pooled={})
    pooled = Counter()
    per_item = {}                    # item: [hits, decisive, same, can't tell], over reviewers
    for name, answers in reviewers:
        ans = {q: v for q, v in answers.items() if q in key and v.get("set") in (None, sid)}
        comp, ctrl, catch = Counter(), Counter(), Counter()
        extra = {}                   # the extra arms of a follow-up review, by kind; never part of the decision
        sides, times = Counter(), []
        for q, v in ans.items():
            rec, c = key[q], v["choice"]
            u = unblind(rec, c)
            t = per_item.setdefault(q, [0, 0, 0, 0])
            if c in ("A", "B"):
                t[0] += hit(rec, u)
                t[1] += 1
            else:
                t[2 if c == "same" else 3] += 1
            if c in ("A", "B"):
                sides[c] += 1
            if "ms" in v:
                times.append(v["ms"])
            if rec["kind"] == "comparison":
                comp[u] += 1
            elif rec["kind"] == "control":
                ctrl[u] += 1
            elif rec["kind"] == "catch":
                catch["right" if u.endswith("(sharp)") else "wrong"] += 1
            else:
                extra.setdefault(rec["kind"], Counter())[u] += 1
        k, n = comp["snapped"], comp["snapped"] + comp["original"]
        pooled.update(comp)
        dec = sides["A"] + sides["B"]
        out["reviewers"].append(dict(
            name=name, answered=len(ans), comparisons=dict(comp), controls=dict(ctrl),
            extra_arms={k: dict(v) for k, v in extra.items()},
            catch_right=catch["right"], catch_total=catch["right"] + catch["wrong"],
            catch_flag=(catch["right"] + catch["wrong"]) > 0 and catch["right"] < 6,
            snapped_share=k / n if n else None, ci=wilson(k, n) if n else None, sign_p=sign_test(k, n) if n else None,
            share_A=sides["A"] / dec if dec else None, median_seconds=statistics.median(times) / 1000 if times else None))
    k, n = pooled["snapped"], pooled["snapped"] + pooled["original"]
    p = sign_test(k, n) if n else float("nan")
    if n and p < 0.05:
        verdict = "snapping helps" if k > n - k else "snapping hurts"
    else:
        verdict = "no detectable difference"
    out["pooled"] = dict(comparisons=dict(pooled), decisive=n, snapped_preferred=k, original_preferred=n - k,
                         snapped_share=k / n if n else None, ci=wilson(k, n) if n else None, sign_p=p, verdict=verdict)
    arms = {}
    for kind in sorted(set(v["kind"] for v in key.values())):
        qs = [q for q, v in key.items() if v["kind"] == kind]
        got = [per_item[q] for q in qs if q in per_item]
        arm = cluster_share([(t[0], t[1]) for t in got], len(qs))
        arm.update(target="the sharp panel" if kind in ("catch", "catch_weak") else TARGET.get(kind, "snapped"),
                   same=sum(t[2] for t in got), unsure=sum(t[3] for t in got),
                   display=sorted(set(key[q].get("display", "fixed") for q in qs)))
        arms[kind] = arm
    out["arms"] = arms
    if is_matched(key):
        m = arms["comparison"]
        if m["ci"] is None:
            out["matched"] = dict(reading="no decisive answers", above_fixed=None, ref=FIXED_SHARE)
        else:
            out["matched"] = dict(reading=matched_reading(*m["ci"]), above_fixed=m["ci"][0] > FIXED_SHARE,
                                  ref=FIXED_SHARE)
    return out


def fmt_share(r):
    if r.get("snapped_share") is None:
        return "-"
    lo, hi = r["ci"]
    return f"{r['snapped_share']:.0%} ({lo:.0%}-{hi:.0%})"


ARM_NAMES = {"comparison": "Comparisons at matched level (the decision)", "comparison_fixed": "Comparisons, fixed scale",
             "level_only": "Level only, fixed scale", "catch": "Catch", "catch_weak": "Weak catch", "control": "Controls"}


def fmt_arm(a):
    if a["share"] is None:
        return "-"
    lo, hi = a["ci"]
    return f"{a['share']:.0%} ({lo:.0%} to {hi:.0%})"


def matched_header(res):
    m, a, P = res["matched"], res["arms"]["comparison"], res["pooled"]
    lines = [f"# Ink legibility review at matched level: set `{res['set']}`", ""]
    if a["share"] is None:
        lines += ["**Reading: no decisive answers on the comparisons at matched level.**"]
    else:
        lo, hi = a["ci"]
        more = (" The interval also lies above the first review's 29%: a smaller loss than on the fixed display."
                if m["reading"] == READINGS[0] and m["above_fixed"] else "")
        lines += [f"**Reading: {m['reading']}.** On the {a['items']} comparisons shown at matched level, pooled over "
                  f"{len(res['reviewers'])} reviewer(s), the snapped map was preferred {a['hits']} times and the original "
                  f"{a['decisive'] - a['hits']} times: snapped {a['share']:.0%} of decisive answers, with a 95% interval "
                  f"clustered by item of {lo:.0%} to {hi:.0%}. {a['same']} answers were \"about the same\" and "
                  f"{a['unsure']} \"can't tell\". As fixed in `results/ink_review_matched_plan.md`: less legible if the "
                  f"interval lies below 50%, more legible if above, \"dimmer, not shown less legible\" if it includes 50% "
                  f"and lies above the first review's {m['ref']:.0%}, and inconclusive otherwise.{more}"]
    lines += ["", "| Arm | Display | Items | Decisive answers | Counted as a hit | Hit share of decisive (95% interval, by item) | Same | Can't tell |",
              "|---|---|---|---|---|---|---|---|"]
    order = ["comparison", "comparison_fixed", "level_only", "catch", "catch_weak", "control"]
    for kind in sorted(res["arms"], key=lambda k: (order.index(k) if k in order else len(order), k)):
        r = res["arms"][kind]
        lines.append(f"| {ARM_NAMES.get(kind, kind)} | {', '.join(r['display'])} | {r['items']} | {r['decisive']} | "
                     f"{r['target']} | {fmt_arm(r)} | {r['same']} | {r['unsure']} |")
    lines += ["", f"Only the first row decides. For comparison only, the first review's rule on the comparisons at "
                  f"matched level (a pooled two-sided exact sign test) gives: {P['verdict']} (p = {P['sign_p']:.3g}).", ""]
    return lines


def report(res):
    P = res["pooled"]
    c = P["comparisons"]
    if "matched" in res:
        lines = matched_header(res)
    else:
        lines = [f"# Ink legibility review: set `{res['set']}`", "",
                 f"**Result: {P['verdict']}.** On the 60 comparisons, pooled over {len(res['reviewers'])} reviewer(s), "
                 f"the snapped map was preferred {P['snapped_preferred']} times and the original {P['original_preferred']} "
                 f"times (snapped {fmt_share(P)} of decisive answers; two-sided exact sign test p = {P['sign_p']:.3g}). "
                 f"{c.get('same', 0)} answers were \"about the same\" and {c.get('unsure', 0)} \"can't tell\".", ""]
    lines += ["| Reviewer | Answered | Snapped preferred | Original preferred | Same | Can't tell | Snapped share of decisive (95% CI) | Sign test p | Catch items right | A share | Median time |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    num = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    for r in res["reviewers"]:
        cc = r["comparisons"]
        flag = " (below 6: reported)" if r["catch_flag"] else ""
        lines.append(f"| {r['name']} | {r['answered']} | {cc.get('snapped', 0)} | {cc.get('original', 0)} | "
                     f"{cc.get('same', 0)} | {cc.get('unsure', 0)} | {fmt_share(r)} | {num(r['sign_p'], '.3g')} | "
                     f"{r['catch_right']} of {r['catch_total']}{flag} | {num(r['share_A'], '.0%')} | "
                     f"{num(r['median_seconds'], '.1f')} s |")
    if "matched" not in res or "control" in res["arms"]:
        lines += ["", "Controls (random places, no ink condition):", ""]
        for r in res["reviewers"]:
            cc = r["controls"]
            lines.append(f"- {r['name']}: snapped {cc.get('snapped', 0)}, original {cc.get('original', 0)}, "
                         f"same {cc.get('same', 0)}, can't tell {cc.get('unsure', 0)}")
    if any(r["extra_arms"] for r in res["reviewers"]):
        lines += ["", "Extra arms (not part of the decision):", ""]
        for r in res["reviewers"]:
            for kind, cc in sorted(r["extra_arms"].items()):
                lines.append(f"- {r['name']}, {kind}: " + ", ".join(f"{k}: {v}" for k, v in sorted(cc.items())))
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    data = json.load(open(sys.argv[1]))
    key = data["key"] if "key" in data else data
    reviewers = []
    for path in sys.argv[3:]:
        d = json.load(open(path))
        reviewers.append((d.get("reviewer", path), d["answers"]))
    res = score(key, reviewers)
    open(sys.argv[2], "w").write(report(res))
    json.dump(res, open(sys.argv[2].rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(report(res))


if __name__ == "__main__":
    main()
