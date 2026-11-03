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
"""
import hashlib
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_review import sign_test, wilson  # noqa: E402


def set_id(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def unblind(rec, choice):
    if choice in ("same", "unsure"):
        return choice
    return rec["A"] if choice == "A" else rec["B"]


def score(key, reviewers):
    sid = set_id(key)
    out = dict(set=sid, reviewers=[], pooled={})
    pooled = Counter()
    for name, answers in reviewers:
        ans = {q: v for q, v in answers.items() if q in key and v.get("set") in (None, sid)}
        comp, ctrl, catch = Counter(), Counter(), Counter()
        sides, times = Counter(), []
        for q, v in ans.items():
            rec, c = key[q], v["choice"]
            u = unblind(rec, c)
            if c in ("A", "B"):
                sides[c] += 1
            if "ms" in v:
                times.append(v["ms"])
            if rec["kind"] == "comparison":
                comp[u] += 1
            elif rec["kind"] == "control":
                ctrl[u] += 1
            else:
                catch["right" if u.endswith("(sharp)") else "wrong"] += 1
        k, n = comp["snapped"], comp["snapped"] + comp["original"]
        pooled.update(comp)
        dec = sides["A"] + sides["B"]
        out["reviewers"].append(dict(
            name=name, answered=len(ans), comparisons=dict(comp), controls=dict(ctrl),
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
    return out


def fmt_share(r):
    if r.get("snapped_share") is None:
        return "-"
    lo, hi = r["ci"]
    return f"{r['snapped_share']:.0%} ({lo:.0%}-{hi:.0%})"


def report(res):
    P = res["pooled"]
    c = P["comparisons"]
    lines = [f"# Ink legibility review: set `{res['set']}`", "",
             f"**Result: {P['verdict']}.** On the 60 comparisons, pooled over {len(res['reviewers'])} reviewer(s), "
             f"the snapped map was preferred {P['snapped_preferred']} times and the original {P['original_preferred']} "
             f"times (snapped {fmt_share(P)} of decisive answers; two-sided exact sign test p = {P['sign_p']:.3g}). "
             f"{c.get('same', 0)} answers were \"about the same\" and {c.get('unsure', 0)} \"can't tell\".", "",
             "| Reviewer | Answered | Snapped preferred | Original preferred | Same | Can't tell | Snapped share of decisive (95% CI) | Sign test p | Catch items right | A share | Median time |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    num = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    for r in res["reviewers"]:
        cc = r["comparisons"]
        flag = " (below 6: reported)" if r["catch_flag"] else ""
        lines.append(f"| {r['name']} | {r['answered']} | {cc.get('snapped', 0)} | {cc.get('original', 0)} | "
                     f"{cc.get('same', 0)} | {cc.get('unsure', 0)} | {fmt_share(r)} | {num(r['sign_p'], '.3g')} | "
                     f"{r['catch_right']} of {r['catch_total']}{flag} | {num(r['share_A'], '.0%')} | "
                     f"{num(r['median_seconds'], '.1f')} s |")
    lines += ["", "Controls (random places, no ink condition):", ""]
    for r in res["reviewers"]:
        cc = r["controls"]
        lines.append(f"- {r['name']}: snapped {cc.get('snapped', 0)}, original {cc.get('original', 0)}, "
                     f"same {cc.get('same', 0)}, can't tell {cc.get('unsure', 0)}")
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
