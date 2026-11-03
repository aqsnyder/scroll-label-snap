"""Unblind and score a finished A/B review.

Usage: python tools/score_review.py KEY_JSON ANSWERS_JSON [OUT_MD]
  ANSWERS_JSON: {item_id: {"choice": "A"|"B"|"same"|"unsure", "ms": ..., "set": ...}, ...}

Reports how often the snapped label was preferred, the original preferred, rated the same,
or judged impossible: per stratum, for the hazy cubes flagged in the automatic evaluation,
by direction of the move and per cube. Adds an exact sign test on the decisive answers,
catch-item accuracy (the original should win against a label pushed off the surface), the
reviewer's left/right bias and answer times. Answers stamped with a different set id than
the key are ignored.
"""
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict

HAZY = {"sample_00002", "sample_00008", "sample_00017", "sample_00018"}


def set_id(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def verdict(rec, ans):
    """What the reviewer preferred, in unblinded terms."""
    c = ans["choice"]
    if c in ("same", "unsure"):
        return c
    return rec["A"] if c == "A" else rec["B"]


def sign_test(k, n):
    """Two-sided exact binomial p-value for k successes out of n at p = 0.5."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def table(rows, title):
    lines = [f"### {title}", "",
             "| Group | Items | Snapped preferred | Original preferred | Same | Can't tell | Snapped share of decisive (95% CI) | Sign test p |",
             "|---|---|---|---|---|---|---|---|"]
    for name, vs in rows:
        n = len(vs)
        if n == 0:
            continue
        cnt = Counter(vs)
        pct = lambda k: f"{cnt[k]} ({100 * cnt[k] / n:.0f}%)"
        s, o = cnt["snapped"], cnt["original"]
        lo, hi = wilson(s, s + o)
        share = f"{100 * s / (s + o):.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)" if s + o else "-"
        p = sign_test(s, s + o)
        lines.append(f"| {name} | {n} | {pct('snapped')} | {pct('original')} | {pct('same')} | "
                     f"{pct('unsure')} | {share} | {p:.3g} |" if s + o else
                     f"| {name} | {n} | {pct('snapped')} | {pct('original')} | {pct('same')} | {pct('unsure')} | - | - |")
    return "\n".join(lines)


def main(key_path, answers_path, out_md=None):
    key = json.load(open(key_path))
    answers = json.load(open(answers_path))
    sid = set_id(key)
    foreign = [q for q, a in answers.items() if a.get("set", sid) != sid]
    answers = {q: a for q, a in answers.items() if q in key and q not in foreign}

    by = defaultdict(list)
    per_cube = defaultdict(list)
    catch = defaultdict(lambda: [0, 0])
    sides, times = Counter(), []
    for qid, rec in key.items():
        if qid not in answers:
            continue
        ans = answers[qid]
        if ans["choice"] in ("A", "B"):
            sides[ans["choice"]] += 1
        if isinstance(ans.get("ms"), (int, float)):
            times.append(ans["ms"] / 1000)
        v = verdict(rec, ans)
        if rec["stratum"] == "catch":
            direction = "pushed into the gap (+3.0)" if rec["delta"] > 0 else "pushed into the sheet (-2.5)"
            catch[direction][0] += v == "original"
            catch[direction][1] += 1
            continue
        by[rec["stratum"]].append(v)
        if rec["stratum"] == "no move":
            continue
        by["all moved items"].append(v)
        by["hazy cubes (2, 8, 17, 18)" if rec["cube"] in HAZY else "other cubes"].append(v)
        by["label moved with +normal (delta > 0)" if rec["delta"] > 0 else "label moved with -normal (delta < 0)"].append(v)
        per_cube[rec["cube"]].append(v)

    rows = [(k, by[k]) for k in ("big move", "moderate move", "all moved items", "hazy cubes (2, 8, 17, 18)",
                                 "other cubes", "label moved with +normal (delta > 0)",
                                 "label moved with -normal (delta < 0)")]
    out = [table(rows, "Blind review: which label sits on the papyrus surface?"), ""]

    ctrl = Counter(by["no move"])
    ctrl_iou = [rec["window_iou"] for rec in key.values() if rec["stratum"] == "no move" and "window_iou" in rec]
    if by["no move"]:
        what = (f"the two labels agree across the whole window, IoU {min(ctrl_iou):.2f}-{max(ctrl_iou):.2f}"
                if ctrl_iou else "snapped and original agree within 0.25 voxel at the item's centre; the rest of "
                                 "the window can differ")
        out.append(f"Controls ({what}): {ctrl['same']} of "
                   f"{len(by['no move'])} rated the same; {ctrl['snapped'] + ctrl['original']} got a preference "
                   f"({ctrl['snapped']} snapped, {ctrl['original']} original), {ctrl['unsure']} can't tell.")
    ok = sum(c[0] for c in catch.values())
    tot = sum(c[1] for c in catch.values())
    out.append(f"Catch items (original vs. the original pushed off the surface): original chosen in {ok} of {tot}"
               + "".join(f"; {name}: {c[0]} of {c[1]}" for name, c in sorted(catch.items())) + ".")
    if sides:
        out.append(f"Left/right: A chosen in {sides['A']} and B in {sides['B']} of the decisive answers "
                   f"(each candidate was A in exactly half the items).")
    if times:
        out.append(f"Answer time: median {statistics.median(times):.1f} s; {sum(t < 1 for t in times)} answers "
                   f"under 1 s.")
    out.append(f"Answered {len(answers)} of {len(key)} items (set {sid})."
               + (f" Ignored {len(foreign)} answer{'s' * (len(foreign) != 1)} from another set." if foreign else ""))

    out += ["", "### Per cube (moved items only)", "",
            "| Cube | Items | Snapped | Original | Same | Can't tell |", "|---|---|---|---|---|---|"]
    for cube in sorted(per_cube):
        c = Counter(per_cube[cube])
        tag = " (hazy)" if cube in HAZY else ""
        out.append(f"| {cube}{tag} | {len(per_cube[cube])} | {c['snapped']} | {c['original']} | {c['same']} | {c['unsure']} |")

    text = "\n".join(out)
    print(text)
    if out_md:
        with open(out_md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
