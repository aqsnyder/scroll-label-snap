"""Power of the matched-display review (results/ink_review_matched_plan.md), by simulation.

  python tools/ink_review_power.py OUT_JSON [--sims 4000] [--seed 321]

Each simulated reviewer answers every item of an arm once. An answer is decisive with probability
DEC, and a decisive answer picks the snapped map with probability expit(logit(p) + u + r), where u
is the item's offset (normal, sd ITEM_SD, shared by reviewers) and r the reviewer's (normal, sd
REVIEWER_SD, shared by items). p is not adjusted for those offsets, so with them the true pooled
share sits a little nearer 0.5 than p.

Each simulated arm is read as tools/score_ink_review.py reads the real one: the pooled share of
decisive answers with a 95% interval clustered by item, then matched_reading (less legible at
matched level, more legible, "dimmer, not shown less legible", inconclusive). Reported per design:
how often each reading comes out when p is 0.29 (the first review's share), 0.4 and 0.5, and for
the 20-item arms how often the interval lies below 0.5. The interval treats the reviewers as
fixed, so with reviewer offsets it is too narrow for a population of reviewers; the readings are
about the reviewers who took part.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_ink_review import READINGS, matched_reading  # noqa: E402


def simulate(rng, sims, n_items, n_rev, p, dec, item_sd, reviewer_sd):
    """Hits and decisive answers per simulated item: two (sims, n_items) arrays."""
    mu = np.log(p / (1 - p))
    u = rng.normal(0.0, item_sd, (sims, n_items, 1))
    r = rng.normal(0.0, reviewer_sd, (sims, 1, n_rev))
    pick = 1.0 / (1.0 + np.exp(-(mu + u + r)))
    decisive = rng.random((sims, n_items, n_rev)) < dec
    snapped = rng.random((sims, n_items, n_rev)) < pick
    return (decisive & snapped).sum(2).astype(float), decisive.sum(2).astype(float)


def cluster_ci(H, D, z=1.96):
    """Vectorised score_ink_review.cluster_share over simulations: share, lower and upper ends."""
    n_items = H.shape[1]
    k, n = H.sum(1), D.sum(1)
    s = k / np.maximum(n, 1)
    se = np.sqrt(((H - s[:, None] * D) ** 2).sum(1) * n_items / (n_items - 1)) / np.maximum(n, 1)
    return s, s - z * se, s + z * se, n


def readings(lo, hi, n):
    names = [matched_reading(a, b) if m > 0 else "no decisive answers" for a, b, m in zip(lo, hi, n)]
    return {k: names.count(k) / len(names) for k in READINGS + ("no decisive answers",) if names.count(k)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_json")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=321)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = dict(sims=a.sims, seed=a.seed, primary={}, small_arms={})
    print("60 comparisons at matched level: share of simulations giving 'less legible' when p = 0.29, "
          "and 'dimmer, not shown less legible' when p = 0.5")
    for R, dec, isd, rsd in itertools.product((2, 3), (0.88, 0.7, 0.5), (0.0, 1.0), (0.0, 0.3)):
        rec = {}
        for p in (0.29, 0.4, 0.5):
            H, D = simulate(rng, a.sims, 60, R, p, dec, isd, rsd)
            s, lo, hi, n = cluster_ci(H, D)
            rec[str(p)] = dict(readings=readings(lo, hi, n), mean_share=float(s.mean()),
                               mean_halfwidth=float(((hi - lo) / 2).mean()))
        name = f"R{R}_dec{dec}_item{isd}_reviewer{rsd}"
        out["primary"][name] = rec
        print(f"  {R} reviewers, {dec:.0%} decisive, item sd {isd}, reviewer sd {rsd}: "
              f"{rec['0.29']['readings'].get(READINGS[0], 0):.2f} / {rec['0.5']['readings'].get(READINGS[2], 0):.2f}"
              f"  (half-width {rec['0.5']['mean_halfwidth']:.3f})", flush=True)
    print("20-item arms (fixed scale, level only): share of simulations whose interval lies below 0.5 when p = 0.29")
    for R, dec, isd in itertools.product((2, 3), (0.88, 0.6), (0.0, 0.5)):
        H, D = simulate(rng, a.sims, 20, R, 0.29, dec, isd, 0.0)
        s, lo, hi, n = cluster_ci(H, D)
        name = f"R{R}_dec{dec}_item{isd}"
        out["small_arms"][name] = dict(below_half=float((hi < 0.5).mean()), mean_halfwidth=float(((hi - lo) / 2).mean()))
        print(f"  {R} reviewers, {dec:.0%} decisive, item sd {isd}: {out['small_arms'][name]['below_half']:.2f}"
              f"  (half-width {out['small_arms'][name]['mean_halfwidth']:.3f})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("->", a.out_json)


if __name__ == "__main__":
    main()
