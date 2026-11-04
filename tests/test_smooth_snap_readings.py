"""tools/smooth_snap_readings.py (results/smooth_snap_plan.md) on synthetic score files, and
tools/fine_labels.py score's --save-draws."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import fine_labels as fl  # noqa: E402
import smooth_snap_readings as ssr  # noqa: E402


def score_file(path, pixels, reference, maps):
    """maps: {name: (difference, draws)}; the interval is the draws' 2.5-97.5 percentiles."""
    scores = {reference: dict(auc=0.7)}
    for name, (d, draws) in maps.items():
        lo, hi = np.percentile(draws, [2.5, 97.5])
        scores[name] = dict(auc=0.7 + d, minus_reference=dict(auc=d, auc_95=dict(
            lo=float(lo), hi=float(hi), resamples=len(draws), draws=[float(x) for x in draws])))
    json.dump(dict(pixels=pixels, reference=reference, scores=scores), open(path, "w"))
    return str(path)


def test_widen_and_seed_calls():
    assert ssr.widen(0.01, -0.017, 0.037) == pytest.approx([0.01 - 1.8 * 0.027, 0.01 + 1.8 * 0.027])
    assert ssr.seed_call(0.01, 0.2) == "gain" and ssr.seed_call(-0.2, -0.01) == "loss"
    assert ssr.seed_call(-0.01, 0.02) == "no detectable difference"
    assert ssr.across_seeds(0.06, (0.01, 0.11), 0.05, (0.005, 0.095)) == "gain at both seeds"
    assert ssr.across_seeds(-0.06, (-0.11, -0.01), -0.05, (-0.09, -0.01)) == "loss at both seeds"
    # both seeds above 0, but the smaller gain is under the gap between the seeds
    assert ssr.across_seeds(0.12, (0.05, 0.19), 0.03, (0.001, 0.06)) == "seeds disagree"
    assert ssr.across_seeds(0.06, (0.01, 0.11), 0.0, (-0.04, 0.04)) == "seeds disagree"
    assert ssr.across_seeds(0.01, (-0.03, 0.05), -0.01, (-0.05, 0.03)) == "no detectable difference at both seeds"


def test_pooling_by_pixels_and_draws(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 0.01, 1000)
    b = rng.normal(-0.02, 0.01, 1000)
    f1 = score_file(tmp_path / "t1.json", 100, "traced", dict(smooth=(0.02, a)))
    f2 = score_file(tmp_path / "t2.json", 300, "traced", dict(smooth=(-0.02, b)))
    rows = ssr.pool([f1, f2])
    p = rows["smooth"]["pooled"]
    assert rows["smooth"]["reference"] == "traced"
    assert p["diff"] == pytest.approx(0.25 * 0.02 - 0.75 * 0.02)
    lo, hi = np.percentile(0.25 * a + 0.75 * b, [2.5, 97.5])
    assert p["ci"] == pytest.approx([lo, hi]) and p["widened"] == pytest.approx(ssr.widen(p["diff"], lo, hi))
    assert [f["diff"] for f in rows["smooth"]["files"]] == [0.02, -0.02]


def test_readings_and_report(tmp_path):
    rng = np.random.default_rng(2)
    files = {}
    for seed, shift in (("42", 0.0), ("43", 0.004)):
        for t, n in (("s1", 1_000_000), ("s2", 600_000), ("s3", 900_000)):
            files.setdefault(("A", seed), []).append(score_file(
                tmp_path / f"{t}_{seed}.json", n, "traced",
                dict(dropin=(-0.03 + shift, rng.normal(-0.03 + shift, 0.012, 1000)),
                     smooth=(0.002 + shift, rng.normal(0.002 + shift, 0.012, 1000)))))
    man = dict(groups=dict(A={"42": files[("A", "42")], "43": files[("A", "43")]}), decide=["A", "smooth"])
    json.dump(man, open(tmp_path / "man.json", "w"))
    out = tmp_path / "readings.md"
    subprocess.run([sys.executable, os.path.join(TOOLS, "smooth_snap_readings.py"), str(tmp_path / "man.json"), str(out)],
                   check=True, capture_output=True)
    res = json.load(open(tmp_path / "readings.json"))
    d = res["decision"]
    assert d["reading"] == "no detectable difference at both seeds" and d["map"] == "smooth"
    # pooled sd about 0.012 * sqrt(sum w^2) ~ 0.0072, widened half-width ~ 0.025: a 0.05 loss is excluded
    assert d["no_loss_over_margin"] and 0 < d["largest_loss_not_excluded"] < 0.05
    assert res["groups"]["A"]["dropin"]["reading"] == "loss at both seeds"
    text = open(out, encoding="utf-8").read()
    assert "**Decision (smooth against the reference, group A): no detectable difference at both seeds.**" in text
    assert "loss of more than 0.05 is excluded" in text and "s2_42.json" in text


def test_missing_draws_and_mismatched_files(tmp_path):
    f = tmp_path / "nodraws.json"
    json.dump(dict(pixels=10, reference="traced", scores=dict(traced=dict(auc=0.7), smooth=dict(
        auc=0.71, minus_reference=dict(auc=0.01, auc_95=dict(lo=0.0, hi=0.02, resamples=10))))), open(f, "w"))
    with pytest.raises(SystemExit, match="save-draws"):
        ssr.pool([str(f)])
    g1 = score_file(tmp_path / "g1.json", 10, "traced", dict(smooth=(0.0, np.zeros(10))))
    g2 = score_file(tmp_path / "g2.json", 10, "dropin", dict(smooth=(0.0, np.zeros(10))))
    with pytest.raises(SystemExit, match="reference"):
        ssr.pool([g1, g2])


def test_fine_labels_keeps_draws_only_when_asked():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.random(n) < 0.3
    a = np.where(truth, 150, 100) + rng.integers(-60, 60, n)
    b = np.where(truth, 160, 100) + rng.integers(-60, 60, n)
    blocks = np.arange(n) // 40
    plain = fl.block_bootstrap_auc_diff(a, b, truth, blocks, 200)
    kept = fl.block_bootstrap_auc_diff(a, b, truth, blocks, 200, keep_draws=True)
    assert "draws" not in plain and {k: kept[k] for k in plain} == plain
    assert len(kept["draws"]) == kept["resamples"] == 200
    assert list(np.percentile(kept["draws"], [2.5, 97.5])) == pytest.approx([plain["lo"], plain["hi"]])
