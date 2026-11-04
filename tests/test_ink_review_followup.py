"""The follow-up options of tools/make_ink_review.py and tools/score_ink_review.py (results/ink_review_matched_plan.md),
on synthetic maps, and tools/ink_review_power.py. The byte-identity test compares against the builder as it was at
678121a, read with git; it is skipped where that commit isn't available (a shallow clone, no git)."""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)
import ink_review_power as irp  # noqa: E402
import make_ink_review as mir  # noqa: E402
import score_ink_review as sir  # noqa: E402

BEFORE = "678121a"          # main before these options: the builder that made the first review's set


def load_before(tmp_path):
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{BEFORE}:tools/make_ink_review.py"], capture_output=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {BEFORE} isn't available here")
    path = tmp_path / "make_ink_review_before.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("make_ink_review_before", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def render_zarr(path, valid):
    z = zarr.open_array(str(path), mode="w", shape=(3,) + valid.shape, chunks=(3, 128, 128), dtype=np.uint8)
    z[:] = np.broadcast_to(valid.astype(np.uint8) * 100, (3,) + valid.shape)
    return str(path)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(mir, "CROP", 64)
    monkeypatch.setattr(mir, "HALF", 32)


@pytest.fixture
def world(tmp_path):
    """Original with strokes; snapped = the same strokes, dimmer and compressed, plus its own small noise."""
    H, W = 1800, 1800
    valid = np.ones((H, W), bool)
    valid[600:700, :] = False
    r = render_zarr(tmp_path / "r.zarr", valid)
    rng = np.random.default_rng(5)
    base = ndi.gaussian_filter(rng.normal(size=(H, W)).astype(np.float32), 1.0)
    base = base / base.std()
    p_o = 90 + 10 * base
    for row in range(40, H - 40, 37):
        p_o[row:row + 5, :] += 90
    p_s = 75 + 0.8 * (p_o - 90) + 3 * rng.normal(size=(H, W))
    p_o = np.clip(np.rint(p_o), 0, 255).astype(np.uint8)
    p_s = np.clip(np.rint(p_s), 0, 255).astype(np.uint8)
    tifffile.imwrite(tmp_path / "o.tif", p_o)
    tifffile.imwrite(tmp_path / "s.tif", p_s)
    return dict(tmp=tmp_path, valid=valid, r=r, p_o=p_o, p_s=p_s)


def run(module, monkeypatch, w, name, extra=()):
    out, key_path = w["tmp"] / name, w["tmp"] / f"{name}_key.json"
    monkeypatch.setattr(sys, "argv", ["make_ink_review.py", str(out), str(key_path), "--original", str(w["tmp"] / "o.tif"),
                                      "--snapped", str(w["tmp"] / "s.tif"), "--renders", w["r"], w["r"],
                                      "--practice-map", str(w["tmp"] / "o.tif"), "--practice-renders", w["r"]] + list(extra))
    module.main()
    return out, key_path


def tree_hash(folder):
    h = {}
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            h[os.path.relpath(p, folder)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return h


def panels(out, it):
    return np.asarray(Image.open(out / it["a"])), np.asarray(Image.open(out / it["b"]))


def vc_of(valid, v):
    y, x = v["centre_yx"]
    return valid[y - 32:y + 32, x - 32:x + 32]


def display_of(p, y, x):
    return np.clip((p[y - 32:y + 32, x - 32:x + 32] / 255.0 - 0.25) / 0.5, 0, 1)


def test_default_is_byte_identical_to_the_tool_before_the_options(world, small, monkeypatch):
    old = load_before(world["tmp"])
    monkeypatch.setattr(old, "CROP", 64)
    monkeypatch.setattr(old, "HALF", 32)
    o1, k1 = run(old, monkeypatch, world, "old")
    o2, k2 = run(mir, monkeypatch, world, "new")
    assert tree_hash(o1) == tree_hash(o2)
    d1, d2 = json.load(open(k1)), json.load(open(k2))
    assert d1["key"] == d2["key"] and d1["meta"] == d2["meta"]


def test_midway_panels_have_the_same_histogram_and_keep_ranks(world, small, monkeypatch):
    out, kp = run(mir, monkeypatch, world, "mid", ["--display", "midway"])
    data, kd = json.load(open(out / "items.json")), json.load(open(kp))
    key = kd["key"]
    assert kd["meta"]["display"] == "midway"
    unmatched_gap, matched_gap = [], []
    for it in data["items"]:
        v = key[it["id"]]
        A, B = panels(out, it)
        vc = vc_of(world["valid"], v)
        assert A[~vc].max(initial=0) == 0 and B[~vc].max(initial=0) == 0
        assert np.abs(np.sort(A[vc]).astype(int) - np.sort(B[vc]).astype(int)).mean() < 3
        assert abs(float(A[vc].mean()) - float(B[vc].mean())) < 2
        assert abs(float(A[vc].std()) - float(B[vc].std())) < 2.5
        if v["kind"] in ("comparison", "control"):
            assert v["display"] == "midway"
            y, x = v["centre_yx"]
            src = {"original": world["p_o"], "snapped": world["p_s"]}[v["A"]][y - 32:y + 32, x - 32:x + 32]
            # one LUT: a source level never maps to two grey levels, and the map is monotone
            lv = np.unique(src[vc])
            assert max(np.unique(A[vc & (src == u)]).size for u in lv) == 1
            means = np.array([A[vc & (src == u)].mean() for u in lv])
            assert np.all(np.diff(means) >= 0)
            so, ss = display_of(world["p_o"], y, x), display_of(world["p_s"], y, x)
            unmatched_gap.append(abs(so[vc].std() - ss[vc].std()) * 255)
            matched_gap.append(abs(float(A[vc].std()) - float(B[vc].std())))
    assert np.mean(unmatched_gap) > 3 * np.mean(matched_gap)        # the synthetic snapped map really is dimmer


def test_catch_under_midway_still_differs_in_structure(world, small, monkeypatch):
    out, kp = run(mir, monkeypatch, world, "mid2", ["--display", "midway", "--n-comparisons", "20", "--n-controls", "0",
                                                    "--weak-catch", "4", "--weak-sigma", "2"])
    data, key = json.load(open(out / "items.json")), json.load(open(kp))["key"]
    kinds = [v["kind"] for v in key.values()]
    assert kinds.count("comparison") == 20 and kinds.count("control") == 0
    assert kinds.count("catch") == 8 and kinds.count("catch_weak") == 4
    for it in data["items"]:
        v = key[it["id"]]
        if v["kind"] not in ("catch", "catch_weak"):
            continue
        A, B = panels(out, it)
        sharp, blur = (A, B) if v["A"].endswith("(sharp)") else (B, A)
        vc = vc_of(world["valid"], v)
        hs = ndi.gaussian_gradient_magnitude(sharp.astype(np.float32), 1.0)[vc].mean()
        hb = ndi.gaussian_gradient_magnitude(blur.astype(np.float32), 1.0)[vc].mean()
        assert hs > hb
        assert abs(float(sharp[vc].std()) - float(blur[vc].std())) < 3


def test_fixed_arm_and_level_only(world, small, monkeypatch):
    out, kp = run(mir, monkeypatch, world, "arms", ["--display", "midway", "--n-comparisons", "20", "--n-controls", "0",
                                                    "--fixed-arm", "10", "--level-only", "10"])
    data, key = json.load(open(out / "items.json")), json.load(open(kp))["key"]
    kinds = [v["kind"] for v in key.values()]
    assert kinds.count("comparison_fixed") == 10 and kinds.count("level_only") == 10
    # no place twice, and no leak of the kind in what reviewers see
    cs = [tuple(v["centre_yx"]) for v in key.values()]
    assert len(set(cs)) == len(cs)
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            assert max(abs(cs[a][0] - cs[b][0]), abs(cs[a][1] - cs[b][1])) >= 64
    assert set(data["items"][0]) == {"id", "a", "b"}
    assert sum(v["A"] == "original" for v in key.values() if v["kind"] == "level_only") == 5
    assert sum(v["A"] == "original" for v in key.values() if v["kind"] == "comparison_fixed") == 5
    for it in data["items"]:
        v = key[it["id"]]
        y, x = v["centre_yx"]
        vc = vc_of(world["valid"], v)
        A, B = panels(out, it)
        if v["kind"] == "comparison_fixed":
            assert v["display"] == "fixed"
            src = {"original": world["p_o"], "snapped": world["p_s"]}[v["A"]]
            assert np.abs(A[vc].astype(int) - np.rint(display_of(src, y, x)[vc] * 255)).max() <= 1
        if v["kind"] == "level_only":
            assert v["display"] == "fixed"
            orig, dim = (A, B) if v["A"] == "original" else (B, A)
            exp_orig = np.rint(display_of(world["p_o"], y, x) * 255)
            assert np.abs(orig[vc].astype(int) - exp_orig[vc]).max() <= 1
            # structure: the dimmed panel is a monotone function of the original crop
            src = world["p_o"][y - 32:y + 32, x - 32:x + 32]
            lv = np.unique(src[vc])
            means = np.array([dim[vc & (src == u)].mean() for u in lv])
            assert np.all(np.diff(means) >= -1e-9)
            assert max(np.unique(dim[vc & (src == u)]).size for u in lv) == 1
            # level: the dimmed panel has the histogram of the snapped crop on the same scale
            exp = np.rint(display_of(world["p_s"], y, x) * 255)
            assert abs(float(dim[vc].mean()) - float(exp[vc].mean())) < 2


def test_reuse_places(world, small, monkeypatch):
    o1, k1 = run(mir, monkeypatch, world, "first", ["--n-comparisons", "20", "--n-controls", "4"])
    o2, k2 = run(mir, monkeypatch, world, "second", ["--seed", "99", "--display", "midway", "--reuse-places", str(k1)])
    a, b = json.load(open(k1))["key"], json.load(open(k2))["key"]
    for kind in ("comparison", "control"):
        assert (sorted(v["centre_yx"] for v in a.values() if v["kind"] == kind)
                == sorted(v["centre_yx"] for v in b.values() if v["kind"] == kind))
    # catch places are new and do not overlap the reused ones
    old = [v["centre_yx"] for v in a.values() if v["kind"] in ("comparison", "control")]
    for v in b.values():
        if v["kind"] == "catch":
            assert all(max(abs(v["centre_yx"][0] - c[0]), abs(v["centre_yx"][1] - c[1])) >= 64 for c in old)


def test_too_many_items_stops_instead_of_overlapping(world, small, monkeypatch):
    with pytest.raises(SystemExit):
        run(mir, monkeypatch, world, "toomany", ["--n-comparisons", "20", "--n-controls", "0", "--level-only", "100000"])


def test_scorer_keeps_extra_arms_out_of_the_decision():
    key = {}
    for i in range(10):
        key[f"q{i:03d}"] = dict(kind="comparison", A="original", B="snapped")
    key["q100"] = dict(kind="level_only", A="original", B="original at the snapped level")
    key["q101"] = dict(kind="comparison_fixed", A="snapped", B="original")
    key["q102"] = dict(kind="catch_weak", A="original (sharp)", B="original (blurred)")
    key["q103"] = dict(kind="catch", A="snapped (sharp)", B="snapped (blurred)")
    ans = {f"q{i:03d}": dict(choice="B", ms=1000) for i in range(10)}
    ans.update(q100=dict(choice="A", ms=1), q101=dict(choice="B", ms=1), q102=dict(choice="B", ms=1),
               q103=dict(choice="A", ms=1))
    res = sir.score(key, [("r", ans)])
    r = res["reviewers"][0]
    assert r["comparisons"] == {"snapped": 10}
    assert r["extra_arms"] == {"level_only": {"original": 1}, "comparison_fixed": {"original": 1},
                               "catch_weak": {"original (blurred)": 1}}
    assert r["catch_right"] == 1 and r["catch_total"] == 1
    assert res["pooled"]["snapped_preferred"] == 10
    assert res["arms"]["level_only"]["hits"] == 0 and res["arms"]["catch_weak"]["hits"] == 0
    assert "matched" not in res                      # comparisons without "display": not a matched set
    assert "Extra arms" in sir.report(res)


def test_cluster_share_by_hand():
    a = sir.cluster_share([(2, 2), (0, 2), (1, 2), (1, 1)], 5)    # a fifth item got no answer
    s = 4 / 7
    se = math.sqrt(((2 - 2 * s) ** 2 + (0 - 2 * s) ** 2 + (1 - 2 * s) ** 2 + (1 - s) ** 2) * 5 / 4) / 7
    assert a["share"] == pytest.approx(s) and a["se"] == pytest.approx(se)
    assert a["ci"] == pytest.approx([s - 1.96 * se, s + 1.96 * se]) and a["decisive"] == 7 and a["items"] == 5
    assert sir.cluster_share([], 60)["share"] is None


def test_matched_reading():
    assert sir.matched_reading(0.30, 0.45) == "less legible at matched level"
    assert sir.matched_reading(0.52, 0.70) == "more legible at matched level"
    assert sir.matched_reading(0.35, 0.60) == "dimmer, not shown less legible"
    assert sir.matched_reading(0.30, 0.50) == "dimmer, not shown less legible"     # 0.5 inside the interval
    assert sir.matched_reading(0.20, 0.55) == "inconclusive"
    assert sir.matched_reading(0.29, 0.55) == "inconclusive"                        # must lie above 0.29


def matched_key(n_comp=60):
    key, q = {}, 0
    plan = ([("comparison", "midway")] * n_comp + [("comparison_fixed", "fixed")] * 20 + [("level_only", "fixed")] * 20
            + [("catch", "midway")] * 8 + [("catch_weak", "midway")] * 4)
    for i, (kind, disp) in enumerate(plan):
        if kind in ("catch", "catch_weak"):
            ab = ("original (sharp)", "original (blurred)")
        elif kind == "level_only":
            ab = ("original", "original at the snapped level")
        else:
            ab = ("original", "snapped")
        if i % 2:
            ab = ab[::-1]
        key[f"q{q:03d}"] = dict(kind=kind, A=ab[0], B=ab[1], display=disp)
        q += 1
    return key


def answer(key, pick):
    """Answers that choose, item by item, the label pick(qid, rec) returns (or 'same'/'unsure')."""
    out = {}
    for q, rec in key.items():
        want = pick(q, rec)
        c = want if want in ("same", "unsure") else ("A" if rec["A"] == want else "B")
        out[q] = dict(choice=c, ms=2000, set=sir.set_id(key))
    return out


def test_scorer_reads_a_matched_set():
    key = matched_key()
    comps = [q for q, v in key.items() if v["kind"] == "comparison"]

    def reviewer(j):
        def pick(q, rec):
            if rec["kind"] == "comparison":
                return "snapped" if (comps.index(q) + j) % 2 else "original"      # half and half
            if rec["kind"] in ("catch", "catch_weak"):
                return rec["A"] if rec["A"].endswith("(sharp)") else rec["B"]
            if rec["kind"] == "level_only":
                return "original"
            return "original" if int(q[1:]) % 3 else "snapped"
        return pick
    res = sir.score(key, [(f"r{j}", answer(key, reviewer(j))) for j in range(3)])
    assert res["matched"]["reading"] == "dimmer, not shown less legible"
    a = res["arms"]["comparison"]
    assert a["hits"] == 90 and a["decisive"] == 180 and a["display"] == ["midway"]
    assert res["arms"]["level_only"]["share"] == 0.0 and res["arms"]["level_only"]["target"] == "original at the snapped level"
    assert res["arms"]["catch"]["share"] == 1.0 and res["arms"]["catch_weak"]["decisive"] == 12
    text = sir.report(res)
    assert text.startswith("# Ink legibility review at matched level")
    assert "**Reading: dimmer, not shown less legible.**" in text and "Controls (random" not in text
    assert "Comparisons, fixed scale" in text and "Weak catch" in text and "Only the first row decides" in text


def test_scorer_flags_a_smaller_loss():
    key = matched_key()
    comps = [q for q, v in key.items() if v["kind"] == "comparison"]

    def reviewer(j):
        def pick(q, rec):
            if rec["kind"] != "comparison":
                return "same"
            i = comps.index(q)
            return "snapped" if j == 0 or (j == 1 and i < 12) else "original"     # 72 of 180: 0.4
        return pick
    res = sir.score(key, [(f"r{j}", answer(key, reviewer(j))) for j in range(3)])
    a = res["arms"]["comparison"]
    assert a["share"] == pytest.approx(0.4) and a["ci"][1] < 0.5 and a["ci"][0] > 0.29
    assert res["matched"] == dict(reading="less legible at matched level", above_fixed=True, ref=0.29)
    assert "a smaller loss than on the fixed display" in sir.report(res)


def test_first_review_report_is_unchanged():
    res_dir = os.path.join(REPO, "results")
    key = json.load(open(os.path.join(res_dir, "ink_review_key.json")))["key"]
    revs = []
    for name in ("reviewer1", "reviewer2"):
        d = json.load(open(os.path.join(res_dir, f"ink_review_answers_{name}.json")))
        revs.append((d.get("reviewer", name), d["answers"]))
    text = sir.report(sir.score(key, revs))
    committed = open(os.path.join(res_dir, "ink_review.md"), encoding="utf-8").read()
    assert committed.startswith(text)
    assert "matched" not in sir.score(key, revs)


def test_power_tool_uses_the_scorers_interval():
    rng = np.random.default_rng(0)
    H, D = irp.simulate(rng, 5, 60, 3, 0.29, 0.88, 1.0, 0.3)
    s, lo, hi, n = irp.cluster_ci(H, D)
    for i in range(5):
        a = sir.cluster_share([(h, d) for h, d in zip(H[i], D[i]) if d or h], 60)
        assert a["share"] == pytest.approx(s[i]) and a["ci"] == pytest.approx([lo[i], hi[i]])
    r = irp.readings(np.array([0.3, 0.35]), np.array([0.45, 0.6]), np.array([10, 10]))
    assert r == {"less legible at matched level": 0.5, "dimmer, not shown less legible": 0.5}
