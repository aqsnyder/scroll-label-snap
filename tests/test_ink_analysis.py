"""The ink test's scoring tools on synthetic maps: label AUC/F1 and the grid shift, the tile table,
the blind review builder (counts, balance, spacing, validity, what the images show) and the scorer."""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
import ink_scores as isc  # noqa: E402
import make_ink_review as mir  # noqa: E402
import score_ink_review as sir  # noqa: E402
from scroll_label_snap.tifxyz import write_tifxyz  # noqa: E402

TOOLS = os.path.join(HERE, "..", "tools")


def brute_auc(s, t):
    pos, neg = s[t], s[~t]
    d = pos[:, None].astype(int) - neg[None, :].astype(int)
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def test_auc_f1_match_brute_force():
    rng = np.random.default_rng(3)
    t = rng.random(800) < 0.3
    s = np.clip(rng.normal(120, 30, 800) + 40 * t, 0, 255).astype(np.uint8) // 8 * 8   # ties on purpose
    r = isc.auc_f1(s, t)
    assert abs(r["auc"] - brute_auc(s, t)) < 1e-12
    best = 0.0
    for th in range(256):
        pred = s >= th
        tp, fp, fn = (pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum()
        best = max(best, 2 * tp / max(2 * tp + fp + fn, 1))
    assert abs(r["best_f1"] - best) < 1e-12
    assert isc.auc_f1(np.where(t, 200, 100).astype(np.uint8), t)["auc"] == 1.0


def test_overlap():
    a, b = isc.overlap((10, 12), (11, 9), 2, -3)
    A, B = np.arange(120).reshape(10, 12), np.zeros((11, 9))
    assert A[a].shape == B[b].shape
    assert (a[0].start, a[1].start, b[0].start, b[1].start) == (0, 3, 2, 0)


def render_zarr(path, valid, layers=21, value=90):
    vol = np.zeros((layers,) + valid.shape, np.uint8)
    vol[:, valid] = value
    z = zarr.open_array(str(path), mode="w", shape=vol.shape, chunks=(layers, 128, 128), dtype="u1")
    z[:] = vol
    z.attrs.update(pixels_per_vertex=20, rows=[0, (valid.shape[0] - 1) // 20 + 1], cols=[0, (valid.shape[1] - 1) // 20 + 1])
    return str(path)


def test_labels_and_sanity(tmp_path):
    rng = np.random.default_rng(0)
    H, W, dy, dx = 300, 260, 3, -2
    valid = np.ones((H, W), bool)
    valid[:, :20] = False
    r = render_zarr(tmp_path / "r.zarr", valid)
    ink = np.zeros((H, W), bool)
    ink[100:140, 50:200] = True
    good = np.where(ink, 180, 90).astype(np.uint8)
    noisy = rng.integers(60, 190, (H, W)).astype(np.uint8)
    tifffile.imwrite(tmp_path / "good.tif", good)
    tifffile.imwrite(tmp_path / "noisy.tif", noisy)
    # the labels' grid: lab[y + dy, x + dx] = ink[y, x]
    lab = np.zeros((H + 5, W + 1), np.uint8)
    (sa, sb) = isc.overlap((H, W), lab.shape, dy, dx)
    lab[sb] = ink[sa] * 255
    mask = np.zeros_like(lab)
    mask[sb] = 1
    for name, arr in (("lab", lab), ("mask", mask)):
        z = zarr.open_array(str(tmp_path / f"{name}.zarr"), mode="w", shape=arr.shape, dtype="u1")
        z[:] = arr
    json.dump(dict(inplane_shift_yx=[dy, dx]), open(tmp_path / "check.json", "w"))
    off = np.zeros(lab.shape, np.uint8)
    off[sb] = good[sa]
    tifffile.imwrite(tmp_path / "official.tif", off)
    subprocess.run([sys.executable, os.path.join(TOOLS, "ink_scores.py"), "labels", str(tmp_path / "out.json"),
                    "--map", f"original={tmp_path / 'good.tif'}", "--map", f"snapped={tmp_path / 'noisy.tif'}",
                    "--renders", r, r, "--labels", str(tmp_path / "lab.zarr"), "--mask", str(tmp_path / "mask.zarr"),
                    "--check", str(tmp_path / "check.json"), "--official", str(tmp_path / "official.tif")], check=True)
    res = json.load(open(tmp_path / "out.json"))
    assert res["scores"]["original"]["auc"] == 1.0 and res["scores"]["official"]["auc"] == 1.0
    assert abs(res["scores"]["snapped"]["auc"] - 0.5) < 0.05
    assert res["snapped_minus_original"]["auc"] < -0.4
    assert res["pixels"] == int(valid[sa].sum())
    subprocess.run([sys.executable, os.path.join(TOOLS, "ink_scores.py"), "sanity", str(tmp_path / "s.json"),
                    "--ours", str(tmp_path / "good.tif"), "--official", str(tmp_path / "official.tif"),
                    "--renders", r, "--check", str(tmp_path / "check.json")], check=True)
    assert json.load(open(tmp_path / "s.json"))["pearson_r"] > 0.999


def test_tiles(tmp_path):
    H, W = 41 * 20 + 1, 31 * 20 + 1                # 42 x 32 vertices, 20 pixels per vertex
    valid = np.ones((H, W), bool)
    r = render_zarr(tmp_path / "r.zarr", valid)
    rng = np.random.default_rng(1)
    p_o = rng.integers(70, 120, (H, W)).astype(np.uint8)
    p_s = p_o.copy()
    p_s[:, W // 2:] = np.clip(p_s[:, W // 2:].astype(int) + 40, 0, 255)     # right half gains
    # dense mesh (factor 7): |shift| grows to the right, flags in the right half
    factor, (hv, wv) = 7, (42, 32)
    Hd, Wd = (hv - 1) * factor + 1, (wv - 1) * factor + 1
    P = np.zeros((Hd, Wd, 3), np.float32)
    P[..., 1], P[..., 2] = np.mgrid[0:Hd, 0:Wd]
    P[..., 0] = 50
    shift = np.tile(np.linspace(0, 3, Wd, dtype=np.float32), (Hd, 1))
    flag = np.zeros((Hd, Wd), np.uint8)
    flag[:, Wd // 2:] = 255
    write_tifxyz(str(tmp_path / "dense.tifxyz"), P, np.ones((Hd, Wd), bool), {"scale": [0.35, 0.35]},
                 channels=dict(snap_shift=shift, snap_flag=flag, snap_limit=np.zeros_like(flag)))
    json.dump(dict(factor=factor), open(tmp_path / "snap.json", "w"))
    tifffile.imwrite(tmp_path / "o.tif", p_o)
    tifffile.imwrite(tmp_path / "s.tif", p_s)
    out = tmp_path / "tiles"
    subprocess.run([sys.executable, os.path.join(TOOLS, "ink_scores.py"), "tiles", str(out), "--original",
                    str(tmp_path / "o.tif"), "--snapped", str(tmp_path / "s.tif"), "--renders", r, r,
                    "--dense", str(tmp_path / "dense.tifxyz"), "--snap-json", str(tmp_path / "snap.json"),
                    "--name", "synth", "--tile", "128"], check=True)
    res = json.load(open(out / "synth_tiles.json"))
    assert res["tiles"] == len(res["rows"]) > 0
    right = [t for t in res["rows"] if t["j0"] >= W // 2]
    left = [t for t in res["rows"] if t["j1"] <= W // 2]
    assert all(t["d_mean"] > 0.1 for t in right) and all(abs(t["d_mean"]) < 1e-9 for t in left)
    assert all(t["flagged_share"] == 1.0 for t in right if t["j0"] >= W // 2 + 40)
    assert res["spearman"]["d_mean_vs_abs_shift"]["rho"] > 0.5
    for f in ("synth_tile_changes.png", "synth_change_map.png", "synth_extreme_tiles.png"):
        assert (out / f).stat().st_size > 1000


@pytest.fixture
def small_crops(monkeypatch):
    monkeypatch.setattr(mir, "CROP", 64)
    monkeypatch.setattr(mir, "HALF", 32)


def test_review_builder(tmp_path, small_crops, monkeypatch):
    from PIL import Image
    H, W = 1400, 1500
    valid = np.ones((H, W), bool)
    valid[600:700, :] = False                                   # a strip off the mesh
    r = render_zarr(tmp_path / "r.zarr", valid)
    rng = np.random.default_rng(5)
    p_o = rng.integers(64, 120, (H, W)).astype(np.uint8)
    p_s = p_o.copy()
    p_o[200:1300:50, :] = 180                                   # "ink" lines, >= 0.5
    for name, arr in (("o", p_o), ("s", p_s)):
        tifffile.imwrite(tmp_path / f"{name}.tif", arr)
    out, key_path = tmp_path / "rev", tmp_path / "key.json"
    monkeypatch.setattr(sys, "argv", ["make_ink_review.py", str(out), str(key_path), "--original", str(tmp_path / "o.tif"),
                                      "--snapped", str(tmp_path / "s.tif"), "--renders", r, r,
                                      "--practice-map", str(tmp_path / "o.tif"), "--practice-renders", r])
    mir.main()
    data = json.load(open(out / "items.json"))
    kd = json.load(open(key_path))
    key = kd["key"]
    assert data["set"] == sir.set_id(key) and kd["meta"]["pool"] == "higher map >= 0.5"
    kinds = [v["kind"] for v in key.values()]
    assert kinds.count("comparison") == 60 and kinds.count("control") == 10 and kinds.count("catch") == 8
    assert sum(1 for v in key.values() if v["kind"] == "comparison" and v["A"] == "original") == 30
    assert sum(1 for v in key.values() if v["kind"] == "control" and v["A"] == "original") == 5
    assert sum(1 for v in key.values() if v["kind"] == "catch" and v["A"].endswith("(sharp)")) == 4
    assert sorted(v["A"].split()[0] for v in key.values() if v["kind"] == "catch") == ["original"] * 4 + ["snapped"] * 4
    cs = [v["centre_yx"] for v in key.values()]
    for (a, b) in itertools.combinations(cs, 2):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 64
    for v in key.values():
        y, x = v["centre_yx"]
        assert 32 <= y <= H - 32 and 32 <= x <= W - 32 and v["crop_valid"] >= 0.9
        if v["kind"] == "comparison":
            assert v["pmax_centre"] >= 0.5
    # what the images show: display scaling, black off the mesh, the catch item's blurred side
    for it in data["items"]:
        v = key[it["id"]]
        A = np.asarray(Image.open(out / it["a"]))
        B = np.asarray(Image.open(out / it["b"]))
        assert A.shape == B.shape == (64, 64)
        y, x = v["centre_yx"]
        vc = valid[y - 32:y + 32, x - 32:x + 32]
        assert A[~vc].max(initial=0) == 0 and B[~vc].max(initial=0) == 0
        if v["kind"] == "catch":
            sharp, blur = (A, B) if v["A"].endswith("(sharp)") else (B, A)
            assert sharp[vc].std() > blur[vc].std()
        else:
            src = {"original": p_o, "snapped": p_s}[v["A"]]
            expect = np.rint(np.clip((src[y - 32:y + 32, x - 32:x + 32] / 255.0 - 0.25) / 0.5, 0, 1) * 255)
            assert np.abs(A[vc].astype(int) - expect[vc]).max() <= 1
    assert len(data["practice"]) == 4 and all((out / p["a"]).exists() for p in data["practice"])


def test_scorer():
    key = {}
    for i in range(60):
        key[f"c{i}"] = dict(kind="comparison", A="original" if i % 2 else "snapped", B="snapped" if i % 2 else "original")
    for i in range(10):
        key[f"k{i}"] = dict(kind="control", A="original", B="snapped")
    for i in range(8):
        key[f"x{i}"] = dict(kind="catch", A="original (sharp)" if i % 2 else "snapped (blurred)",
                            B="original (blurred)" if i % 2 else "snapped (sharp)")
    sid = sir.set_id(key)

    def answers(pref_snapped, n_dec, catch_right):
        out = {}
        for i in range(60):
            rec = key[f"c{i}"]
            if i < n_dec:
                want = "snapped" if i < pref_snapped else "original"
                out[f"c{i}"] = dict(choice="A" if rec["A"] == want else "B", ms=1500, set=sid)
            else:
                out[f"c{i}"] = dict(choice="same", ms=1500, set=sid)
        for i in range(8):
            rec = key[f"x{i}"]
            sharp = "A" if rec["A"].endswith("(sharp)") else "B"
            out[f"x{i}"] = dict(choice=sharp if i < catch_right else ("B" if sharp == "A" else "A"), ms=900, set=sid)
        out["k0"] = dict(choice="same", ms=800, set=sid)
        out["stale"] = dict(choice="A", set="other")
        return out

    res = sir.score(key, [("one", answers(40, 50, 8)), ("two", answers(38, 50, 5))])
    P = res["pooled"]
    assert (P["snapped_preferred"], P["original_preferred"]) == (78, 22)
    assert P["verdict"] == "snapping helps" and P["sign_p"] < 1e-6
    assert res["reviewers"][1]["catch_flag"] and not res["reviewers"][0]["catch_flag"]
    assert res["reviewers"][0]["answered"] == 69
    res2 = sir.score(key, [("one", answers(12, 50, 8))])
    assert res2["pooled"]["verdict"] == "snapping hurts"
    res3 = sir.score(key, [("one", answers(26, 50, 8))])
    assert res3["pooled"]["verdict"] == "no detectable difference"
    assert "Result: no detectable difference" in sir.report(res3)
