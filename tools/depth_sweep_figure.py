"""The figure of the depth diagnostic (results/depth_sweep_plan.md), from the scores
run_depth_sweep.ps1 writes. Presentation only: every number comes from those files.

  python tools/depth_sweep_figure.py OUT_PNG --w052 W052_COMPARE_JSON [--w052-extra JSON] --o841 NAME=SHIFTS_JSON [--o841 ...]

Left: w052, the median change in the tiles' 99th-percentile ink probability against the original
map, for the whole mesh moved d voxels; the snapped mesh moved d voxels too when those runs exist
(else a reference line); the mirrored mesh as a reference line.
Right: PHerc. 0841, the change in AUC against the ink labels for the traced segments moved d
voxels, with 95% block-bootstrap intervals.
"""
import argparse
import json

import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # the validated categorical slots 1-3
SHIFTS = [("dm2", -2), ("dm1", -1), ("d0", 0), ("dp1", 1), ("dp2", 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_png")
    ap.add_argument("--w052", required=True)
    ap.add_argument("--w052-extra", help="more w052 maps (each with a vs_original comparison), merged in")
    ap.add_argument("--o841", action="append", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=150)

    w = json.load(open(a.w052))["maps"]
    if a.w052_extra:
        for k, v in json.load(open(a.w052_extra))["maps"].items():
            w[k] = v["vs_original"]
    have = lambda k: k in w and not w[k].get("missing")  # noqa: E731

    def series(keys_at_d, base):
        xs, ys = [], []
        for key, d in SHIFTS:
            k = keys_at_d.get(d)
            if d == 0 and base is not None:
                xs.append(0)
                ys.append(base)
            elif k and have(k):
                xs.append(d)
                ys.append(w[k]["d_p99_median"])
        return xs, ys

    xs, ys = series({-2: "dm2", -1: "dm1", 1: "dp1", 2: "dp2"}, 0.0)
    ax1.plot(xs, ys, color=SERIES[0], linewidth=1.8, marker="o", markersize=6, zorder=3,
             markeredgecolor="white", markeredgewidth=1.2, label="original mesh, moved")
    ax1.annotate("original mesh, moved", (xs[0], ys[0]), textcoords="offset points", xytext=(0, 7), ha="left",
                 fontsize=8, color=INK2)
    snapped_moved = have("snapped_dm1") or have("snapped_dp1")
    if snapped_moved and have("snapped"):
        sx, sy = series({-1: "snapped_dm1", 1: "snapped_dp1"}, w["snapped"]["d_p99_median"])
        ax1.plot(sx, sy, color=SERIES[1], linewidth=1.8, marker="o", markersize=6, zorder=3,
                 markeredgecolor="white", markeredgewidth=1.2, label="snapped mesh, moved")
        ax1.annotate("snapped mesh, moved", (sx[-1], sy[-1]), textcoords="offset points", xytext=(9, 0), ha="left",
                     va="center", fontsize=8, color=INK2)
        ax1.legend(frameon=False, fontsize=8, loc="center left")
    refs = [("mirrored", "mirrored mesh")] + ([] if snapped_moved else [("snapped", "snapped mesh")])
    for key, label in refs:
        if have(key):
            y = w[key]["d_p99_median"]
            ax1.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            ax1.annotate(f"{label} {y:+.3f}", (2.45, y), textcoords="offset points", xytext=(0, 3), ha="right",
                         va="bottom", fontsize=8, color=INK2)
    ax1.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax1.set_xlim(-2.5, 2.5)
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo, max(hi, 0.0) + 0.12 * (max(hi, 0.0) - lo))
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_xlabel("whole mesh moved, voxels (− into papyrus, + toward gap)")
    ax1.set_ylabel("median change in tile 99th percentile")
    ax1.set_title("w052: map level against the original map", loc="left", fontsize=10)
    ax1.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)

    for i, spec in enumerate(a.o841):
        name, path = spec.split("=", 1)
        s = json.load(open(path))["scores"]
        px, py, lo, hi = [], [], [], []
        for key, d in SHIFTS:
            if key not in s:
                continue
            m = s[key].get("minus_reference")
            if m is None:
                continue
            ci = m.get("auc_95") or {"lo": m["auc"], "hi": m["auc"]}
            px.append(d + (i - 1) * 0.08)
            py.append(m["auc"])
            lo.append(m["auc"] - ci["lo"])
            hi.append(ci["hi"] - m["auc"])
        c = SERIES[i % len(SERIES)]
        ax2.errorbar(px, py, yerr=[lo, hi], color=c, linewidth=1.8, marker="o", markersize=6, capsize=3,
                     elinewidth=1, markeredgecolor="white", markeredgewidth=1.2, label=name, zorder=3)
        ax2.annotate(name, (px[-1], py[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=INK2)
    ax2.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax2.set_xlim(-2.5, 3.6)
    ax2.set_xticks([-2, -1, 0, 1, 2])
    ax2.set_xlabel("whole segment moved, voxels (− into papyrus, + toward gap)")
    ax2.set_ylabel("change in AUC against the ink labels")
    ax2.set_title("PHerc. 0841: agreement with ink labels (95% intervals)", loc="left", fontsize=10)
    ax2.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out_png, facecolor="white")
    print(f"-> {a.out_png}")


if __name__ == "__main__":
    main()
