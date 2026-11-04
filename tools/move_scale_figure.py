"""The figure of diagnostic 2 (results/move_scale_plan.md), from the comparison
run_move_scale.ps1 writes. Presentation only: every number comes from that file.

  python tools/move_scale_figure.py OUT_PNG --compare W052_COMPARE_MOVES_JSON
         [--shifts W052_COMPARE_JSON] --bar KEY=LABEL [--bar ...]

One bar per mesh: the median change in the tiles' 99th-percentile ink probability against the
original map, with the plan's "keeps" (-0.005) and "loses" (-0.015) lines. --shifts adds the
range the original mesh gave when moved 1-2 voxels (the depth diagnostic) as a band.
"""
import argparse
import json

INK, INK2, GRID = "#0b0b0b", "#52514e", "#c3c2b7"
BAR = "#2a78d6"
BAND = "#e7f1fb"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_png")
    ap.add_argument("--compare", required=True)
    ap.add_argument("--shifts")
    ap.add_argument("--bar", action="append", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    maps = json.load(open(a.compare))["maps"]
    bars = [b.split("=", 1) for b in a.bar]
    bars = [(k, lab) for k, lab in bars if k in maps and not maps[k].get("missing")]
    vals = [maps[k]["d_p99_median"] for k, _ in bars]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(bars) + 1.5), dpi=150)
    y = list(range(len(bars)))[::-1]
    if a.shifts:
        s = json.load(open(a.shifts))["maps"]
        sv = [s[k]["d_p99_median"] for k in ("dm2", "dm1", "dp1", "dp2") if k in s and not s[k].get("missing")]
        if sv:
            ax.axvspan(min(sv + [0.0]), max(sv + [0.0]) + 0.0004, color=BAND, zorder=0,
                       label="range of the original mesh moved 1–2 voxels")
            ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.barh(y, vals, height=0.62, color=BAR, zorder=2)
    for yi, v in zip(y, vals):
        ax.annotate("0.000" if abs(v) < 5e-4 else f"{v:+.3f}", (v, yi), textcoords="offset points", xytext=(-4, 0), ha="right", va="center",
                    fontsize=8, color=INK2)
    for x, lab in ((-0.005, "keeps"), (-0.015, "loses")):
        ax.axvline(x, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(lab, (x, len(bars) - 0.42), fontsize=8, color=INK2, ha="center", va="bottom", annotation_clip=False)
    ax.axvline(0, color=INK2, linewidth=1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in bars])
    lo = min(vals + [-0.02])
    ax.set_xlim(lo * 1.25, 0.006)
    ax.set_ylim(-1.6, len(bars) - 0.1)
    ax.set_xlabel("median change in tile 99th percentile against the original map")
    ax.set_title("w052: which part of the snapper's moves lowers the map level", loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(a.out_png, facecolor="white")
    print(f"-> {a.out_png}")


if __name__ == "__main__":
    main()
