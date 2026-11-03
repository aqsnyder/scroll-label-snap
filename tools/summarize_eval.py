"""Combine eval_snap.py outputs into one table (markdown) and a figure.

Usage: python tools/summarize_eval.py OUT_PREFIX EVAL_JSON [EVAL_JSON ...] [--pair LO-HI]

The refiner's threshold grid is reported as the pair that is best on the cubes being summarized
(its most generous setting, picked on the test cubes themselves). --pair also reports a pair
picked beforehand, e.g. the best pair on the development cubes when summarizing held-out cubes.
"""
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(out_prefix, *paths, pair=None):
    cubes = []
    for pth in paths:
        with open(pth) as f:
            cubes += json.load(f)["cubes"]
    cubes.sort(key=lambda c: c["cube"])

    def col(method, key, section="drift_test"):
        return np.array([c[section][method][key] for c in cubes])

    def wavg(method, key):
        # pool over cubes, weighting by the number of evaluated points
        v = col(method, key)
        w = np.array([c["drift_test"][method]["n"] for c in cubes], float)
        return float(np.sum(v * w) / np.sum(w))

    rows = [("No correction (drifted labels)", "drifted (no correction)"),
            ("Existing refiner (alpha-composite)", "baseline")]
    if all("baseline, offset-corrected" in c["drift_test"] for c in cubes):
        rows += [("Existing refiner, its median offset removed", "baseline, offset-corrected"),
                 ("Existing refiner, thresholds from each cube's gap and papyrus levels, offset removed",
                  "baseline, adaptive thresholds, offset-corrected")]
    rows += [("Snapper", "snapper")]
    lines = ["| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |",
             "|---|---|---|---|"]
    for label, m in rows:
        lines.append(f"| {label} | {wavg(m, 'mean'):.2f} | {wavg(m, 'within_1'):.0%} | "
                     f"{wavg(m, 'over_2'):.0%} |")
    if all("baseline_grid" in c for c in cubes):
        def gavg(key, stat):
            v = np.array([c["baseline_grid"][key]["offset_corrected"][stat] for c in cubes])
            w = np.array([c["baseline_grid"][key]["offset_corrected"]["n"] for c in cubes], float)
            ok = w > 0
            return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
        keys = list(cubes[0]["baseline_grid"])
        best = min(keys, key=lambda k: gavg(k, "mean"))
        grid_rows = []
        if pair and pair != best:
            grid_rows.append(f"| Existing refiner, thresholds {pair} (picked beforehand), offset removed | "
                             f"{gavg(pair, 'mean'):.2f} | {gavg(pair, 'within_1'):.0%} | {gavg(pair, 'over_2'):.0%} |")
        note = "picked beforehand, and " if pair == best else ""
        grid_rows.append(f"| Existing refiner, thresholds {best} ({note}best of {len(keys)} pairs on these cubes), "
                         f"offset removed | {gavg(best, 'mean'):.2f} | {gavg(best, 'within_1'):.0%} | {gavg(best, 'over_2'):.0%} |")
        lines[len(lines) - 1:len(lines) - 1] = grid_rows
    lines += ["", "| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |",
              "|---|---|---|"]
    cons = [("Existing refiner", "baseline consistency")]
    if all("baseline, adaptive thresholds consistency" in c["drift_test"] for c in cubes):
        cons.append(("Existing refiner, thresholds from each cube's levels", "baseline, adaptive thresholds consistency"))
    for label, m in cons + [("Snapper", "snapper consistency")]:
        lines.append(f"| {label} | {wavg(m, 'mean'):.2f} | {wavg(m, 'within_1'):.0%} |")
    clean_rows = [("Original", "original"), ("Existing refiner", "baseline(original)")]
    if all("baseline adaptive(original)" in c["edge_alignment"] for c in cubes):
        clean_rows.append(("Existing refiner, thresholds from each cube's levels, offset removed",
                           "baseline adaptive(original)"))
    clean_rows.append(("Snapper", "snap(original)"))
    al = {k: np.mean([c["edge_alignment"][k]["mean_step"] for c in cubes]) for _, k in clean_rows}
    pos = {k: np.mean([c["edge_alignment"][k]["frac_positive"] for c in cubes]) for _, k in clean_rows}
    lines += ["", "| Clean labels | Intensity step across label | Papyrus on the correct side |",
              "|---|---|---|"]
    for label, k in clean_rows:
        lines.append(f"| {label} | {al[k]:.1f} | {pos[k]:.0%} |")
    moved = np.mean([c["clean_label_moves"]["frac_moved_1vox"] for c in cubes])
    med = np.median([c["clean_label_moves"]["median_signed"] for c in cubes])
    n_pts = sum(c["n_points"] for c in cubes)
    n_tr = sum(c["n_trusted"] for c in cubes)
    lines += ["", f"{len(cubes)} cubes, {n_pts:,} evaluated surface points, {n_tr:,} trusted points. "
              f"On clean labels the snapper moved {moved:.0%} of points by 1 voxel or more "
              f"(median signed shift {med:+.2f} voxels)."]
    table = "\n".join(lines)
    with open(out_prefix + ".md", "w") as f:
        f.write(table + "\n")
    print(table)

    # per-cube figure; one colour and marker per method in both panels
    x = np.arange(len(cubes))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    style = {"none": ("C0", "o"), "default": ("C1", "s"), "adaptive": ("C2", "D"), "snapper": ("C3", "^")}
    drift_lines = [("no correction", "drifted (no correction)", "none"),
                   ("existing refiner, default thresholds", "baseline", "default")]
    adaptive = "baseline, adaptive thresholds, offset-corrected"
    if all(adaptive in c["drift_test"] for c in cubes):
        drift_lines.append(("existing refiner, thresholds from the cube's levels, offset removed", adaptive, "adaptive"))
    drift_lines.append(("snapper", "snapper", "snapper"))
    for label, m, s in drift_lines:
        axes[0].plot(x, col(m, "within_1"), "-", marker=style[s][1], color=style[s][0], ms=4, label=label)
    axes[0].set_ylabel("share of trusted points within 1 voxel")
    axes[0].set_title("Drift test: recovering known drift")
    clean_style = {"original": ("original labels", "none"), "baseline(original)": ("", "default"),
                   "baseline adaptive(original)": ("", "adaptive"), "snap(original)": ("", "snapper")}
    for _, k in clean_rows:
        s = style[clean_style[k][1]]
        axes[1].plot(x, [c["edge_alignment"][k]["mean_step"] for c in cubes], "-", marker=s[1], color=s[0], ms=4,
                     label=clean_style[k][0] or None)
    axes[1].set_ylabel("mean intensity step across label")
    axes[1].set_title("Clean labels: how sharply labels sit on the papyrus edge")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([c["cube"].replace("sample_", "") for c in cubes], rotation=90,
                           fontsize=7 if len(cubes) <= 30 else 4)
        ax.grid(alpha=0.3)
    # one legend for both panels, below them (in the clean-label panel blue is the original labels)
    h, l = axes[0].get_legend_handles_labels()
    l[0] = "no correction (left) / original labels (right)"
    fig.legend(h, l, loc="lower center", ncol=len(h), fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(out_prefix + ".png", dpi=80)
    print("wrote", out_prefix + ".md", out_prefix + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_prefix")
    ap.add_argument("eval_json", nargs="+")
    ap.add_argument("--pair", help="threshold pair picked beforehand, e.g. 70-110")
    a = ap.parse_args()
    main(a.out_prefix, *a.eval_json, pair=a.pair)
