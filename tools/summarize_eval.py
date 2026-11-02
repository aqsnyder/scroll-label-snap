"""Combine eval_snap.py outputs into one table (markdown) and a figure.

Usage: python tools/summarize_eval.py OUT_PREFIX EVAL_JSON [EVAL_JSON ...]
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(out_prefix, *paths):
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
            ("Existing refiner (alpha-composite)", "baseline"),
            ("Snapper", "snapper")]
    lines = ["| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |",
             "|---|---|---|---|"]
    for label, m in rows:
        lines.append(f"| {label} | {wavg(m, 'mean'):.2f} | {wavg(m, 'within_1'):.0%} | "
                     f"{wavg(m, 'over_2'):.0%} |")
    lines += ["", "| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |",
              "|---|---|---|"]
    for label, m in [("Existing refiner", "baseline consistency"), ("Snapper", "snapper consistency")]:
        lines.append(f"| {label} | {wavg(m, 'mean'):.2f} | {wavg(m, 'within_1'):.0%} |")
    al = {k: np.mean([c["edge_alignment"][k]["mean_step"] for c in cubes])
          for k in ("original", "snap(original)", "baseline(original)")}
    pos = {k: np.mean([c["edge_alignment"][k]["frac_positive"] for c in cubes])
           for k in al}
    lines += ["", "| Clean labels | Intensity step across label | Papyrus on the correct side |",
              "|---|---|---|"]
    for label, k in [("Original", "original"), ("Existing refiner", "baseline(original)"),
                     ("Snapper", "snap(original)")]:
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

    # per-cube figure
    x = np.arange(len(cubes))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for label, m, mk in [("no correction", "drifted (no correction)", "o"),
                         ("existing refiner", "baseline", "s"), ("snapper", "snapper", "^")]:
        axes[0].plot(x, col(m, "within_1"), mk + "-", label=label)
    axes[0].set_ylabel("share of trusted points within 1 voxel")
    axes[0].set_title("Drift test: recovering known drift")
    axes[1].plot(x, [c["edge_alignment"]["original"]["mean_step"] for c in cubes], "o-", label="original")
    axes[1].plot(x, [c["edge_alignment"]["baseline(original)"]["mean_step"] for c in cubes], "s-",
                 label="existing refiner")
    axes[1].plot(x, [c["edge_alignment"]["snap(original)"]["mean_step"] for c in cubes], "^-",
                 label="snapper")
    axes[1].set_ylabel("mean intensity step across label")
    axes[1].set_title("Clean labels: how sharply labels sit on the papyrus edge")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([c["cube"].replace("sample_", "") for c in cubes], rotation=90, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=80)
    print("wrote", out_prefix + ".md", out_prefix + ".png")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
