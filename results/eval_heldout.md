# Held-out test: 100 cubes never used during development

- **Which cubes.** `tools/fetch_cubes.py --count 100 --seed 2026 --exclude <the 20 development cubes>` picked 100 of the bucket's 892 cube pairs at random. That selection rule was pushed (commit 13ab04a) before the cubes were downloaded or evaluated. The names are in `heldout_selection.json`.
- **Settings.** The snapper and evaluation ran with the v2 defaults. Nothing was tuned on these cubes.
- **Run.** `tools/eval_folder.py` ran on Lysander (Ryzen AI Max+ 395) with 12 workers and took 47 minutes. Per-cube numbers are in `eval_heldout.json`.
- **Rerun with fairer refiner baselines** (3 Nov, 100 minutes). It adds the refiner with its median offset removed, with thresholds from each cube's own gap and papyrus levels, and with each of 13 threshold pairs. The clean-label row for the second comes from `tools/clean_alignment.py` (pending). Every number of the first run reproduced exactly on all 100 cubes. The pair 70-110 was picked on the development cubes beforehand; 50-110 is the best pair on these cubes, picked after seeing them, so it is the refiner's ceiling here.

| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |
|---|---|---|---|
| No correction (drifted labels) | 1.14 | 57% | 15% |
| Existing refiner (alpha-composite) | 2.66 | 18% | 62% |
| Existing refiner, its median offset removed | 2.20 | 33% | 46% |
| Existing refiner, thresholds from each cube's gap and papyrus levels, offset removed | 0.73 | 78% | 6% |
| Existing refiner, thresholds 70-110 (picked beforehand), offset removed | 0.77 | 77% | 7% |
| Existing refiner, thresholds 50-110 (best of 13 pairs on these cubes), offset removed | 0.69 | 80% | 5% |
| Snapper | 0.55 | 88% | 3% |

| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |
|---|---|---|
| Existing refiner | 1.36 | 63% |
| Existing refiner, thresholds from each cube's levels | 0.92 | 73% |
| Snapper | 1.04 | 75% |

| Clean labels | Intensity step across label | Papyrus on the correct side |
|---|---|---|
| Original | 19.0 | 63% |
| Existing refiner | -6.2 | 45% |
| Snapper | 47.1 | 86% |

100 cubes, 10,746,898 evaluated surface points, 717,523 trusted points. On clean labels the snapper moved 70% of points by 1 voxel or more (median signed shift +0.00 voxels).

## Compared with the 20 development cubes

Development numbers are from [`eval_dev.md`](eval_dev.md), with drift seed k for cube k as here.

| | Development (20 cubes) | Held-out (100 cubes) |
|---|---|---|
| Drift test: within 1 voxel, no correction | 57% | 57% |
| Drift test: within 1 voxel, existing refiner, default thresholds | 9% | 18% |
| Drift test: within 1 voxel, existing refiner, thresholds from each cube's levels, offset removed | 80% | 78% |
| Drift test: within 1 voxel, snapper | 89% | 88% |
| Drift test: mean error, snapper (voxels) | 0.52 | 0.55 |
| Drift test: over 2 voxels, snapper | 2.9% | 3.0% |
| Consistency: within 1 voxel, snapper | 81% | 75% |
| Consistency: within 1 voxel, existing refiner, thresholds from each cube's levels | 77% | 73% |
| Consistency: within 1 voxel, existing refiner, default thresholds | 61% | 63% |
| Clean labels: papyrus on the correct side, original | 67% | 63% |
| Clean labels: papyrus on the correct side, snapper | 91% | 86% |

Held-out cubes: 100, 10,746,898 evaluated surface points, 717,523 trusted points.
The snapper beats no correction on 97 of 100 cubes (median gain +31 points of 'within 1 voxel').
Cubes where it does not: sample_00184 (51% vs 55%, 2498 trusted points), sample_00186 (55% vs 61%, 452 trusted points), sample_00478 (49% vs 61%, 3886 trusted points).
The snapper does as well as or better than the refiner with each cube's own levels on 94 of 100 cubes (median gain +10 points). It does worse on sample_00478 (49% vs 59%), sample_00639 (79% vs 84%), sample_00404, sample_00614, sample_00394 and sample_00905 (2 points or less each).
