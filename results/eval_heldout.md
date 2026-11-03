# Held-out test: 100 cubes never used during development

- **Which cubes.** `tools/fetch_cubes.py --count 100 --seed 2026 --exclude <the 20 development cubes>` picked 100 of the bucket's 892 cube pairs at random. That selection rule was pushed (commit 13ab04a) before the cubes were downloaded or evaluated. The names are in `heldout_selection.json`.
- **Settings.** The snapper and evaluation ran with the v2 defaults. Nothing was tuned on these cubes.
- **Run.** `tools/eval_folder.py` ran on Lysander (Ryzen AI Max+ 395) with 12 workers and took 47 minutes. Per-cube numbers are in `eval_heldout.json`.

| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |
|---|---|---|---|
| No correction (drifted labels) | 1.14 | 57% | 15% |
| Existing refiner (alpha-composite) | 2.66 | 18% | 62% |
| Snapper | 0.55 | 88% | 3% |

| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |
|---|---|---|
| Existing refiner | 1.36 | 63% |
| Snapper | 1.04 | 75% |

| Clean labels | Intensity step across label | Papyrus on the correct side |
|---|---|---|
| Original | 19.0 | 63% |
| Existing refiner | -6.2 | 45% |
| Snapper | 47.1 | 86% |

100 cubes, 10,746,898 evaluated surface points, 717,523 trusted points. On clean labels the snapper moved 70% of points by 1 voxel or more (median signed shift +0.00 voxels).

## Compared with the 20 development cubes

| | Development (20 cubes) | Held-out (100 cubes) |
|---|---|---|
| Drift test: within 1 voxel, no correction | 58% | 57% |
| Drift test: within 1 voxel, existing refiner | 9% | 18% |
| Drift test: within 1 voxel, snapper | 91% | 88% |
| Drift test: mean error, snapper (voxels) | 0.46 | 0.55 |
| Drift test: over 2 voxels, snapper | 1.3% | 3.0% |
| Consistency: within 1 voxel, snapper | 83% | 75% |
| Consistency: within 1 voxel, existing refiner | 61% | 63% |
| Clean labels: papyrus on the correct side, original | 68% | 63% |
| Clean labels: papyrus on the correct side, snapper | 91% | 86% |

Held-out cubes: 100, 10,746,898 evaluated surface points, 717,523 trusted points.
The snapper beats no correction on 97 of 100 cubes (median gain +31 points of 'within 1 voxel').
Cubes where it does not: sample_00184 (51% vs 55%, 2498 trusted points), sample_00186 (55% vs 61%, 452 trusted points), sample_00478 (49% vs 61%, 3886 trusted points).
