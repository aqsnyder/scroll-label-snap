# Development cubes: snapper vs the existing refiner, with fair baselines

The 20 cubes the method was built and tuned on, rerun with three fairer versions of VC3D's refiner (`tools/eval_folder.py`, then `tools/clean_alignment.py` for the refiner's clean-label row, then `tools/summarize_eval.py`). Cube k uses drift seed k, as in the held-out run. [`eval_v2.md`](eval_v2.md) used seed 0 for every cube, so its rows differ slightly; the code is the same.

| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |
|---|---|---|---|
| No correction (drifted labels) | 1.12 | 57% | 15% |
| Existing refiner (alpha-composite) | 3.09 | 9% | 75% |
| Existing refiner, its median offset removed | 2.67 | 20% | 61% |
| Existing refiner, thresholds from each cube's gap and papyrus levels, offset removed | 0.68 | 80% | 4% |
| Existing refiner, thresholds 70-110 (best of 13 pairs on these cubes), offset removed | 0.64 | 82% | 3% |
| Snapper | 0.52 | 89% | 3% |

| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |
|---|---|---|
| Existing refiner | 1.57 | 61% |
| Existing refiner, thresholds from each cube's levels | 0.81 | 77% |
| Snapper | 0.84 | 81% |

| Clean labels | Intensity step across label | Papyrus on the correct side |
|---|---|---|
| Original | 25.7 | 67% |
| Existing refiner | -17.8 | 35% |
| Existing refiner, thresholds from each cube's levels, offset removed | 35.7 | 75% |
| Snapper | 58.4 | 91% |

20 cubes, 2,500,819 evaluated surface points, 258,394 trusted points. On clean labels the snapper moved 69% of points by 1 voxel or more (median signed shift +0.00 voxels).
