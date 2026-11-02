| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |
|---|---|---|---|
| No correction (drifted labels) | 1.10 | 58% | 14% |
| Existing refiner (alpha-composite) | 3.09 | 9% | 75% |
| Snapper | 0.46 | 91% | 1% |

| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |
|---|---|---|
| Existing refiner | 1.53 | 61% |
| Snapper | 0.79 | 83% |

| Clean labels | Intensity step across label | Papyrus on the correct side |
|---|---|---|
| Original | 25.7 | 67% |
| Existing refiner | -17.8 | 35% |
| Snapper | 58.4 | 91% |

20 cubes, 2,500,819 evaluated surface points, 258,394 trusted points. On clean labels the snapper moved 69% of points by 1 voxel or more (median signed shift +0.00 voxels).
