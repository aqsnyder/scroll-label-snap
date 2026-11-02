| Method | Mean error (voxels) | Within 1 voxel | Over 2 voxels |
|---|---|---|---|
| No correction (drifted labels) | 1.10 | 57% | 16% |
| Existing refiner (alpha-composite) | 2.88 | 11% | 70% |
| Snapper | 0.54 | 87% | 3% |

| Consistency (drifted vs clean result) | Mean difference (voxels) | Within 1 voxel |
|---|---|---|
| Existing refiner | 1.55 | 60% |
| Snapper | 0.87 | 79% |

| Clean labels | Intensity step across label | Papyrus on the correct side |
|---|---|---|
| Original | 25.7 | 67% |
| Existing refiner | -16.2 | 36% |
| Snapper | 58.6 | 92% |

20 cubes, 2,500,819 evaluated surface points, 258,394 trusted points. On clean labels the snapper moved 69% of points by 1 voxel or more (median signed shift +0.00 voxels).
