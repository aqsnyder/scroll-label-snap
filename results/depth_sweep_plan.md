# Diagnostic: how much does the ink map depend on where the surface sits?

This was written and pushed before any of the maps below existed. It follows the ink test ([`ink_test.md`](ink_test.md)), where snapping made w052's ink maps less legible, and asks why. It is a diagnostic, not a new test of the tool. Nothing here changes the ink test's outcomes.

## Questions

1. **Depth sensitivity.** How much does the ink map change when the whole surface moves 1 or 2 voxels along its normal? The snapper moved w052's mesh by 1.69 voxels on average, in both directions.
2. **Direction.** Does it matter which way the snapper moved the mesh? A mirrored mesh, moved as far the other way, separates the size of the move from its direction.

## Runs

All runs use the ink test's model and settings: `ink_9um` seed 42, the same inference options, and the forward layer direction.

- **Uniform shifts, without re-rendering.**
  - Each render has 21 layers, 1 voxel apart, and the model reads the middle 17 (layers 2–18).
  - villa's `--layer-start 2+d --layer-end 19+d` reads layers 2+d to 18+d instead. That moves the surface d voxels along its normal: d > 0 toward the gap, d < 0 into the papyrus.
  - The shifts run:
    - w052, original mesh: d = −2, −1, +1, +2.
    - w035, original mesh: d = −2, −1, 0, +1, +2.
    - PHerc. 0841, the three traced segments (their A.1 renders from the follow-up): d = −2, −1, 0, +1, +2.
  - d = 0 with the same options must reproduce the existing map exactly. That is the check that the window does what it should.
  - Optional, last: w052, snapped mesh, d = −1 and +1.
- **Mirrored snap.**
  - w052: a mesh moved by the snapper's shift the other way, 2 × original − snapped at each vertex of the drop-in output (`tools/depth_sweep.py mirror`). It is rendered with the ink test's renderer (c5ffa62) and settings.
  - PHerc. 0841: each traced segment rendered at minus the snapped dense depth, the follow-up's A.2 depth negated (`tools/depth_sweep.py negate`, then `tools/render_depth.py` as in the follow-up).

## Measures

- **w052.** Each map is compared with the ink test's original map (`tools/depth_sweep.py compare`), over pixels valid in the original, snapped and mirrored renders. The measures are those of the ink test's descriptive analysis:
  - in 256-pixel tiles with at least half their pixels valid, the median change in the 99th percentile, and the share of tiles losing or gaining more than 0.02;
  - the whole-map mean and 99th percentile;
  - the Pearson r between the maps.

  The ink test's snapped map is compared the same way, and is the yardstick.
- **w035:** AUC and best F1 against its ink labels at each d (`tools/ink_scores.py labels`). This is biased toward d = 0, as before.
- **PHerc. 0841:** ΔAUC against the traced map (d = 0), with the follow-up's block-bootstrap 95% interval (`tools/fine_labels.py score`), and the pooled mean weighted by pixels.
  - The mirrored depth is scored together with the snapped one (A.2), on the same pixels.
  - It is scored twice: against the traced map, and against the snapped map.

## Reading, fixed now

- **Depth sensitivity**, on w052. Take the shift with the largest loss among d = −2, −1, +1 and +2.
  - **It explains the loss** if that shift loses at least as much as snapping did: a median tile change of −0.027 or lower, or at least 53% of tiles losing more than 0.02.
  - **It doesn't explain the loss alone** if even that shift loses clearly less: a median above −0.015 and under 35% of tiles losing.
  - **Partly** in between.
- **Direction**, on w052: the mirrored map's median tile change, set against the snapped map's.
  - **Within 0.01 of it:** the direction carries no information for the model, and moving at all is what costs.
  - **Lower by more than 0.01:** the snapper's moves point the right way on average, and the cost comes from something else.
  - **Higher by more than 0.01:** the snapper's moves point the wrong way for this model.
- **Direction**, on PHerc. 0841: the mirrored map minus the snapped map, with its 95% interval. An interval that excludes 0 counts as a difference.
- **The 0841 and w035 sweeps** show whether the model's agreement with real ink labels peaks at the traced depth (d = 0) or elsewhere. They are reported as they come out.
