# Pre-registered test: does a smooth snap change agreement with ink labels?

Written and pushed before any surface in this test was made, rendered or run through the ink model. Changes after this point will be listed as amendments at the end, with the reason. It doesn't change any earlier outcome.

## Why

- **The dimming comes from fine-scale moves.** Diagnostic 2 ([`move_scale.md`](move_scale.md)) found that the snapper's moves finer than about 30–40 voxels dim this ink model's map on w052.
- **A smoothed move is the candidate.** With the moves smoothed at σ = 1 vertex (20 voxels), the map keeps its level (mean p −0.0014 against −0.0145 for the snap, seed 42). The mesh keeps 43% of the snap's gain at the face.
- **What it hasn't had.** It has never been scored against ink labels with a fixed rule, or at a second seed. The November review ([`review_2026_11.md`](review_2026_11.md)) asks for a labelled test at both seeds, and a blind review, before a smooth snap is offered.

**What is already known**, disclosed here:
- **Exploratory runs exist.** The review's exploratory counterfactual job ran depth fields low-passed at 8 to 128 pixels on the three targets below, at seed 42 only.
  - It reports that no smooth arm raised AUC.
  - Whether any of those surfaces is close to this plan's wasn't checked, and their numbers weren't read in writing this plan.
- **Seed 43** hasn't been run with any smooth surface on PHerc. 0841.
- **The likely outcome is no detectable difference.** The 0841 test read the dense snap of these targets as 0 ± 0.05 AUC ([`ink_test_0841.md`](ink_test_0841.md)).

## Data

- **PHerc. 0841**, as in the 0841 test ([`ink_test_0841_plan.md`](ink_test_0841_plan.md)):
  - the scan at 9.366 µm, level 0;
  - the three labelled segments with their meshes (20 voxels between vertices);
  - their labels carried to 9.366 µm (each target's `labels.npz`), over each target's `roi.json`.
- **The three targets:** `20260220213127-w00`, `…auto_grown_20260220144552896` and `…auto_grown_20260220174252405`.
- **Windings (part B):** KLAVIS's spiral-fit windings, in the 0841 test's five winding–target pairs.

## Surfaces

**A. The targets** (primary). Per target:
1. **traced:** the target as traced (depth 0);
2. **drop-in:** the target's drop-in snap output (`<target>_snapped.tifxyz`), from the 0841 test's runs at 627d058 with default settings, used as it is;
3. **smooth:** the drop-in output's moves along the target's vertex normals, smoothed with a Gaussian of 1 vertex (20 voxels), normalised over valid vertices. This is Diagnostic 2's lp1. It is made with `tools/depth_sweep.py filter OUT --original <target mesh> --snapped <target>_snapped.tifxyz --low-pass 1`.

**B. KLAVIS's windings** (secondary). Per pair:
1. **as received;**
2. **drop-in:** the 0841 test's drop-in snap of the winding;
3. **smooth:** that snap's moves smoothed with a Gaussian of 1 vertex (about 23.5 voxels), the same way.

Nothing is snapped again.

## Pipeline

All of it uses the tools at the commit that adds this plan, from a pinned copy.

- **Depth.** Each surface's depth is carried onto its target with `tools/depth_transfer.py --refine 3 --normal-sigma 10`, the corrected transfer, with the 0841 plan's other settings:
  - within 10 voxels;
  - a hit within 2 voxels;
  - `--min-dot 0.8`.

  The traced target is its own surface.
- **Render.** `tools/render_depth.py`: 21 layers 1 voxel apart, along the target's normals, `--normals flipped --center 1`.
- **Ink model.** `ink_9um` `hybrid_3d2d`, seed 42 and seed 43 (`step-075000.pth`), through villa `deb1b69` on the CPU. The options are `--no-compile --overlap 0.5 --blend-mode hann --batch-size 4 --loader-threads 4`, in the forward direction, which the 0841 plan selected on all three targets.
- **Fresh maps.** Every map is made fresh by one script, so all arms share one pipeline. The rerun after the 0841 test already made some of these maps with the same transfer and settings: traced, and the windings as received and drop-in. Those maps are compared with the rerun's, and any difference is reported. The fresh maps are the ones used.

## Scoring

- **Per target (A) or pair (B), and per seed.** `tools/fine_labels.py score --save-draws` takes all three maps and their three renders, so they share one pixel set: supervised, valid in every render, and non-zero in every map.
  - It is run with the traced map (A) or the winding as received (B) as the reference, and again with the drop-in map as the reference.
  - The measure is the AUC difference, with the plan's paired bootstrap over 128 × 128-pixel blocks (1000 resamples, seed 0).
- **Pooled** over the targets (or pairs): the mean weighted by scored pixels. Its interval comes from the targets' bootstrap draws, combined draw by draw with the same weights.
- **Widened intervals.** On these targets, a rolled-label null ran about 1.8 times the bootstrap's standard error. So every interval that a reading uses is the bootstrap's 95% interval widened 1.8 times about its estimate. Both are reported.
- **The readings** come from `tools/smooth_snap_readings.py`, with groups A (traced reference), A against drop-in, and B, deciding on A's smooth map.

## Readings, fixed now

**At one seed**, on a widened interval:
- a gain if it lies above 0;
- a loss if it lies below 0;
- no detectable difference otherwise.

**Over the two seeds:**
- **gain at both seeds**, or **loss at both seeds**: both seeds agree, and both estimates are larger than the difference between them;
- **no detectable difference at both seeds**: neither seed shows one;
- **seeds disagree**: anything else.

**The decision: pooled smooth − traced (A).**
- Its reading, as above.
- With "no detectable difference at both seeds", the write-up also gives the largest pooled loss that isn't excluded: minus the lower of the two widened lower ends. It says "no loss larger than 0.05" only if both lower ends are above −0.05.
- **What follows, fixed now.**
  - **The smooth snap goes on to a blind review of its maps** if the reading is "gain at both seeds", or "no detectable difference at both seeds" with no loss larger than 0.05. That review would be pre-registered separately. Until it has been done, the smooth snap isn't offered for ink work.
  - **Otherwise it isn't offered for ink work**, and the write-up says what the labels allow.

**Reported alongside, not deciding:**
- **A:** drop-in − traced and smooth − drop-in, pooled, and every target's reading over the two seeds.
- **B:** drop-in − as received and smooth − as received, for each pair and pooled. Pairs on the same target share its labels, so B's pooled interval is only indicative. B can't make the smooth snap safe for windings: five pairs on three targets can't say that ([`review_2026_11.md`](review_2026_11.md)). A loss of the smooth winding at both seeds would be reported as a reason not to use it on windings.
- **Map level, per target and seed.** `tools/depth_sweep.py compare` against the traced map, over pixels valid in all three renders. It is read by Diagnostic 2's rule on the median tile change in the 99th percentile:
  - **keeps the level** at −0.005 or higher;
  - **loses it** at −0.015 or lower;
  - **partly** in between.

  This is map level, not legibility.
- **Placement at the face.** `tools/face_step_check.py` runs on each target's three renders, at 1 voxel, with the face offset and normal sign from that target's `snap_segment.json`.
  - It gives the share of pixels with gap outside and papyrus inside, and the smooth snap's share of the drop-in's gain.
  - If the tool can't read these renders, this measure is dropped and that is reported.
- **The moves.** The standard deviation and neighbour correlation of each surface's moves, as `filter` prints them.

## Precision

At part A's precision in the 0841 test, the widened pooled interval is about ±0.05 at a single seed. So "no loss larger than 0.05" can come out only if both pooled estimates are about 0 or higher. A small loss would read "no detectable difference at both seeds" with a loss of up to 0.05–0.1 not excluded.

## Running it

- **Size.** 24 surfaces are carried and rendered: 9 on the targets and 15 for the pairs. There are 48 ink maps of about 30 seconds each on Lysander's CPU (16 threads). The whole run should take under an hour.
- **When.** It runs only after the pre-registered orientation test ([`orient_test_plan.md`](orient_test_plan.md)) has reported, and with the owner's go-ahead.

## Reporting

All results go into `results/smooth_snap.md`, the README and the write-up, whichever way they come out. Credit for the windings goes to KLAVIS.

## Amendments

(None yet.)
