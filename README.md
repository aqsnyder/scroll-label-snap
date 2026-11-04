# scroll-label-snap

Moves approximate recto-surface labels in Herculaneum scroll CT back onto the real papyrus surface.

This is work in progress toward a [Vesuvius Challenge](https://scrollprize.org/prizes) progress-prize submission. It targets the "label quality" open problem in the [2026 open problems](https://scrollprize.org/2026_open_problems). Surface models are trained on hand-traced labels, and those labels drift off the surface. The organizers name this one of the main unwrapping bottlenecks and suggest "label snapping" as a fix.

## Status, 4 November 2026: the w052 ink test went against snapping, and on PHerc. 0841 no gain repeats

Three tests of the mesh snapper came in this week. Two were pre-registered.

- **Ink legibility on a held-out segment** (pre-registered).
  - Two blind reviewers compared ink maps from the original and the snapped w052 mesh. They preferred the original 75 times and the snapped mesh 31 times (p = 2 × 10⁻⁵).
  - The loss is spread over the whole segment, not concentrated where the mesh moved most.
  - Details: [`results/ink_test.md`](results/ink_test.md).
- **Ink labels on a scroll the model hasn't seen** (pre-registered, suggested by KLAVIS).
  - On three traced PHerc. 0841 segments, snapping made no detectable difference to agreement with the ink labels.
  - On KLAVIS's spiral-fit windings, as run, the drop-in output was about even and the dense output lost up to 0.22 AUC. Part of that loss came from a bug in our depth transfer. With it fixed (post hoc, exploratory), the dense output's pooled loss goes from −0.078 to −0.051 at the plan's seed, and from −0.098 to −0.002 at a second model seed. Neither output shows a loss or a gain that repeats at both seeds, though the intervals still allow a pooled loss of up to about 0.09 AUC.
  - Details: [`results/ink_test_0841.md`](results/ink_test_0841.md).
- **Face placement against an independent reference** (by Chase).
  - The test used fresh PHerc. 0139 sites: 173 vertices of model-output surfaces at 28 sites. Chase compared them with the recto face found in two higher-resolution scans.
  - At default settings, 18% of vertices ended within 1 voxel of that face, against 17% before snapping. By design, the snapper keeps each mesh's own offset from the face.
  - With `--target-offset 0` and the learned profile off, the share was 43%. His tool, snap_labels, reached 80%.

A follow-up diagnostic ([`results/depth_sweep.md`](results/depth_sweep.md)) looked for the cause:
- **Not the surface's average depth.** Moving a whole mesh 1–2 voxels barely changes the ink map's level, and agreement with ink labels doesn't change detectably.
- **The snapper's varying moves.** Snapping lowered w052's map level, and so did a mirrored mesh moved as far the other way, a little less. On PHerc. 0841, the mirrored depth cost 0.07 AUC while the snapped one made no detectable difference (read as 0 ± 0.05). The mirrored depth hasn't been rerun without the depth transfer bug's false holes, about 0.5% of hits.
- **Their vertex-scale bumps, specifically** ([`results/move_scale.md`](results/move_scale.md)). The drop-in output's moves vary from one vertex to the next. With the moves smoothed over about 20 voxels, w052 keeps its map level, and the mesh keeps 43% of the snapper's gain at the face. With only the bumps, it loses as much as the full snap.

So, for now:
- **Don't use snapped meshes for ink detection.**
- **Read the drift tests below as consistency** (does the snapper put a drifted label back where it was?), not as accuracy against the true face.
- **The label-cube results** and their blind reviews below still stand for what they measured. None of these three tests looked at the cubes. The [November review](results/review_2026_11.md) did: snapped labels gain tunnels and merge sheets, and the drift test can't rank the snapper against a plain per-sample search.

**Pre-registered next, not run yet.**
- **A blind review at matched level** ([`results/ink_review_matched_plan.md`](results/ink_review_matched_plan.md)) asks whether snapped w052 maps are less legible or only dimmer.
- **A labelled test of a smooth snap** on PHerc. 0841, at both seeds ([`results/smooth_snap_plan.md`](results/smooth_snap_plan.md)).
- **The same ink tests on Chase Hendrick's released finesnap surfaces** ([`results/finesnap_ink_plan.md`](results/finesnap_ink_plan.md)): two labelled PHerc. 0841 segments and w052, as released, smoothed, and centred on the published depth.

![Chase's real drift test: snap_labels and four scroll-label-snap settings against a two-scan recto face on PHerc. 0139](results/independent_drift_test_chase.png)

*Chart by Chase, shared in the Vesuvius Challenge Discord and used with his permission. In his test, scroll-label-snap ran on 13 × 13-vertex crops. That gives its offset estimate and learned profile fewer samples than a whole segment would.*

## How it works

For each labeled cube:

1. **Sample the surface.** The labeled surface is sampled every ~3 voxels.
   - Normals come from the shape of the label band.
   - Each normal is oriented to point from papyrus toward the gap.
   - Orientation is made consistent along each sheet, and then across neighbouring sheets. Every wrap's recto faces the scroll centre, so neighbouring sheets should point the same way.
2. **Score candidate shifts** from −4 to +4 voxels along the normal, by adding two terms:
   - **Step edge.** A narrow step finds the edge. A wide step checks that papyrus is behind it and a real gap in front, which rejects thin cracks inside a sheet.
   - **Learned profile match.** Normalized cross-correlation with the cube's own typical cross-section at a label.
3. **Choose all shifts at once** with an exact graph cut (Ishikawa construction). It minimizes:
   - the score cost
   - an L1 smoothness term between neighbouring samples on the same sheet
   - a hard limit of 1 voxel shift difference between neighbours
   - a weak pull toward the original label
   - a penalty for moving a label outward (toward the gap) to a spot with papyrus on both sides. There the normal is probably flipped, or the label sits on the far face of a sheet, and pushing it further only buries it in the sheet (new in v2).
4. **Move the label band** along the solved shift field. Each voxel's displacement is blended from nearby samples on the same sheet, and the band is resampled with interpolation, so a moved band keeps clean edges (new in v2; v1 left stair steps). Not-annotated regions (label 2) are never changed. A per-voxel confidence map is written as well.

`scroll_label_snap/core.py` also contains a port of the existing VC3D refiner (`ObjAlphaCompRefinement`) as a baseline. `tests/test_solver.py` checks the graph cut against brute force.

## Results on the label cubes

Two sets of cubes, evaluated the same way with the default parameters (`tools/eval_folder.py`):
- **Development (20 cubes):** the cubes the method was built and tuned on. Full tables in [`results/eval_dev.md`](results/eval_dev.md).
- **Held-out (100 cubes):** other cube pairs picked at random from the bucket. The seed was fixed and pushed before any of them were downloaded, and nothing was tuned on them. Full tables in [`results/eval_heldout.md`](results/eval_heldout.md).

Per-cube numbers are in the matching `.json` files. A before/after on cube sample_00005 is in [`results/labels_before_after_00005.png`](results/labels_before_after_00005.png). It shows the three largest changes and three random slices.

### The existing refiner as a baseline

`scroll_label_snap/core.py` contains a port of VC3D's alpha-composite refiner (`ObjAlphaCompRefinement`), the refinement step that exists today. With VC3D's default opacity thresholds (118 and 165 of 255), it does worse than leaving the labels alone here. In these cubes the gap is about 50 and papyrus about 125, so the lower threshold sits just below the papyrus level and the upper one well above it. To make it a fair baseline, it is also scored in three better-set-up versions:
- **Offset removed.** The refiner works to its own convention: it puts the surface just past the first voxel above the threshold, not where these annotators draw it. Each version has its median shift on the clean labels removed.
- **Thresholds from each cube's own levels**, offset removed: the mean brightness of gap and of papyrus (the two Otsu classes). This needs no tuning.
- **The best fixed thresholds**, offset removed: the best of 13 threshold pairs on the development cubes (70 and 110). On the held-out cubes we report that pair, and also the pair that is best on the held-out cubes themselves, as the refiner's ceiling.

The offset removal uses the clean labels, which a real user wouldn't have; the snapper takes its convention from the labels it is given. With these drift fields that difference is small: on the development cubes, each field's median is within 0.15 voxels of zero.

### Drift test

Each cube's labels were pushed off the surface by a known smooth random drift (std 1.5 voxels, correlation length 40 voxels). Each method was then run on the drifted labels, and we measured how far the result lands from the original label. This is scored only at **trusted points**: the 10% of samples (7% held out) where, by the snapper's own edge score, the strongest edge within reach is already at the label and no other edge comes close (258,394 points in development, 717,523 held out). A method that goes to the strongest edge returns to these points almost by construction.

| Method | Within 1 voxel: development | Within 1 voxel: held-out | Mean error (voxels): development | Mean error: held-out |
|---|---|---|---|---|
| No correction (drifted labels) | 57% | 57% | 1.12 | 1.14 |
| Existing refiner, VC3D default thresholds | 9% | 18% | 3.09 | 2.66 |
| Existing refiner, default thresholds, offset removed | 20% | 33% | 2.67 | 2.20 |
| Existing refiner, thresholds from each cube's own levels, offset removed | 80% | 78% | 0.68 | 0.73 |
| Existing refiner, best thresholds on development (70-110), offset removed | 82% | 77% | 0.64 | 0.77 |
| **Snapper** | **89%** | **88%** | **0.52** | **0.55** |

The refiner's ceiling on the held-out cubes, the pair that is best there (50-110, picked after seeing them), reaches 80% within 1 voxel (mean error 0.69).

A per-sample best score with no graph cut beats the snapper on all 20 development cubes (94% vs 89%, +4.4 points, 95% CI 3.7 to 5.2; held-out cubes not run). The order depends on how trust is defined: trust a point because the edges of its neighbours 4-12 voxels away sit at the label, without reading its own CT, and the snapper leads on all 20 (85.5% vs 82.2%). So this test doesn't show that the graph cut helps or hurts ([`results/review_2026_11.md`](results/review_2026_11.md)).

**Consistency** is measured at all points. It asks whether correcting the drifted labels lands in the same place as correcting the clean labels.

| Within 1 voxel | Development | Held-out |
|---|---|---|
| Existing refiner, default thresholds | 61% | 63% |
| Existing refiner, thresholds from each cube's own levels | 77% | 73% |
| **Snapper** | **81%** | **75%** |

**Clean labels.** This is the intensity step across each label, papyrus side minus gap side. It is circular: the snapper optimizes edge strength, and a per-sample search with no graph cut scores higher (95% vs 91% on the development cubes, higher on all 20). The drift test picks its points with the same edge score, so it isn't independent of this table.

| Papyrus on the correct side of the label | Development | Held-out |
|---|---|---|
| Original labels | 67% | 63% |
| Existing refiner, default thresholds | 35% | 45% |
| Existing refiner, thresholds from each cube's own levels, offset removed | 75% | 71% |
| **Snapper** | **91%** | **86%** |

![per-cube results, held-out cubes](results/eval_heldout.png)

The same figure for the development cubes is [`results/eval_dev.png`](results/eval_dev.png).

What this shows:
- **Most of the refiner's poor default result is its thresholds.** Set from each cube's own levels, with its offset removed, it goes from 9% to 80% on the development cubes (18% to 78% held out). That fix is simple and needs no tuning; it may be worth having in VC3D on its own.
- **The snapper is still ahead of the best-set-up refiner on accuracy.**
  - Within 1 voxel: +9 points on the development cubes and +10 held out.
  - Mean error: 0.52 vs 0.68 voxels in development, 0.55 vs 0.73 held out.
  - Over 2 voxels: 3% vs 4% in development, 3% vs 6% held out.
  - Cube by cube it does as well or better on 19 of 20 development cubes and 94 of 100 held-out cubes. The median held-out cube gains 10 points.
- **On consistency the two are close.** The snapper is within 1 voxel of its own clean-label result more often (81% vs 77% in development, 75% vs 73% held out). Held out, it also has slightly more large differences (14% vs 12% over 2 voxels), so its mean difference is larger (1.04 vs 0.92 voxels).
- **Against no correction**, the snapper is better on 97 of the 100 held-out cubes. It is worse on three: sample_00184, sample_00186 and sample_00478.
- **Where the refiner wins.** The refiner with the cube's own levels scores higher on sample_00018 in development (72% vs 67%) and on six held-out cubes. Two of those gaps are large: sample_00478 (59% vs 49%) and sample_00639 (84% vs 79%). The other four are 2 points or less.

### Drift draws, and an earlier number

The drift is random, so per-cube numbers depend on the draw. Earlier versions of this README reported 91% within 1 voxel for the snapper on the development cubes. That run seeded cube k with k // 2 (it ran as two processes, each counting from 0); the current runs seed cube k with k, as the held-out run always did. The code is the same: with the old seeds, all 20 development cubes give exactly the old drifted labels, and sample_00002 and sample_00018 give exactly the old snapper results.

Most cubes move by a few points between draws. sample_00018, a hazy cube, goes from 93% with the old draw to 67% with the new one:
- **What happens.** With the new draw, 15% of its trusted points end more than 2 voxels off (8% for the refiner with the cube's own levels). Most of them are about 3 voxels toward the papyrus side, in many small patches, and not where the drift was largest.
- **Why.** This draw reverses the sheet orientation at 32% of the cube's trusted points. The snapper gets 10% within 1 voxel at those points and 93% at the others, the same as the old draw. A per-sample search has the same problem (5% and 98%). Turning the profile term off lifts the reversed points to 44% and lowers the others to 90%, which gives 75% in all and 6.5% more than 2 voxels off.
- **Next.** Keeping sheet orientation stable under drift comes first (`Params.orient_cube`, off by default, which met its pre-registered condition in [`results/orient_test.md`](results/orient_test.md)), then making the learned profile independent of where the input labels sit. A change like that needs a fresh held-out set, since this one has now been used.

`tools/drift_draws.py` runs both checks: `seeds` rebuilds an older run's drifted labels with a given seed scheme, and `diagnose` describes one cube's large misses under one draw. The v1-to-v2 comparison below comes from the earlier runs. We haven't confirmed that v1 used the same draws.

### The whole dataset

All 892 cube pairs in the bucket have been snapped with the v2 defaults (`tools/snap_folder.py`, 12 workers on a Ryzen AI Max+ 395). That is 116 million surface samples in 15.8 CPU-hours, about 1.3 hours of wall time. Per-cube numbers are in [`results/full_run_summary.csv`](results/full_run_summary.csv):

| Per cube | Median | 5th-95th percentile |
|---|---|---|
| Surface samples | 121,680 | 49,634-234,488 |
| Share moved by 1 voxel or more | 70% | 61-77% |
| Share flagged (label on a reversed face) | 6.9% | 2.3-17.5% |
| Mean confidence | 0.80 | 0.66-0.89 |

The most-flagged cubes are also the lowest-confidence ones: samples 00798, 00256 and 00323 have 25-34% flagged. That makes them good places for a human to look first.

### Changes since v1

From the earlier development runs, [`results/eval_v1.md`](results/eval_v1.md) and [`results/eval_v2.md`](results/eval_v2.md), with the old drift seeds:

| | v1 | v2 |
|---|---|---|
| Drift test: within 1 voxel | 87% | 91% |
| Drift test: mean error (voxels) | 0.54 | 0.46 |
| Consistency: within 1 voxel | 79% | 83% |
| Extra edge roughness of the band vs. the original labels | +11% | +6% |

- **Smooth warp.** Roughness is the number of exposed voxel faces per band voxel, over all 20 cubes (`tools/roughness.py`). Moving a band by a smooth shift adds roughness by itself: the drift test's smooth drift (std 1.5, corr. 40) adds +6.1% over these cubes, so the snapper's +6.0% is at that floor. A per-sample search with no graph cut adds +18.8%. See [`results/warp_v1_vs_v2.png`](results/warp_v1_vs_v2.png).
- **Inside-the-sheet guard.** On the 120 moves in the blind review, v2 puts 3 of the 8 losses back on the original label (q057, q070, q107). It leaves 90 of the 98 wins within 0.5 voxel of where v1 put them; 2 went back to the original. See [`results/review_v1_losses_v2.png`](results/review_v1_losses_v2.png); `tools/recheck_review.py` produced the per-item numbers in [`results/review_v1_items_under_v2.json`](results/review_v1_items_under_v2.json).
  - The guard only penalizes outward moves. We chose that after seeing that penalizing inward moves as well also undid 2 wins, so these review numbers are optimistic. A fresh review is the real test.
- **Drift-test inputs.** The drifted labels and the existing refiner's output now go through the smooth warp too, so their rows moved slightly from v1.

### Caveats

- **Trusted points are the easy places.** In hazy regions the picture is less clear; the blind review below covers them.
- **A cube's outer layer is not corrected, and these tables don't score it.** Samples within 8 voxels of a cube's face stay where they are, which is 10% of the band voxels on the development cubes (7-14% per cube). The tables score only samples at least 10 voxels from a face. The outermost 3 voxels of these cubes hold no labels at all (all label 2).
- **The refiner rows come from our port** of `ObjAlphaCompRefinement`, run on the same samples, normals and drift as the snapper. The better-set-up versions are not how VC3D runs it today. They show how much of the gap to the snapper is down to setup.

### Known failure mode

The automatic evaluation flagged hazy cubes (sample 2; samples 8, 17 and 18), where the snapper moves many clean labels 2+ voxels. The blind review found those moves are mostly right: 20 of 21 decisive answers there preferred the snapped label.

The moves the review rejected look different. In 6 of the 8 losses, the original label sits on a face whose gap is on the opposite side from the oriented normal. Either the normal is flipped there, or the label is on the far face of its sheet. v1 then pushed the label outward into papyrus. The v2 guard stops 3 of these 6; the other 3 still go into the sheet.

Of the two remaining losses, one is a small move. In the other, the snapper put the band centre on the papyrus side of the edge, which is where this cube's own labels usually sit (sample 11 places them about 1.5 voxels in). The reviewer preferred the band nearer the edge.

Compare [`results/compare_sample_00002.png`](results/compare_sample_00002.png) with [`results/compare_sample_00005.png`](results/compare_sample_00005.png) for what v1 does on a hazy and a clear cube.

### Flagged labels: which face does the label belong to?

The snapper flags a label when it sits on a clear face of the opposite polarity (papyrus on the +n side, gap on the −n side) and no face of the expected polarity is within ±4 voxels. That is 3–19% of samples per cube (median 7%); sample 2 is the highest. The flags are written to `OUT_DIR/flags/` next to the labels.

The flag can't be turned into an automatic fix, because the scan alone can't say which face such a label belongs to. Two different situations look identical locally:
- **Far face:** the label is on the far face of its own sheet. It should stay on the face it's on.
- **Crossed gap:** the label has drifted across the gap onto the next sheet. It should move back.

The blind reviews show both. Of the 196 moved items across the two reviews, the flag caught:
- 7 of the 15 where the reviewer preferred the original
- 7 of the 18 marked "can't tell"
- only 11 of the 161 where they preferred the snapped label

So flagging about 7% of labels points a human at half the losses. This matches the "humans correct only the uncertain regions" idea in the open problems.

`Params(face_polarity=True)` snaps flagged labels onto the face they're on instead of searching along +n. At the reviewed points this keeps all 7 flagged losses within 1.5 voxels of the original. It also pulls back 10 of the 11 flagged wins, which the reviewers preferred moved, so it is off by default and the default output is unchanged from v2. The per-item positions are in [`results/reversed_face_check.json`](results/reversed_face_check.json).

## Snapping traced meshes (VC3D tifxyz)

The same scores and graph cut also snap a traced VC3D surface. A tifxyz grid has vertices about 20 voxels apart. It is densified to about 3-voxel samples, which are snapped and then fit back onto the original grid. The normals come from the grid. Which side is recto comes from the scroll's geometry, because every wrap's recto faces the scroll centre. Each mesh keeps its own offset from the papyrus face. Details are in [`results/mesh_w035.md`](results/mesh_w035.md).

It was developed on segment w035 of PHerc. 0139 at 9.362 µm, the one in the scrollprize.org ink tutorial, using 8 random patches of 24 × 24 vertices. It was then tested on a held-out segment, w052, a different wrap on the same scan. That test was set out before the data was downloaded, and nothing was tuned on it ([`results/mesh_w052.md`](results/mesh_w052.md)).

| | w035 (development) | w052 (held out) |
|---|---|---|
| Papyrus on the recto side of the face: original → snapped | 55% → 88% | 70% → 93% |
| Drift test, within 1 voxel: no correction → snapper | 70% → **90%** | 69% → **94%** |
| VC3D refiner, thresholds tuned on the same patches, offset removed | 61% | 68% |
| VC3D refiner, thresholds from each patch's own gap and papyrus levels, offset removed | 57% | 70% |
| Snapping onto the m7 surface prediction (the PHerc. 343 winner's method), offset removed | 41% | 63% |
| Consistency: snapper / refiner with each patch's own levels / m7 snapping | 84% / 75% / 69% | 90% / 84% / 76% |
| Fibre check, patches with the recto signature: snapped / original / m7 snapping | 8 / 5 / 7 of 8 | 8 / 6 / 6 of 8 |

**Which output these numbers score.** The snapper's position numbers in this section, and in `results/mesh_w035.md`, `results/mesh_w052.md` and `results/w035_whole.md`, score `snapped_dense.tifxyz` (the dense output), the surface as solved. The exception is the one-voxel face check under "A whole segment", which reads the ink test's renders of `snapped.tifxyz`. `snapped.tifxyz` (the drop-in output) keeps the input's 20-voxel grid, which can't hold that surface. `snapped_fine.tifxyz` (5 voxels, meta scale 0.2, written only with `--fine-grid 5`) can. Scored on the written files, read bilinearly, w035 / w052:

| | Input mesh | `snapped.tifxyz` (20 voxels) | `snapped_fine.tifxyz` (5 voxels) | `snapped_dense.tifxyz` (about 3 voxels) |
|---|---|---|---|---|
| Drift test, within 1 voxel (std 1.5, corr. 40) | 70% / 69% | 72% / 83% | 90% / 94% | 90% / 94% |
| Drift test, within 1 voxel (std 2.0, corr. 80) | 42% / 41% | 73% / 82% | 90% / 92% | 90% / 92% |
| Papyrus on the recto side of the face | 55% / 70% | 81% / 89% | 88% / 92% | 88% / 93% |
| Mean step across the face | 8.4 / 30.2 | 46.8 / 64.7 | 59.7 / 74.8 | 60.4 / 75.3 |
| The whole of w035 on disk, with channels | 0.9 MB | 1.4 MB | 23 MB | 71 MB |

`tools/eval_mesh.py --output dense|fine|grid` scores each one. This table scores position against the snapper's own face (see the note on drift tests in the status section), not reading. The pre-registered ink test (`results/ink_test_plan.md`) rendered `snapped.tifxyz`, so its result describes the 20-voxel column: reviewers preferred the snapped map less often than the original's (`results/ink_test.md`). The fine grid has had no blind review. In an exploratory look at 15 w052 sites, on central 256-pixel crops where such changes run larger than over the whole map, its ink map was dimmer than the 20-voxel file's at both model seeds (mean probability -0.048 and -0.041 against the original, where `snapped.tifxyz` gave -0.032 and -0.019). So treat it as a geometry output, not one for ink reading ([`results/review_2026_11.md`](results/review_2026_11.md)). `tools/render_surface_volume.py` renders the fine grid onto the input's pixels. The fine grid's normals are rougher, like the dense surface's: 9.6° from the original mesh's on average, against 4.7° at 20 voxels, so a layer 8 voxels off the surface wanders by about 1.3 voxels instead of 0.65. None of these files has been opened in VC3D.

The w035 results are below.

- **Where the original mesh sits.** Read at its own typical face position, the original mesh has papyrus on the recto side at only 55% of samples. After snapping, 88%.
- **Drift test on w035.** Known smooth drifts were applied to the mesh vertices. At 15,324 trusted samples, the share within 1 voxel of the original position went:

| Method | Within 1 voxel (drift std 1.5, corr. 40) | Within 1 voxel (drift std 2.0, corr. 80) |
|---|---|---|
| No correction | 70% | 42% |
| VC3D alpha-composite refiner, thresholds tuned on these patches, offset removed | 61% | 57% |
| **Snapper** | **90%** | **90%** |

![w035 before and after](results/mesh_w035_render.png)

Each patch shows the CT read through the original and the snapped mesh, just inside and just outside the recto face, plus a cross-section.

**Fibre check.** Recto fibres run around the scroll and verso fibres along its axis. So under a surface on a recto face, the fibre direction should turn from across the axis to along it, going deeper into the papyrus. `tools/fibre_check.py` measures this from the CT alone, except that its depth origin is the snapper's estimate of where the face is.
- **Snapped surface:** shows the pattern in 8 of 8 patches.
- **Original mesh:** 5 of 8, with near-zero size.
- **Control snapped onto the opposite faces:** 4 of 8.

So surfaces moved onto recto-polarity faces show the recto pattern. The effect is small but consistent. It doesn't single out the snapper: a plain gradient search with a light median scores +0.18 on w035 (snapper +0.15) and +0.21 on w052 (snapper +0.20), 8 of 8 patches each, and on w052 the original mesh moved 2 voxels toward the papyrus at every vertex scores +0.18, 8 of 8 (w035: +0.05, 6 of 8). See [`results/fibre_w035.png`](results/fibre_w035.png).

**Review flags.** 15% of samples stop at the ±4-voxel search limit (`snap_limit`), and 10.5% sit on a reversed face (`snap_flag`). On the input grid, `snap_review` marks the 17% of vertices where most of the surrounding surface is flagged.

**Open question.** At this resolution, sheets are often thin layers packed into bundles. Where two recto faces lie within ±4 voxels of a wandering trace, neither the drift test nor the fibre check can say which sheet the tracer meant. That needs expert review, or a downstream check such as ink detection on renders of both meshes.

**Ink detection.** A pre-registered test has now measured whether a snapped surface reads better, and it reads worse: blind reviewers preferred the ink maps from the original w052 mesh, 75 to 31 ([`results/ink_test.md`](results/ink_test.md)). This matches the PHerc. 306B winners, who found that snapping to CT density peaks, after snapping to m7, did not make their letters more legible.

That test rendered `snapped.tifxyz`. The dense and fine surfaces have not had a blind review.

**Options.**
- `--target-offset T`, on `snap_tifxyz.py` and `snap_segment.py`, puts the mesh T voxels from the face along papyrus → gap (negative = inside the papyrus). On one synthetic scene it lands within 0.24 voxel of T for T from -3 to +3. On real data, Chase's test above put `--target-offset 0` about a voxel inside the two-scan face (median +0.91 voxels with the learned profile off, +1.34 with it on, pinned at 347ea5b4, before the change below). This version has not been run against that reference.
- **`--target-offset` and the learned profile.** It now moves the learned profile too. `results/ink_test_0841_plan.md` uses `--target-offset 0 --param smooth_w=0.1`, at its pinned commit 627d058, and Chase's test ran `--target-offset 0` at 347ea5b4. At both, the profile stays at each tile's own position. The same flags now give other surfaces wherever a tile's own offset is not 0, and whether those sit closer to the two-scan face has not been measured. Read patch by patch, on w035 patch_3 (own offset +1.5), 58.4% of samples differ by at least 1 voxel, and on w052 patch_2 (own offset -1.0), 15.2% do. Patches with an own offset of 0 are identical. Add `--param template_follows_offset=false` to get the earlier behaviour.
- With `--tifxyz`, the recto side comes from `--axis`, `--normals`, or the mesh's own curve if it is at least 1,600 voxels across. Shorter pieces of w035 and w052 can fit a tight false circle with a confident, wrong sign. Otherwise the tool stops and asks. `--normals ct` lets the CT vote decide; that vote is wrong on 4 of the 16 patches here.

### A whole segment

`tools/snap_segment.py` snaps a whole segment tile by tile, reading the scan from the open-data bucket as it goes and caching what it downloads. All of w035 (291 × 262 vertices, 3.1 million samples) took 7 minutes with 8 workers. None of its 279 tiles failed, and the blended result has no seams at tile boundaries. Details are in [`results/w035_whole.md`](results/w035_whole.md).

- **Papyrus on the recto side, read from the renders:** 54% for the original mesh, 86% for the snapped one. The 8 patches gave 55% → 88%.
- **Review flags:** `snap_review` marks 17.7% of vertices.
- **w052, whole, for the ink test:** 711 tiles in 15 minutes, none failed. The mean |shift| was 1.69 voxels, and `snap_review` marks 13% of vertices ([`results/w052_whole_snap_segment.json`](results/w052_whole_snap_segment.json)).
- **One voxel either side of the face** (a check suggested by Sean in the Discord, added after the fact): gap outside and papyrus inside at 21% → 42% of pixels on w035, and 22% → 42% on w052. This is close to what the snapper optimises. See [`results/ink_test.md`](results/ink_test.md), which also has two voxels.
- **One offset, in the step-edge term only.** The tool uses one offset for the whole segment, the median of 12 sample tiles (+0.5 voxels for w035, range 0 to +1.5). Each tile's learned profile stays where that tile's mesh sits, so a tile whose own offset differs lands between its own depth and the segment's, not at one depth for the whole segment. `--recentre-template` moves the profile too. It changes the output, so neither this run nor the ink test's runs use it.

![w035, original and snapped, 1.5 voxels inside and outside the recto face](results/w035_whole_compare.png)

```bash
pip install "zarr>=2.16,<3" fsspec aiohttp requests        # to read the scan from the open-data bucket
python tools/fetch_segment.py segments/w035 --segment <tifxyz URL> --volume <OME-Zarr URL> --patches 8
python tools/snap_tifxyz.py snapped segments/w035/patch_*   # snapped.tifxyz (same grid) + snapped_dense.tifxyz; add --fine-grid 5 for snapped_fine.tifxyz
python tools/eval_mesh.py out/eval_mesh.json segments/w035/patch_*   # add --output grid or fine to score those files instead of the dense one
python tools/render_mesh.py out/render.png snapped segments/w035/patch_1 --sections
python tools/fibre_check.py out/fibre.json segments/w035/patch_* --snapped snapped --png out/fibre.png
python tools/snap_to_prediction.py snapped_m7 segments/w035/patch_*   # baseline: snap onto a surface prediction (fetch with --extra m7=URL)
python tools/snap_segment.py out/w035_whole --segment <tifxyz URL> --volume <OME-Zarr URL> --cache zarr_cache --workers 8 --name w035
python tools/render_segment.py out/w035_whole/w035 out/w035_whole --volume <OME-Zarr URL> --cache zarr_cache   # before/after renders
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# one or more cubes
python tools/snap_cube.py OUT_DIR images/sample_00001.tif labels/sample_00001.tif

# a whole folder laid out like the dataset (images/ + labels/), in parallel; resumable
# writes labels/, confidence/ and flags/ (labels that may be on the wrong face)
python tools/snap_folder.py IN_DIR OUT_DIR --workers 12

# evaluation, summary and before/after close-ups
python tools/eval_snap.py out/eval.json images/sample_00001.tif labels/sample_00001.tif
python tools/summarize_eval.py out/summary out/eval.json
python tools/render_compare.py out/cmp.png images/X.tif labels/X.tif OUT_DIR/labels/X.tif z
```

Each worker peaked at about 1 GB on the development cubes. A 320³ cube takes 30–120 s per core.

### Windows

```bat
git clone https://github.com/aqsnyder/scroll-label-snap
cd scroll-label-snap
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tools\snap_folder.py D:\vesuvius\kaggle D:\vesuvius\snapped --workers 12
```

## Data

The tools run on the Vesuvius Challenge surface-label cubes ([Hugging Face bucket](https://huggingface.co/buckets/scrollprize/datasets/tree/surfaces/kaggle)): `images/sample_NNNNN.tif` paired with `labels/sample_NNNNN.tif`. Each cube is uint8. Most are 320³; 51 of the 892 are 256³ and one is 384³.

| Label value | Meaning |
|---|---|
| 0 | annotated background |
| 1 | recto surface |
| 2 | not annotated |

The data is licensed CC BY-NC 4.0 and is not included here. The example images in `results/` are crops of that data. The mesh figures are renders of PHerc. 0139 segments w035 and w052 and their scan, and the ink-map figures are made from those renders. They come from the [Vesuvius Challenge open data](https://scrollprize.org/data), also CC BY-NC 4.0. The PHerc. 0841 results use that scroll's open data, under the same licence.

## Blind human review

The automatic tests can't say whether a big move on a hazy cube is a fix or a mistake, so a
person judges a sample by eye without knowing which label is which.

- `tools/make_review.py` picks, per cube, 3 points where the snapper moved the label by 2+
  voxels, 3 where it moved 1-2 voxels and 1 where it didn't move (a control), spread at least
  32 voxels apart: 140 comparisons. It adds 14 catch items, where the original label is
  compared with itself pushed 3 voxels into the gap or 2.5 voxels into the sheet, at points
  where the snapper and the original agree; a reviewer who can't spot those can't be trusted
  on the rest.
- Each item shows a 48 x 48 voxel slice, cut perpendicular to the axis closest to the surface,
  with a zoomed-out context view and the two candidates side by side. Item order is shuffled,
  and exactly half the comparisons (and half the catch items) put each candidate on the left.
- The reviewer answers "A is better", "B is better", "About the same" or "Can't tell"
  (`review/index.html`). The page shows only item ids and images; the answer key stays
  with whoever runs the review until the answers are in.
- `tools/score_review.py` unblinds the answers and reports preferences per stratum, for the
  hazy cubes, by direction of the move and per cube, with an exact sign test, catch
  accuracy, left/right bias and answer times.

```bash
python tools/make_review.py review/ out/review_key.json data/images data/labels out/labels --seed 0
python tools/score_review.py out/review_key.json answers.json results/review_v1.md
```

The first review set is `32989576ca01`. Its answer key was sealed before any answers came in:
the SHA-256 of the key (as `json.dumps(key, sort_keys=True)`) is
`32989576ca01fe3ce79d7ab8491025ed9730d72ac8529d673898dc4182edcb28`, and an encrypted copy is
`review/sealed_key_32989576ca01.enc` (AES-256-CBC, `openssl enc -pbkdf2 -iter 200000`). The
key, the passphrase and the answers are now in `results/`.

### Result of review v1 ([`results/review_v1.md`](results/review_v1.md))

Where the snapper moved the label, the reviewer preferred the snapped label 98 times and the
original 8 times. That is 92% of decisive answers (95% CI 86-96%); 14 of 120 were "about the
same" or "can't tell". Big moves (2+ voxels) went 47 to 6 and moderate moves (1-2 voxels) 51
to 2. The hazy cubes flagged by the automatic evaluation went 20 to 1, so the large moves there
are mostly fixes, not mistakes.

Five of the eight losses put the label inside a thick sheet instead of on its face. Caveats:
- One reviewer, who owns this repo.
- The snapped band's stair-step edge can give it away.
- The controls and catch items were only clean at the centre point of each 48-voxel window.

### Review v2 (set `af7d4c85d7f6`)

Built with `tools/make_review_v2.py --seed 1` on the v2 output, for a reviewer who did not work on the tool:
- **Fresh points.** No item is within 32 voxels of a v1 item.
- **111 items:**
  - 40 big moves and 40 moderate moves (2 of each per cube).
  - 15 controls, where the two labels agree across the whole window (IoU 0.80 or more), not just at its centre.
  - 16 catch items: the original label pushed 3 voxels into the gap or 2.5 voxels into the sheet, 8 each way. They are only used where both labels agree across the window and the scan confirms the push direction: a dark gap in front, or bright papyrus behind.
- **A short tutorial first:** worked examples (good, too far out, too far in) and 4 practice items that show the right answer after each pick.
- **Per-reviewer storage.** Answers are stored under each reviewer's name, so several people can take the same set.
- **Sealed key.** The SHA-256 of the answer key is `af7d4c85d7f6f07106839d0af2de42655095ba677316cb9c3ae9ddec95dc08f6`, and an encrypted copy is `review/sealed_key_af7d4c85d7f6.enc`. The page is `review/v2/index.html`.

### Result of review v2 ([`results/review_v2.md`](results/review_v2.md))

A second reviewer, who did not work on the tool and had never seen these scans, judged the v2 output on fresh items:
- **Moves:** where the snapper moved the label, she preferred the snapped label 63 times and the original 7 times. That is 90% of decisive answers (95% CI 81-95%); 10 of 80 were "about the same" or "can't tell".
- **Big moves (2+ voxels)** went 27 to 6; **moderate moves (1-2 voxels)** 36 to 1; **the hazy cubes** 12 to 1.
- **Checks:** she got all 16 catch items and all 4 practice items right. She rated 11 of the 15 controls "about the same".
- **Losses:** four of the seven are the same failure mode found in review v1, where the label sits on a face whose gap is opposite the normal.

## Next steps

1. **Decide flagged labels with more context.** Following each sheet over a larger area, or using the traced mesh's own history, could tell which face a flagged label belongs to. Until then, flagged labels are the place for human review.
2. **Use the confidence map:** leave or flag low-confidence regions instead of moving them.
3. **Keep sheet orientation stable under drift first** (`Params.orient_cube`, off by default. Its pre-registered test on 100 cubes met the plan's condition: [`results/orient_test.md`](results/orient_test.md). Adopting it means re-snapping all 892 cubes, which is undecided). **Then make the learned profile independent of the input labels** (see "Drift draws" above; in Chase's test, turning it off helped), and test that on a fresh held-out set.
4. **Meshes: a smooth snap (not started).** The drop in map level is in the moves' vertex-scale bumps ([`results/move_scale.md`](results/move_scale.md)). Moves smoothed over about one vertex spacing keep the map level; `tools/depth_sweep.py filter --low-pass 1` makes such a mesh after the fact. Whether it reads better needs a pre-registered blind review.
5. **Meshes: face placement.** Run the current `--target-offset 0` against the two-scan face; at 347ea5b4 it sat about a voxel inside in Chase's test. Compare with snap_labels in the blind ink test.
6. **Ink: a blind review with matched displays.** It is the only test here that can separate a dimmer map from a less legible one ([`results/review_2026_11.md`](results/review_2026_11.md)). Pre-register it first.
