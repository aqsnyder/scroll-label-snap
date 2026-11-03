# Pre-registered test: does snapping a mesh help or hurt ink detection?

Written and pushed before any surface volume was rendered and before any ink model was run on a snapped mesh. Everything below is fixed in advance: the data, the model, the inputs, the scoring and what counts as harm. Changes after this point will be listed as amendments at the end, with the reason.

## Why

Everything measured so far is geometric: where the mesh sits relative to the papyrus face, consistency under drift, the fibre pattern under the surface. The organisers' goal is reading text. The PHerc. 306B winners found that snapping their surface to CT density peaks "gave no improvement" in legibility. Snapping changes which layers of the rendered surface volume hold the recto face, and ink detection depends on that. So the effect could go either way, and it has to be measured.

## Data

- **Primary segment: w052** of PHerc. 0139 (`20260227000000-w052_2026022705`, 9.362 µm mesh on volume `20250728140407`). It is our held-out mesh segment, and it is not among the training segments of the ink model below, which are w029, w030, w035, w039, w040, w041, w043, w044 and w045 (from the ink_9um dataset README).
- **Secondary segment: w035** (`20260317000000-w035_2026031718`). The ink model was trained on it, with labels aligned to a render of the original mesh. That biases this test toward the original mesh. Its value is that ink labels exist, so accuracy can be scored directly.

## Meshes

- **Original:** the segment's tifxyz from the open-data bucket (`mesh/...-9.362um.tifxyz`).
- **Snapped:** `tools/snap_segment.py` with default settings, the `<name>_snapped.tifxyz` output, which keeps the original's vertex grid.
  - w035: the run at commit 389023c.
  - w052: a run at the commit that adds this plan.

## Inputs to the ink model

- **Renderer:** `tools/render_surface_volume.py` at the commit that adds this plan, with default settings. It renders 21 layers, 1 voxel apart, centred on the mesh, from scan level 0 at one pixel per voxel. The original and snapped meshes render onto the same pixel grid.
- **Renderer check, before any comparison:**
  - Render the original w035 mesh, and compare it with `tools/compare_surface_volumes.py` against the official 9.362 µm surface volume of w035 (`surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr`). That volume is the one the model was trained on.
  - Pass: a mean correlation of at least 0.9 over the matched layers, at the best alignment.
  - If the best alignment has the layers reversed or offset, re-render with `--normals flipped` or `--center`, so our renders follow the official convention, and check again. The same settings are then used for both meshes.
  - If it still fails, the test stops and that is reported.

## Model and inference

- **Model:** `ink_9um` (Hugging Face `scrollprize/ink_9um`, MIT). The primary checkpoint is `hybrid_3d2d-seed42/step-075000.pth`. The secondary, `hybrid_3d2d-seed43/step-075000.pth`, runs only if time allows.
- **Inference:** the official flat inference, `vesuvius.ink_detection.inference.infer` from ScrollPrize/villa; the commit used is recorded in the run log. Options: `--device cpu --no-compile --overlap 0.5 --blend-mode hann --batch-size 4`.
- **Layer direction:** on the timing crop below, the original mesh's surface volume is run in both directions. The direction with the higher 99th-percentile ink probability is used for every run on that segment, for both meshes.
- **Timing crop and scope:** the 1024 × 1024-pixel window at the centre of the w052 render's valid area. From its timing, project the primary run (w052, two meshes, one direction, seed 42):
  - at most 8 hours: the whole segment;
  - otherwise: the middle half of the grid's rows (vertex rows 96–288 of 385), all columns;
  - if even that projects over 8 hours: stop, and decide on a GPU as an amendment before running.

## Primary outcome: blind legibility review (w052)

- **Items.**
  - 60 comparisons of 512 × 512-pixel crops of the two ink maps, original and snapped, at the same place.
  - Crop centres are drawn with seed 2026 from pixels where the higher of the two maps is at least 0.5. If there are fewer than 60 such places, they are drawn from the top 1% of pixels instead. Centres are at least 512 pixels apart, and both maps must be valid over at least 90% of the crop.
  - 10 controls: random valid crop centres, with no ink condition.
  - 8 catch items: one ink map against itself blurred (Gaussian, σ = 4 pixels). The sharp one is the right answer.
- **Display.** The two maps side by side as grey levels (probability 0 to 1), with left and right randomised and balanced, and items shuffled. Nothing else is shown, so neither the mesh nor the CT gives the answer away.
- **Question.** "Which shows the ink more clearly: strokes or letter shapes that are easier to make out?" The answers are A, B, About the same, and Can't tell.
- **Reviewers.** The repository owner, and at least one person who did not work on the tool, each independently. Anyone who gets fewer than 6 of the 8 catch items right is reported, but their answers still count.
- **Decision.** Use the decisive answers (A or B) on the 60 comparisons, pooled over reviewers, with a two-sided exact sign test.
  - **Snapping helps** if p < 0.05 and most decisive answers prefer the snapped map.
  - **Snapping hurts** if p < 0.05 and most prefer the original.
  - **No detectable difference** otherwise.
  - Per-reviewer results are reported too.

## Secondary outcome: agreement with ink labels (w035, biased toward the original)

- Within the supervision mask of the w035 ink labels (ink_9um dataset), compare each ink map with the labels:
  - ROC AUC;
  - the best F1 over thresholds.
- Both maps use seed 42 and the direction chosen as above, mapped onto the labels' grid with the in-plane shift from the renderer check.
- Report the difference. A snapped map that matches the original despite the bias is evidence of no harm; a lower one is not conclusive.
- If the labels can't be downloaded, this outcome is dropped and that is reported.

## Descriptive: where does it change?

On both segments:
- in 256 × 256-pixel tiles, the change in mean ink probability, and in its 99th percentile, from original to snapped;
- set against the tile's mean |snap shift| and its share of flagged (reversed-face or search-limit) samples, with Spearman correlations;
- the distribution of changes, and crops of the 6 tiles with the largest drop and the 6 with the largest gain.

## Sanity check

On w035, the ink map from the official surface volume against the one from our render of the original mesh: pixel correlation, after the in-plane shift. It should be high, since both see the same mesh.

## Reporting

All outcomes go into `results/ink_test.md`, the README and the write-up, whichever way they come out. If snapping hurts, that is reported in the summary, and the tool's documentation will say where and how much.

## Amendments

### 1. Details fixed before running (3 Nov 2026, before any surface volume was rendered or the ink model was run)

None of these changes the outcomes or what counts as harm. They settle points the plan left open, and correct one fact.

- **Training segments (a correction).** The list under Data came from the dataset README. The model's training config (`vesuvius/src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json` in villa) and its model card list, for PHerc. 0139:
  - w016, w017, w029, w035, w039, w040, w041, w043 and w044 at 2.4 µm (`pherc0139-w028` there is physical segment w044);
  - plus w035, w039, w040, w041 and w044 at 9.362 µm.
  - w052 is in neither list. The run script also reads the list from each checkpoint's embedded config and stops if w052 appears.
- **Software.**
  - villa at commit `deb1b6953d6725966c57bfe40feddbcd14644858` (2 Nov 2026), using only `vesuvius/src`, on the Python path.
  - Python 3.14, with the package versions from villa's own lock file (`tools/requirements-ink.txt`); torch 2.12.1, CPU build.
  - `tools/ink_selfcheck.py` checks the setup first.
  - The renderer, the snapper and the comparison tool run from a git worktree at c5ffa62, the commit that added this plan, so later commits can't change them.
  - Each ink run is logged by `tools/ink_infer.py`: the command, the villa commit, versions and time.
- **One more inference option:** `--loader-threads 4`. It reads patches with threads instead of worker processes, and changes nothing the model computes beyond rounding.
- **Renderer check, made exact** (`tools/ink_plan.py render-check`):
  - The check renders the whole original w035 mesh with default settings, and compares the 2048 × 2048 window at the centre (the tool's default).
  - Layers are matched by the centre of the 17 the model reads: villa's centre crop takes the 17 around layer ⌊L/2⌋ of an L-layer volume.
  - Reversed order means re-rendering with the other `--normals`. If the official volume's centre layer sits more than 0.5 voxel from ours, re-render with `--center` at its offset.
  - Pass: a mean correlation of at least 0.9 at the best alignment, with the same order and centres within 0.5 voxel. A re-render gets one more check. If that fails, the test stops.
- **Timing crop and projection** (`tools/ink_plan.py crop` and `choose`):
  - The crop is centred on the median row and median column of the render's valid pixels (any layer non-zero). The 1024 × 1024 window is moved inside the image if needed.
  - The projection takes the time per patch from the two crop runs (both directions, averaged). Start-up time is included, so it overestimates. That is multiplied by the number of patches the inference runs on the two w052 renders, counted with villa's scheduling rules.
- **Layer direction for w035.** The same rule as for w052, on w035's own crop (same centre rule), with the original mesh and seed 42. The official w035 surface volume, in the sanity check, uses the direction chosen for w035, since our renders follow its layer order after the renderer check.
- **Valid pixels**, for all scoring and crop rules: pixels where both renders have at least one non-zero layer.
- **Display in the review.** The model was trained with label smoothing 0.5, so its probabilities stay between about 0.25 and 0.75 (model card). The review shows both maps with the model card's display scaling, (p − 0.25) / 0.5, clipped to 0–1, identical for both. The 0.5 threshold for crop centres is unchanged; it is 0.5 on both scales.
- **seed 43.** It runs after all seed-42 outputs exist, with the same inputs and directions, on both segments. It is reported as a robustness check (AUC on w035, the tile changes on w052), and isn't part of the primary decision.
- **Sanity check input.** The official inference reads the official w035 surface volume straight from the open-data server, as in the model card's example.
- **Labels.** These come from the ink_9um dataset bucket on Hugging Face (`hf://buckets/scrollprize/datasets/ink_9um/labels/native9-scrollprizeorg-21slices/w035`).
