# Pre-registered test: does a released finesnap surface change what the ink model sees?

Written and pushed before any surface in this test was rendered or run through the ink model. Its arms were set after a look at the surfaces' geometry (the last section), which reads no CT and no ink. Changes after this point will be listed as amendments at the end, with the reason.

## Why

Chase Hendrick's release of snapped labels ([`SharpMeow/vesuvius-snapped-labels`](https://huggingface.co/datasets/SharpMeow/vesuvius-snapped-labels), CC BY-NC 4.0, version 1.0.1, dataset commit `2a302471650dba71c4df68c3d699eba5950a3f28`) is the best-placed set of snapped surfaces we know of:
- **Face placement.** On PHerc. 0139, 91.2% of its finesnap vertices sit within 1 voxel of the recto face, against 23.7% for the published labels.
- **Training gain.** A surface model trained on its labels placed the surface 0.61 voxels closer (0.39 to 0.84).
- **No ink claim.** Its card says the data "say nothing about ink or text".

Our own tests found:
- **On w052,** ink maps from our snapped mesh read worse in a pre-registered blind review ([`ink_review.md`](ink_review.md)).
- **The cause** was the snapper's moves finer than about 30–40 voxels, not where the mesh sits on average ([`move_scale.md`](move_scale.md)).

finesnap's moves are larger than ours, rougher, and carry a net shift of the whole surface (see the geometry check below). The ink model was trained on renders of published meshes. So two things could change what it sees:
- the fine-scale moves, as with our snapper;
- the net shift away from the depth it was trained at.

The release includes w052 and two of the three PHerc. 0841 segments with ink labels, so the tests we already have apply directly.

## Data

- **The finesnap surfaces.** `snapped.tifxyz`, used as released, of three segments:
  - `PHerc0841/20260220213127-w00`;
  - `PHerc0841/20260220214732-auto_grown_20260220144552896`;
  - `PHerc0139/20260227000000-w052_2026022705`.
- **The download.** `run_fetch_finesnap.ps1` fetched them on 4 November 2026, checked all 32 files against the release's `SHA256SUMS` (no mismatch), and recorded the commit. Every vertex that has a position is used, whatever its `abstain` code. Vertices with no position are holes.
- **The published meshes** they were snapped from are the ones the earlier tests used. Their files match the `source_sha256` in each segment's `release.json` byte for byte.
- **Everything else** is as in the earlier tests:
  - the PHerc. 0841 scan and ink labels, carried to 9.366 µm exactly as in [`ink_test_0841_plan.md`](ink_test_0841_plan.md);
  - the w052 renders and ink maps of the ink test ([`ink_test_plan.md`](ink_test_plan.md)).
- **Disclosed:** on the two PHerc. 0841 segments, finesnap read the 2.403 µm scan the ink labels were drawn on. That gives it the finer scan's face, not the labels, but it is a closer look at the same place than a 9.4 µm snapper has.

## Surfaces

For each segment, five surfaces. Each finesnap-derived surface moves the published vertices along the published mesh's vertex normals, with `tools/depth_sweep.py filter OUT --original <published> --snapped <finesnap> ...`.

| Name | Moves | Options |
|---|---|---|
| published | none | |
| finesnap | as released | (the released file itself) |
| finesnap smoothed | the large-scale part: a Gaussian of 1 vertex (20 voxels), normalised over valid vertices, as in [`smooth_snap_plan.md`](smooth_snap_plan.md) | `--low-pass 1` |
| finesnap centred | as released, less their mean over the segment, so the surface keeps the published mesh's average depth | `--minus-mean` |
| finesnap smoothed and centred | the large-scale part, less its mean | `--low-pass 1 --minus-mean` |

So smoothing and centring each come with and without the other.

If a derived surface can't be made because the grids differ, it is left out and that is reported. In A, it is used only if it can be made on both targets, so that every score shares its maps.

## A. PHerc. 0841 ink labels (primary)

**Pipeline.** The same as `smooth_snap_plan.md`, at the commit that adds this plan:
- the corrected depth transfer (`--refine 3 --normal-sigma 10`), with `--max-depth 20` instead of 10, so that finesnap's longest moves are rendered rather than left as holes;
- `tools/render_depth.py` with `--normals flipped --center 1`;
- `ink_9um` at seeds 42 and 43;
- the forward direction, which the 0841 plan selected for both targets.

**Scoring.** These are the methods of `smooth_snap_plan.md`:
- `tools/fine_labels.py score --save-draws`, with all five maps on one pixel set per target;
- pooled over the two targets, weighted by scored pixels;
- bootstrap intervals widened 1.8 times;
- the readings from `tools/smooth_snap_readings.py`.

Two groups are scored. One uses the published map as the reference, the other the finesnap map.

**The decision:** pooled finesnap − published.

**Also read, the same way:**
- every other surface − published;
- finesnap smoothed − finesnap (what the fine-scale moves do);
- finesnap centred − finesnap (what the net shift does).

**What follows, fixed now.**
- **Gain at both seeds:** reported as evidence that the released surfaces read ink on this unseen scroll better than the published ones.
- **Loss at both seeds:** reported as a caution against using them for ink as they are. The smoothed and centred rows then say which part of the moves the loss goes with:
  - **a gain or no detectable difference for finesnap smoothed − finesnap, and no loss for finesnap smoothed:** the fine-scale moves;
  - **the same for finesnap centred:** the net shift from the depth the model was trained at;
  - **anything else** is reported as it comes out.
- **No detectable difference at both seeds:** reported with the largest pooled loss that isn't excluded.
- **Seeds disagree:** reported as such.

**Precision.** With two targets, the widened pooled interval is about ±0.06 at one seed, wider than in the three-target tests.

**Reported alongside:**
- per target and surface, the share of supervised pixels whose depth is more than 10 voxels (about one sheet here) and the share with no depth;
- for comparison, our snapper's drop-in and smoothed rows on the same two targets, from `smooth_snap_plan.md`. They are on that plan's pixel sets.

## B. w052 map level (secondary)

**Pipeline.**
- The four finesnap-derived surfaces are rendered with the ink test's renderer (c5ffa62) and settings, and run through `ink_9um` at seeds 42 and 43.
- The references are the ink test's maps of the original and of our snapped mesh at both seeds.

**Measure.** `tools/depth_sweep.py compare` at each seed:
- the reference is the original map; the maps are the four finesnap-derived ones and our snapped map;
- it runs over pixels valid in all six renders, with `--margin 64`.

The release holds back 25,933 of w052's 167,307 vertices for its own evaluation, and zeros raise the model's output out to about 32 pixels from a hole. So pixels within 64 of a hole or of the mesh edge are left out.

**Reading.** Diagnostic 2's rule ([`move_scale_plan.md`](move_scale_plan.md)), at each seed, on the median tile change in the 99th percentile:
- **keeps the map level** at −0.005 or higher;
- **loses it** at −0.015 or lower;
- **partly** in between.

A category stands for a surface only if it comes out at both seeds; otherwise the seeds disagree. This is map level, not legibility.

**What follows.** If finesnap loses the map level, a blind review of its w052 maps would say whether that matters. That review would be pre-registered separately and isn't part of this plan.

## Running it

- **Size, PHerc. 0841:** 10 surfaces carried and rendered, and 20 ink maps of about 30 seconds each.
- **Size, w052:** 4 renders of about a minute, and 8 ink maps of about 8 minutes each.
- **When:** only after the pre-registered orientation test ([`orient_test_plan.md`](orient_test_plan.md)) has reported, and with the owner's go-ahead.

## Reporting

All results go into `results/finesnap_ink.md`, the README and the write-up, whichever way they come out. Credit for the surfaces and their release goes to Chase Hendrick, and he sees the numbers before they are posted.

## Geometry check (before any render)

This was measured from the downloaded files and the published meshes, with `tools/surface_moves.py`, before this plan was pushed. The outputs are in [`finesnap_ink/`](finesnap_ink/) (`moves_*.json`). It reads no CT and no ink. "Moves" are finesnap's moves along the published mesh's vertex normals, in voxels.

| | w00 (PHerc. 0841) | …144552896 (PHerc. 0841) | w052 (PHerc. 0139) |
|---|---|---|---|
| Grid, as published | 211 × 238, same | 232 × 236, same | 385 × 488, same |
| Vertices with a position | 36,829 of 36,829 | 37,027 of 37,027 | 141,374 of 167,307 |
| Mean move (net shift) | −3.48 | −3.46 | −2.81 |
| Standard deviation | 3.98 | 3.98 | 3.67 |
| Median, 90th and 99th percentile of the size | 3.34, 9.03, 14.1 | 3.35, 9.02, 14.1 | 2.48, 8.03, 13.6 |
| Share moved more than 10 | 7.3% | 7.2% | 5.5% |
| Neighbour correlation | 0.26 | 0.26 | 0.36 |
| Smoothed at 1 vertex: sd, neighbour r | 1.86, 0.87 | 1.88, 0.87 | 1.98, 0.90 |
| `abstain`: snapped / filled / left in place | 29,347 / 7,439 / 43 | 29,235 / 7,667 / 125 | 121,556 / 19,288 / 530 |

- **Pure normal moves.** Tangential motion is under 0.002 voxels on average.
- **Rougher than ours.** For comparison, our snapper's drop-in moves have a neighbour correlation of 0.43 on …144552896 and 0.47 on w052, with net shifts of +0.35 and +0.05.
- **Mostly unrelated to ours.** Finesnap's moves correlate with ours at 0.05 on …144552896 and 0.26 on w052 (0.08 and 0.36 after smoothing both).
- **w052's held-back vertices** form five blocks. The largest has 24,488 vertices.

## Amendments

(None yet.)
