# Follow-up test, PHerc. 0841: does snapping change agreement with ink labels?

Written and pushed before any surface in this test was snapped, rendered or run through the ink model. It is a follow-up to the pre-registered w052 test (`results/ink_test_plan.md`) and doesn't change it. It came out of a suggestion and data from KLAVIS in the Vesuvius Challenge Discord.

## Why

PHerc. 0841 has ink labels in the open data, and it is not among the ink model's training scrolls. The checkpoints' configs list PHerc. 0139, 0814, 1667 and Paris 4.

KLAVIS tried scroll-label-snap on four spiral-fit windings that cross labelled 0841 areas. With the default settings and the drop-in output, he found AUCs about even: changes of −0.011, −0.005, −0.001 and +0.022. His scoring reads a winding's depth only about every 20 voxels, so it can't see the detail the snapper adds between vertices. This test carries depth over at every pixel, about one per voxel.

## Data

- **Scan:** PHerc. 0841, 9.366 µm (`PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr`), level 0.
- **Labelled segments (targets):**
  - `20260220213127-w00`
  - `20260220214732-auto_grown_20260220144552896`
  - `20260221022814-auto_grown_20260220174252405`

  Each comes with its 9.366 µm mesh (`mesh/<id>-on-20250821151531-9.366um.tifxyz`) and its ink labels and supervision mask, drawn on its 2.403 µm render (`ink-labels/2.403um-volume-20260319124803/<date>/`).
- **Windings:** KLAVIS's four spiral-fit windings, w055_fr0, w058_fr0, w063_fr0b and w056_fr0b (tifxyz, about 23.5 voxels between vertices). They are used as received.

## Carrying the labels to 9.366 µm

- `tools/grid_correspondence.py` fits each target's 2.403 µm grid to its 9.366 µm grid: fine row = f_r × coarse row + o_r, and the same for columns. It is used only if the median residual is under 1 voxel. On w00, the residual is 0.12 voxels, with f ≈ 3.900 and o ≈ 1.45.
- `tools/fine_labels.py transfer`:
  - **Footprints.** Each 9.366 µm pixel of the target's render grid covers a footprint of about 3.9 × 3.9 label pixels.
  - **Supervised** means at least half the footprint is supervised.
  - **Ink** means at least half of the supervised part is ink.
- **Area covered.** The bounding box of the supervision mask, plus 128 pixels.

## Surfaces compared

Each surface's depth is carried onto a target's pixel grid along the target's normals with `tools/depth_transfer.py`:
- within 10 voxels of the target;
- hit within 2 voxels of a surface sample;
- with normals within 37° of each other (`--min-dot 0.8`).

The target is then rendered at that depth with `tools/render_depth.py`: along its own normals, 21 layers 1 voxel apart, `--normals flipped --center 1` as in the w052 test. So only the depth differs between surfaces on the same target, and the target's labels line up with every render.

**A. The labelled segments themselves.** Per target:
1. the target as traced (depth 0);
2. the target snapped with `tools/snap_segment.py`, default settings, dense output.

**B. KLAVIS's windings.** A winding is paired with a target if its original depth reaches at least 20,000 of the target's supervised pixels. Per pair:
1. the winding as received;
2. snapped with default settings, drop-in output (`<name>_snapped.tifxyz`), the one KLAVIS scored;
3. snapped with default settings, dense output;
4. snapped with `--target-offset 0 --param smooth_w=0.1`, dense output, KLAVIS's second try.

All snapping uses `tools/snap_segment.py` at the commit that adds this plan, on whole segments and windings, with 8 workers. If the snapper can't settle a target's recto side from its curvature, that target is left out of A and this is reported. A.1 is rendered through the same depth transfer, with the target as its own surface (depth 0).

## Ink model

- `ink_9um` `hybrid_3d2d-seed42/step-075000.pth`, through the official inference, with the settings and environment of the w052 test (villa `deb1b69`, CPU, `--overlap 0.5 --blend-mode hann --batch-size 4 --loader-threads 4`).
- **Layer direction:** per target, the one with the higher 99th-percentile probability over the valid pixels of the target's own render (A.1), in both directions. It is then used for every surface on that target.

## Scoring

`tools/fine_labels.py score`, per target (A) and per pair (B):
- **Pixels scored:** pixels that are supervised, valid in every render of the comparison, and non-zero in every map.
- **Measures:** ROC AUC (ties count half) and the best F1 of each map.
- **Differences** are each snapped version minus the original (A.1, or B.1). Each comes with a 95% interval from a paired bootstrap over 128 × 128-pixel blocks (1000 resamples, seed 0).
- **Reading.** A difference whose interval excludes 0 is reported as a gain or a loss. Otherwise it is "no detectable difference".
- **Pooled result.** Also reported pooled over targets (A) and over pairs (B): the mean difference, with each target or pair weighted by its scored pixels.

## Reporting

All results go into `results/ink_test_0841.md`, the README and the write-up, whichever way they come out. Everything is credited to KLAVIS for the windings, the idea and the first numbers.

## Amendments

(None yet.)
