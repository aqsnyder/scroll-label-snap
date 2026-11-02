# scroll-label-snap

Snap approximate recto-surface labels in Herculaneum scroll CT volumes back onto the real papyrus surface.

This is work in progress toward a [Vesuvius Challenge](https://scrollprize.org/prizes) progress-prize submission. It targets the "label quality" open problem in the [2026 open problems](https://scrollprize.org/2026_open_problems): surface models are trained on hand-traced labels that drift off the true surface.

## Status

| Piece | State |
|---|---|
| `tools/inspect_cube.py`: shapes, label values, slice overlays for one cube | done |
| `tools/diag_labels.py`: how far labels sit from the nearest papyrus edge | done (first version) |
| Snapper: CT edge cost + graph-cut regularization + confidence map | next |
| Evaluation: edge alignment, profile sharpness, hand-checked points | planned |

### First measurement (3 cubes, 20,000 labeled points each)

A clear papyrus-to-gap edge is visible at about 80% of labeled points. Taking the typical label position as zero, the share of those points whose edge is:

| Distance from the label | Share of points |
|---|---|
| within 1 voxel | 55–61% |
| within 2 voxels | 83–85% |
| more than 3 voxels | ~6% |

The edge finder in this first version takes the steepest local edge at each point independently, so it partly locks onto noise. These numbers overstate how far labels really are from the surface. The regularized snapper is meant to separate real offsets from noise.

## Data

The tools run on the Vesuvius Challenge surface-label cubes ([Hugging Face bucket](https://huggingface.co/buckets/scrollprize/datasets/tree/surfaces/kaggle)): `images/sample_NNNNN.tif` paired with `labels/sample_NNNNN.tif`. Each cube is 320³ uint8.

| Label value | Meaning |
|---|---|
| 0 | annotated background |
| 1 | recto surface |
| 2 | not annotated |

The data is licensed CC BY-NC 4.0 and is not included in this repo.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# look at one cube
python tools/inspect_cube.py images/sample_00001.tif labels/sample_00001.tif out/inspect_00001.png

# label-to-edge diagnostic over any number of image/label pairs
python tools/diag_labels.py out/diag images/sample_00001.tif labels/sample_00001.tif \
                                     images/sample_00003.tif labels/sample_00003.tif
```
