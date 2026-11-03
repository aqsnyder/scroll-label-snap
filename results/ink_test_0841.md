# PHerc. 0841 follow-up: snapping and agreement with ink labels

This test was pre-registered in [`ink_test_0841_plan.md`](ink_test_0841_plan.md), pushed at 627d058 before any surface in it was snapped, rendered or run through the ink model. KLAVIS suggested it in the Vesuvius Challenge Discord and supplied the four spiral-fit windings and the first numbers. It ran on 3 November 2026 with the ink model, settings and PC of the w052 test ([`ink_test.md`](ink_test.md)). The data files are in [`ink_test_0841/`](ink_test_0841/).

## Summary

- **The labelled segments themselves (A).** Snapping made no detectable difference on any of the three. All three moved up slightly: ΔAUC +0.006, +0.004 and +0.022, pooled +0.011. Every interval includes 0.
- **KLAVIS's windings (B)**, five winding–segment pairs:
  - **Drop-in output** (the one KLAVIS scored): one gain, w056 on w00, of +0.104 (95% CI +0.032 to +0.184). The other four show no detectable difference. Pooled: +0.019.
  - **Dense output:** two losses (−0.19 and −0.22), and three with no detectable difference. Pooled: −0.078.
  - **KLAVIS's second try** (`--target-offset 0 --param smooth_w=0.1`, dense): three losses. Pooled: −0.070.
- **In short:** snapping was neutral on traced segments. On rough windings, the dense output can lose much of the ink signal, while the drop-in output does not.

## How it ran

- **Labels to 9.366 µm.** Each target's 2.403 µm grid fits its 9.366 µm grid with a median residual of 0.10–0.12 voxels (f ≈ 3.90, o ≈ 1.4–1.45), well under the 1-voxel limit.
- **Layer direction:** forward on all three targets. The 99th percentiles were 0.769 forward against 0.714 reverse on w00, 0.761 / 0.686 on …144552896, and 0.765 / 0.737 on …174252405.
- **Snapping the targets.** The snapper settled the recto side at 98% of vertices on each target, so none was left out of A. Each target's dense snap:
  - face offsets of +0.00, +0.50 and +0.50 voxels;
  - mean |shift| 1.94–1.97 voxels;
  - 75% of samples moved by 1 voxel or more;
  - 16–19% of vertices marked for review.
- **Pairs.** A winding is paired with a target when the winding as received reaches at least 20,000 of the target's supervised pixels:

| Target (supervised pixels) | w055_fr0 | w056_fr0b | w058_fr0 | w063_fr0b |
|---|---|---|---|---|
| w00 (1,600,311) | 0 | **574,678** | 12,014 | 32 |
| …144552896 (983,772) | **377,468** | **307,141** | 93 | 0 |
| …174252405 (1,279,388) | 0 | 0 | **102,390** | **893,960** |

## A. The labelled segments, traced and snapped

| Target | Scored pixels | Ink share | Traced: AUC / best F1 | Snapped (dense): AUC / best F1 | ΔAUC (95% CI) | Reading |
|---|---|---|---|---|---|---|
| w00 | 1,595,783 | 22% | 0.752 / 0.507 | 0.758 / 0.515 | +0.006 (−0.035 to +0.046) | no detectable difference |
| …144552896 | 982,317 | 32% | 0.707 / 0.552 | 0.711 / 0.554 | +0.004 (−0.041 to +0.053) | no detectable difference |
| …174252405 | 1,275,522 | 23% | 0.717 / 0.493 | 0.739 / 0.515 | +0.022 (−0.025 to +0.071) | no detectable difference |
| **Pooled**, weighted by pixels | 3,853,622 | | | | +0.011 | |

## B. KLAVIS's windings

These are AUC differences from the winding as received, with 95% intervals from the block bootstrap. Bold marks an interval that excludes 0.

| Winding on target | Scored pixels | Ink share | As received: AUC | Drop-in output | Dense output | Offset 0, smooth 0.1, dense |
|---|---|---|---|---|---|---|
| w056 on w00 | 450,459 | 24% | 0.553 | **+0.104 (+0.032 to +0.184)** | −0.031 (−0.140 to +0.081) | −0.016 (−0.130 to +0.108) |
| w055 on …144552896 | 278,139 | 34% | 0.750 | −0.069 (−0.173 to +0.045) | **−0.187 (−0.301 to −0.070)** | **−0.189 (−0.299 to −0.083)** |
| w056 on …144552896 | 197,746 | 29% | 0.767 | −0.027 (−0.109 to +0.058) | **−0.220 (−0.334 to −0.117)** | **−0.204 (−0.326 to −0.093)** |
| w058 on …174252405 | 69,444 | 42% | 0.835 | +0.038 (−0.080 to +0.175) | −0.155 (−0.367 to +0.019) | **−0.170 (−0.348 to −0.015)** |
| w063 on …174252405 | 694,316 | 23% | 0.669 | +0.011 (−0.060 to +0.071) | −0.018 (−0.090 to +0.054) | −0.010 (−0.085 to +0.060) |
| **Pooled**, weighted by pixels | 1,690,104 | | | +0.019 | −0.078 | −0.070 |

Each pair is scored on the part of the target its winding covers. So these numbers can't be compared with part A, or across pairs. The best-F1 differences are in the score files and go the same way.

## Reading

- **The drop-in output.**
  - KLAVIS found it about even with his own scoring, which reads depth about every 20 voxels: −0.011, −0.005, −0.001 and +0.022 per winding.
  - With depth carried over at every pixel, it is about even here too (pooled +0.019).
  - The one clear gain is on the pair where the winding as received scored lowest (AUC 0.553).
- **The dense output.** It follows the scan at about 3-voxel detail. On these windings, that lost about 0.2 AUC on two of the five pairs, and 0.16 on a third, where the interval just includes 0. The drop-in output avoided this: it is fitted back onto the winding's own vertices, about 23 voxels apart. On the traced segments, the dense output didn't lose (part A).
- **Not the same measure as the w052 review.** This test measures agreement with ink labels, on a scroll the model hasn't seen. The w052 test measured legibility by eye, with the drop-in output.
