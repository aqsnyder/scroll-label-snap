# PHerc. 0841 follow-up: snapping and agreement with ink labels

This test was pre-registered in [`ink_test_0841_plan.md`](ink_test_0841_plan.md), pushed at 627d058 before any surface in it was snapped, rendered or run through the ink model. KLAVIS suggested it in the Vesuvius Challenge Discord and supplied the four spiral-fit windings and the first numbers. It ran on 3 November 2026 with the ink model, settings and PC of the w052 test ([`ink_test.md`](ink_test.md)). The data files are in [`ink_test_0841/`](ink_test_0841/).

## Summary

- **The labelled segments themselves (A).** Snapping made no detectable difference on any of the three. At the plan's seed, all three moved up slightly: ΔAUC +0.006, +0.004 and +0.022, pooled +0.011. Every interval includes 0.
- **KLAVIS's windings (B)**, five winding–segment pairs:
  - **Drop-in output** (the one KLAVIS scored): as run, one gain, w056 on w00, of +0.104 (95% CI +0.032 to +0.184). The other four show no detectable difference. Pooled: +0.019. The gain doesn't repeat: the same pair gives −0.081 at seed 43 with the same transfer, and +0.014 and −0.008 with the transfer fixed.
  - **Dense output:** as run, two losses (−0.19 and −0.22), and three with no detectable difference. Pooled: −0.078. Part of this was a bug in the depth transfer: with it fixed, pooled −0.051 and −0.002 at seeds 42 and 43.
  - **KLAVIS's second try** (`--target-offset 0 --param smooth_w=0.1`, dense): as run, three losses. Pooled: −0.070. With the transfer fixed, pooled −0.050 and +0.002.
- **In short:** snapping made no detectable difference on traced segments (read as 0 ± 0.05 AUC). On rough windings, the dense output lost ground as run, and part of that came from a bug in our depth transfer. With it fixed (post hoc, exploratory), the pooled loss is −0.051 at the plan's seed and −0.002 at a second model seed, and neither output shows a loss or a gain that repeats at both seeds (see "Later check" below).

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
  - With depth carried over at every pixel, it is about even here too, within 0.05 and leaning negative: pooled +0.019 as run, −0.047 at seed 43 with the same transfer, and −0.022 and −0.027 with the transfer fixed.
  - The one gain as run is on the pair where the winding as received scored lowest (AUC 0.553). It doesn't hold up: at seed 43, with the same transfer, that winding scores 0.666 and the pair gives −0.081.
- **The dense output.** It follows the scan at about 3-voxel detail. As run, it lost about 0.2 AUC on two of the five pairs, and 0.16 on a third, where the interval just includes 0. Part of that was a bug: the depth transfer dropped 19.5% of the dense output's hits and 11.5% of the drop-in output's, and the model's map changes around the holes. The bug accounts for more of the gap to the drop-in output than of the loss against the winding as received. Whether the detail itself costs anything isn't settled. With the transfer fixed, dense minus drop-in is −0.029 and +0.026 at seeds 42 and 43, both including 0. That result depends on the fix: −0.051 without its smoothing and −0.040 with an exact intersection, both excluding 0 (see "Later check" below). On the traced segments, the dense output showed no detectable loss (part A).
- **Not the same measure as the w052 review.** This test measures agreement with ink labels, on a scroll the model hasn't seen. The w052 test measured legibility by eye, with the drop-in output.

## Later check: a bug in the depth transfer (post hoc, exploratory)

This was added after everything above. It is exploratory and changes no pre-registered outcome: the tables above remain the plan's outcome. KLAVIS suggested the test and supplied the windings and the first numbers. KLAVIS's own scoring doesn't use this tool, so those first numbers aren't affected. The rerun, its rule and its readings came from an automated review ([`review_2026_11.md`](review_2026_11.md)), and no person has reviewed the new maps.

**The bug.** The plan asks for a hit within 2 voxels of a surface sample. `tools/depth_transfer.py` at 627d058 measured that to the sample nearest the target's pixel, not the sample nearest the hit. So correct hits failed, more often the deeper the surface lay. Those pixels stayed empty and rendered as zeros through all 21 layers.

An exact ray-mesh intersection, written separately, shows how many hits were dropped:

| Surface | Hits dropped |
|---|---|
| Traced targets | 0 |
| Snapped targets | 0.5% |
| Windings as received | 8.9% |
| Drop-in output | 11.5% |
| Dense output | 19.5% |
| Second try | 21.1% |

So it isn't specific to 3-voxel surfaces, though they lost about twice as much. On the dense output, the share rises from 1.9% under 1 voxel of depth to 57% at 8–10 voxels.

**The rerun.** `98542c1` adds `--refine` and `--normal-sigma`, off by default. With them off, all 26 depth files reproduce bit for bit. Everything was redone with `--refine 3 --normal-sigma 10` at seeds 42 and 43, with the plan's model, settings and scoring. The smoothing and its scale were chosen after the plan. Two checks ran at seed 42 only: refinement without the smoothing, and the exact intersection. The readings were fixed in [`ink_test_0841_rerun/rule.md`](ink_test_0841_rerun/rule.md) before these maps existed, but after one pair had been scored, so they aren't blind. The rule wasn't pushed then, so only local file times show the order. The tables are in [`ink_test_0841_rerun/`](ink_test_0841_rerun/).

**Pooled over the five pairs.** Each version is scored by the plan's pixel rule, so the fixed rows score about a third more pixels (2,237,195 against 1,690,104), mostly deeper ones. Bold marks an interval that excludes 0. The plan reported pooled means only; these intervals come from the rerun's rule.

| Contrast | As run, seed 42 (the plan's outcome) | Same transfer, seed 43 (post hoc) | Fixed, seed 42 | Fixed, seed 43 |
|---|---|---|---|---|
| Drop-in − as received | +0.019 (−0.024 to +0.061) | **−0.047 (−0.089 to −0.002)** | −0.022 (−0.052 to +0.011) | −0.027 (−0.063 to +0.008) |
| Dense − as received | **−0.078 (−0.129 to −0.028)** | **−0.098 (−0.139 to −0.059)** | **−0.051 (−0.093 to −0.004)** | −0.002 (−0.037 to +0.041) |
| Second try − as received | **−0.070 (−0.121 to −0.020)** | **−0.063 (−0.105 to −0.024)** | **−0.050 (−0.093 to −0.003)** | +0.002 (−0.032 to +0.038) |
| Dense − drop-in | **−0.097 (−0.142 to −0.054)** | **−0.051 (−0.090 to −0.011)** | −0.029 (−0.061 to +0.004) | +0.026 (−0.006 to +0.059) |
| Snapped − traced (part A) | +0.011 (−0.017 to +0.037) | −0.007 (−0.034 to +0.022) | +0.010 (−0.018 to +0.038) | +0.016 (−0.009 to +0.042) |

**The readings fixed in advance.**

- **Is the dense output's gap to the drop-in output an artefact? Supported at both seeds.** With the fix, both pooled intervals include 0, no pair's interval is below 0, and the gap narrows by 0.068 and 0.077.
- **That depends on the post-hoc smoothing.** At seed 42, refinement without smoothing gives −0.051 (−0.086 to −0.016): not supported.
- **It depends on the transfer method.** At seed 42, the exact intersection gives −0.040 (−0.073 to −0.004): partly. No pair moves more than 0.05 between it and the fix (largest 0.025).
- **The holes as the mechanism, per pair.** Among pairs with a dense-minus-drop-in loss as run, two things were tested: giving the drop-in output the dense output's holes, and filling the dense output's holes. Both did what the mechanism predicts on 1 of 4 pairs at seed 42, and 1 of 3 at seed 43.
- **Against the winding as received, with the fix.** Drop-in: no detectable pooled difference at both seeds. Dense and second try: seeds disagree (the interval excludes 0 at seed 42 and includes it at seed 43). No pooled or per-pair row repeats at both seeds, in either direction.
- **Part A.** Pooled, the fix moves snapped − traced by 0.000 at seed 42, and by +0.022 at seed 43 (+0.025, +0.021 and +0.0205 per target). The rule flags moves over 0.02: all four are flagged at seed 43, none at seed 42. Per target at seed 43, same transfer: +0.020, −0.056 (−0.111 to −0.003) and −0.002. Fixed: +0.044 (+0.007 to +0.079), −0.035 and +0.019. Neither that loss nor that gain repeats at seed 42.
- **Noise.** Pooled differences between the two seeds run up to 0.066. These are smaller than their own seed difference, so they aren't reported as effects:
  - as run: the drop-in output (+0.019, −0.047);
  - fixed: dense − drop-in (−0.029, +0.026), the second try (−0.050, +0.002), and the dense output at seed 43 (−0.002);
  - per pair: the as-run gain on w056 on w00, whose seed difference is 0.185.

  The fixed dense output at seed 42 (−0.051) clears its seed difference (0.049) by 0.002. No random-move control was run on windings. On the traced targets, small moves change the pooled AUC by at most 0.021, and random moves with the same spectrum by −0.08 to −0.11.

**Added afterwards (not in the fixed readings).**

- **Which pixels decide.** The rule defines two pixel sets and doesn't say which decides. The readings above use primary, a choice made after the results were read. On secondary pixels (the as-run pixels), the artefact reading is the same and the smoothing still changes its category (partly, −0.040). The exact intersection reads supported (−0.035, −0.073 to +0.000), though that depends on the bootstrap draw, and the second try reads no detectable pooled difference at both seeds (−0.043, +0.001).
- **On identical pixels**, the gap narrows by +0.074 (+0.028 to +0.119) and +0.081 (+0.038 to +0.122). Mostly this is the holes, not the depth change: filling only the holes gains +0.074 and +0.071 on the pixels the plan's rule scores.
- **It holds when pooled, not pair by pair.** Only w056 on …144552896, the pair that prompted this check, passes alone at both seeds. Without it, the gap narrows by +0.059 and +0.069 on identical pixels, and by +0.048 and +0.065 on each version's own primary pixels, where +0.048 misses the rule's 0.05. Averaged per target, the intervals are −0.034 to +0.200 and +0.002 to +0.171.
- **The zeros don't act through the normaliser.** In a 2 × 2 test of holes in the input against holes in the normaliser's statistics, the input carries the effect (+0.075, +0.068 on the dense output) and the statistics almost none (+0.004, +0.005). The map's level on non-ink pixels is raised out to about 32 pixels from a hole. So the plan's rule removes hole pixels but not the pixels they affect.
- **The fix adds a different population.** 22–31% of the pixels each pair scores with the fix are new. They sit deeper (median 6.2–7.2 voxels, against 2.4–3.5), and sheet spacing here is 9–14 voxels, close to the plan's 10-voxel limit.

**What this changes (also added afterwards).**

- **Dense against drop-in.** Pooled, most of that gap came from this bug. The first version of this page (e093829) gave the 3-voxel detail as the cause. As a full explanation, that doesn't stand.
- **Dense against as received.** The fix removes 35% of the pooled loss at seed 42 and 98% at seed 43 (36% and 92% on identical pixels). The two pairs flagged above stay negative: w055 −0.115 and −0.054, w056 on …144552896 −0.074 and −0.020.
- **Not shown: that snapping is harmless on windings.** No loss repeats, but no gain does either, and the intervals still allow a pooled loss of up to about 0.09. This is 5 pairs on 3 targets, and the two seeds are two checkpoints reading the same renders.
- **The w052 test isn't touched.** Its renders don't go through this tool, and they have almost no small holes (0.32 and 0.47 per megapixel, original and snapped).
