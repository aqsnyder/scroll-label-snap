# One orientation per cube: `orient_cube` passed its pre-registered rule

This test was pre-registered in [`orient_test_plan.md`](orient_test_plan.md). It tests one option, `Params.orient_cube`, on 100 cubes drawn at random from the 561 left after the plan's exclusions. `orient_normals` ends with a step that flips any sheet disagreeing with the samples around it. The option replaces that step: each sheet takes the side of the cube's dominant normal axis, and the votes of all sheets together choose which way the axis points.

By its own account, the plan was committed before any drawn cube was run with the option. It was first committed at 7785156 on 3 November 2026 and last changed at ae9f9c0, at 00:27 on 4 November, with its amendments listed in it. The test ran on 4 November 2026 from 08:32 to 09:37 (UTC−5), 64.8 minutes, at commit 678121a09b93200ed872d241b44b24c766313325. It ran on one Windows PC, CPU only, with 16 workers, Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1 and PyMaxflow 1.3.2. No cube failed.

The repository owner approved the run. An automated review ran it, checked the result and wrote this report, with no person checking.

The decision and numbers from the plan's tool, `tools/eval_orient_test.py`, are in [`orient_test.json`](orient_test.json), with one row per cube in [`orient_test_cubes.csv`](orient_test_cubes.csv).

## Summary

**The plan's condition is met. It says "Adopt only if the interval's lower end is above 0, and all the guards hold", and on the plan's 100 cubes the lower end is above 0 and all the guards hold. The tool reports `adopt: true`.**

| Check | Result | Limit | Holds |
|---|---|---|---|
| **Primary:** consistency, `orient_cube` minus default | 75.40% → 82.67%, **+7.27 points** (95% interval 5.68 to 8.98) | Lower end above 0 | Yes |
| **Guard:** drift, `orient_cube` minus default | 87.19% → 89.44%, **+2.25 points** (95% interval 1.31 to 3.41) | Lower end above −1 | Yes |
| **Guard:** orientation change under `orient_cube` | **0.013%** on average, against 10.61% for the default | At most 1% | Yes |
| **Guard:** fibre signature on disputed samples | Positive on **41 of 46** qualifying cubes, 89.1% | At least 75% | Yes |

- **The third arm passes its own rule:** Betti numbers equal the original's and merger events are 0 on 100 of 100 cubes. As the plan says, it is not adopted from this test, whichever way it came out.
- **The option stays off unless adopted.** `Params.orient_cube` is still `False`, and this test changed no published label. Adopting would change published labels, so it needs a re-snap of all 892 cubes, which this test doesn't do.
- **The rule's measures have limits.** Consistency measures stability, not accuracy. What else limits them, and the rest of what was found later, is under [Added afterwards](#added-afterwards-not-part-of-the-plan), kept apart because the plan didn't ask for it.

## What the numbers mean

The arms are the default, `orient_cube=True` (arm 2), and `orient_cube=True, keep_topology=True` (the third arm), with the other parameters at their defaults. Each runs on a cube's clean labels and on three drifted versions of them (seeds k, 100+k and 200+k for cube k, std 1.5 voxels, correlation 40). Positions are measured at the clean labels' samples at least 10 voxels from every face.

- **Consistency:** among points where both bands exist, the share where the band from drifted labels is within 1 voxel of the band from clean labels, in the same arm.
- **Drift:** the share where the band from drifted labels is within 1 voxel of the original label, at the fixed set of points trusted under both arms' normals on the clean labels, where the band exists. A trusted point's original label sits on a strong, unambiguous papyrus edge.
- **Orientation change:** the share of points whose nearest sample in the drifted run, within 6 voxels, has a reversed normal.
- **Disputed samples:** those where the default's and `orient_cube`'s normals disagree on the clean labels.
- **Fibre signature:** compares the CT's fibre texture just inside a sheet's two faces, under `orient_cube`'s normals. Positive means the normals agree with the texture. It uses no label polarity or CT step sign, though the samples' positions and normal lines come from the labels, and it assumes the scroll axis is scan z. A cube qualifies with at least 1,000 disputed samples that have fibre data.
- **Missing:** the share of points with no band to measure. It doesn't count against an arm.
- **Axis share:** the normals' squared projection on their dominant axis. Below 70%, every sheet keeps its own vote (the 70% guard).
- **Sign margin:** how decisively the votes choose which way the axis points, from 0 to 1.
- **Vote:** a sheet's own CT vote on which way it faces.
- **Betti numbers and merger events:** the band's components, tunnels and cavities, and the original sheets that end up in one component (`tools/eval_topology.py`). The merger count is a rule, not a tracking of voxels.
- **Contested samples:** samples that end within 3 voxels of a moved sample of another input sheet, at least 0.5 voxel closer than before.

Differences are per cube, averaged over the three seeds, in points. Intervals are 95%, by resampling cubes (5,000 resamples, seed 0). "Better" and "worse" mean by more than 0.5 points.

## What the plan asks to report

### Per arm

Means over the 100 cubes.

| | Default | `orient_cube` | Third arm |
|---|---|---|---|
| Consistency within 1 voxel | 75.40% | 82.67% | 82.47% |
| Drift within 1 voxel | 87.19% | 89.44% | 89.46% |
| Orientation change | 10.61% | 0.013% | 0.013% |
| Missing, consistency points | 0.213% | 0.190% | 0.021% |
| Missing, drift points | 0.027% | 0.013% | 0.004% |
| Contested samples | 0.316% | 0.333% | 0.333% |
| Cubes with Betti numbers equal to the original's | 0 | 0 | 100 |
| Merger events (cubes with any) | 174 (65) | 171 (66) | 0 (0) |

### Ids 161-280 against the others

The plan reports these apart because the topology audit found that the CT in that stretch rarely resolves gaps. The decision uses all 100. The stretch starts lower and stays lower, and the option doesn't close that gap.

| | All 100 | Ids 161-280 (13) | Other ids (87) |
|---|---|---|---|
| Consistency, default → `orient_cube` | 75.40 → 82.67 | 69.48 → 75.68 | 76.28 → 83.72 |
| Difference (interval) | +7.27 (5.68 to 8.98) | +6.21 (2.55 to 10.15) | +7.43 (5.69 to 9.27) |
| Cubes better / worse | 63 / 2 | 9 / 0 | 54 / 2 |
| Drift, default → `orient_cube` | 87.19 → 89.44 | 85.37 → 87.00 | 87.46 → 89.81 |
| Drift difference (interval) | +2.25 (1.31 to 3.41) | +1.63 (0.33 to 3.31) | +2.34 (1.30 to 3.65) |
| Drift cubes better / worse | 36 / 4 | 5 / 1 | 31 / 3 |
| Orientation change, default | 10.61% | 9.65% | 10.75% |
| Orientation change, `orient_cube`, mean (highest cube) | 0.013% (0.095%) | 0.007% (0.019%) | 0.014% (0.095%) |
| Fibre positive, of qualifying cubes | 41 of 46 | 7 of 8 | 34 of 38 |
| Inputs where the 70% guard fired | 0 of 400 | 0 of 52 | 0 of 348 |
| Lowest axis share | 0.716 | 0.794 | 0.716 |
| Lowest sign margin | 0.336 | 0.384 | 0.336 |
| Samples turned against their sheet's vote, mean (highest cube) | 4.41% (40.14%) | 8.25% (31.05%) | 3.84% (40.14%) |
| Disputed samples, mean (highest cube) | 10.70% (67.08%) | 12.36% (40.24%) | 10.46% (67.08%) |
| Missing, consistency: default / `orient_cube` / third arm | 0.213 / 0.190 / 0.021% | 0.391 / 0.382 / 0.014% | 0.186 / 0.161 / 0.022% |
| Missing, drift: default / `orient_cube` / third arm | 0.027 / 0.013 / 0.004% | 0.055 / 0.042 / 0.019% | 0.023 / 0.009 / 0.002% |
| Third arm: consistency | 82.47 | 75.54 | 83.51 |
| Third arm: drift | 89.46 | 87.04 | 89.82 |
| Third arm: cubes meeting its rule | 100 of 100 | 13 of 13 | 87 of 87 |
| Third arm: contested samples | 0.333% | 0.748% | 0.271% |
| Third arm: band over 1 voxel from arm 2's, at contested samples / elsewhere | 7.60 / 0.063% | 5.02 / 0.134% | 8.25 / 0.052% |

### Per cube

The full list is in [`orient_test_cubes.csv`](orient_test_cubes.csv). Axis share and sign margin are the lowest of a cube's four inputs.

| | Lowest | Median | Highest |
|---|---|---|---|
| Consistency difference | −2.92 (sample_00469) | +4.36 | +34.41 (sample_00114) |
| Drift difference | −2.41 (sample_00392) | +0.05 | +41.91 (sample_00114) |
| Orientation change, default | 0.010% (sample_00862) | 6.44% | 66.64% (sample_00114) |
| Orientation change, `orient_cube` | 0.000% | 0.009% | 0.095% (sample_00340) |
| Axis share | 0.716 (sample_00392) | 0.864 | 0.934 (sample_00828) |
| Sign margin | 0.336 (sample_00690) | 0.982 | 0.997 (sample_00816) |
| Samples turned against their sheet's vote | 0.04% (sample_00088) | 0.17% | 40.14% (sample_00711) |
| Disputed samples | 0.03% (sample_00168) | 2.11% | 67.08% (sample_00690) |

- **Consistency:** 63 cubes better, 2 worse and 35 within 0.5 points. The worse ones are sample_00469 (−2.92) and sample_00397 (−1.98).
- **Drift:** 36 cubes better, 4 worse and 60 within 0.5 points. The worse ones are sample_00392 (−2.41), sample_00411 (−2.12), sample_00229 (−1.09) and sample_00108 (−0.67).
- **Fibre signature:** the 5 negative qualifying cubes are sample_00680 (−0.129), sample_00488 (−0.066), sample_00198 (−0.039), sample_00287 (−0.036) and sample_00469 (−0.009). The guard can't count the other 54 cubes: 35 have 1 to 999 disputed samples with fibre data and 19 have none.

### Against the development cubes

The plan's Why gives these figures from the development cubes and says "Whether that holds on fresh cubes is what this test asks". It sets no limit on them, and the rule doesn't score them. They did not hold.

| | Development cubes (plan) | These 100 cubes |
|---|---|---|
| Lowest sign margin | 0.8996 | 0.336 (sample_00690); 21 cubes below 0.8996 |
| Lowest axis share | 0.797 | 0.716 (sample_00392); 10 cubes below 0.797 |
| Samples turned against their sheet's vote | 1.89% | 4.41% as a mean over cubes, 4.59% weighted by samples |
| The same, highest cube | 12.9% | 40.14% (sample_00711); 12 cubes above 12.9% |

### The third arm

- **It passes.** On the clean labels' output, Betti numbers equal the original's at both connectivities and merger events are 0, on 100 of 100 cubes. So there are no events to list, and component counts match on all 100. For comparison, the other two arms have Betti numbers equal on 0 of 100 cubes, with 174 and 171 merger events.
- **Against arm 2** (`orient_cube` alone): consistency −0.20 points (interval −0.22 to −0.18), and drift on the same fixed set +0.017 points (interval 0.004 to 0.030).
- **Contested samples** average 0.333% of a cube's samples over all 100 cubes. 54 cubes have any, with 4.02% at most (sample_00828). Its band differs from arm 2's by more than 1 voxel at 7.60% of them (mean over those 54 cubes) and at 0.063% of other samples. A sample with no band counts as not moved.
- **It is not adopted from this test, whichever way it came out,** because the drift test barely scores the samples it changes. The plan requires a blind review at contested sites first, and that hasn't been done.

## Added afterwards, not part of the plan

Nothing here feeds the rule. All of it was written after the result was known, by the same automated review, and the cut-offs in it (the top 20, the 10% trim, the 1% split, the sheet sizes, the t limits, the gap between stacks) and the 3 and 13 cubes rerun were chosen after seeing the data. No parameter, list, seed or threshold in the plan or the tool was changed. Scripts outside the tool produced these numbers, and they and their outputs aren't in this repository.

**The result repeats, which is not the same as being independently confirmed.**
- Separate code recomputed the decision and every field of both files from the 100 cube records, with no difference above 3 × 10⁻¹⁴ points. It was written after reading the tool, so it could share a misreading.
- All 100 cubes rerun at all 3 seeds gave every stored consistency and drift value of the default and `orient_cube` to within 10⁻¹². The tool itself, rerun with 3 workers on sample_00028, 00470 and 00893 (positions 0, 50 and 99 of the plan's list), matched in full apart from time. This is the same code on the same PC, so it shows the result is deterministic.
- A separately written band measure on those 3 cubes kept the sign of `orient_cube` minus default on all 9 seeds. Its level is 4.6 to 7.6 points lower on all 27 readings; the tool's formula applied to the same volumes matched exactly on the 4 readings tried, so that gap is in how the band is defined. The arms differ by at most 0.07 points per cube on those 3, so this says little about the effect. No cube with a large effect has an independent reading.
- The originals' Betti numbers, recomputed from the labels by a different method, agree with the stored comparison on all 3 arms and 100 cubes. The third arm's own band wasn't saved, so its Betti numbers and merger events are the tool's reading.
- Before the run, the tool on development cube sample_00018 matched the figures in commit 7785156 within 0.05 points at that commit's seeds (17, 117, 217). At seeds 0, 100 and 200 it differed by up to 0.71 points. The run went ahead on the reading that different seeds explain this.

**The primary result's sign doesn't rest on a few cubes, missing points, a seed or a cube size, though its size depends on the largest gains.**
- Median +4.36 (1.72 to 6.95). 10% trimmed mean +6.02 (4.37 to 7.87). Without the top 20 cubes, +3.80 (2.79 to 4.85). Without the 23 ids the plan lists as read earlier, +7.57 (5.68 to 9.50).
- Counting every missing point as a miss gives +7.27 (5.68 to 8.98), and +2.26 on drift (1.31 to 3.42). The arms' point counts differ by up to 209 on drift and 918 on consistency, and the worst overlap those counts allow leaves the lower ends at 1.29 and 5.49.
- By seed offset 0, 100 and 200: +8.03 (6.25 to 9.96), +6.96 (5.21 to 8.77) and +6.82 (5.09 to 8.70).
- By drawn size: 320-cubed (90 cubes) +7.32 (5.61 to 9.15), 256-cubed (10 cubes) +6.81 (2.69 to 11.50).

**Consistency rewards stability: a steady wrong sign can still beat the default, so a gain doesn't show the signs are right.**
- 95.4% of the gain comes from points the default reversed under drift. Those points add +6.94 (5.44 to 8.55) and the rest +0.34 (0.09 to 0.59).
- On the 38 cubes where the default reverses under 1% of points, the difference is −0.04 (−0.27 to +0.16). On the other 62 it is +11.76 (9.84 to 13.69).
- As a control, on 13 cubes picked by hand (the largest gains, the losses, the fold below and cubes with almost no dispute), every normal of `orient_cube` was reversed. Consistency was 77.73%, against 71.13% for the default and 81.73% for `orient_cube`: below `orient_cube` on all 13 cubes, but above the default on 7. Drift separates them more: 54.44%, against 83.94% and 89.57%, below `orient_cube` on all 13 and below the default on 11.

**The drift guard barely scores the changed signs.**
- A point in its fixed set must be trusted under both orientations, so the set holds 0.157% of disputed points, and none at all in 38 of the 50 cubes with at least 1,000 disputed points.
- Its +2.25 has a median of +0.05, and 5 cubes hold 44% of the total. Fixed points the default reversed (287 per cube on average in the 78 cubes that have any, median 47) gain +36.26 (28.82 to 43.78). The other fixed points gain −0.01 (−0.19 to +0.17).
- Over all points, against the original label, the gain is +0.41 (0.17 to 0.64), and on disputed points −0.16 (−1.50 to +1.13). Of the 50 cubes with at least 1,000 disputed points, five lose on them by much more than the guard shows: sample_00469 −25.51 (guard +0.01), sample_00506 −17.27 (+2.86), sample_00711 −14.30 (+8.98), sample_00198 −11.19 (+4.39) and sample_00397 −4.24 (+0.74). The original label isn't truth, so a move off it could be a correction.
- The drift losses mostly follow the cube-wide convention offset, where the steepest edge sits relative to the label along the normals. The snapper estimates it on each run from the labels and the normals it is given, so the option's signs can change it. On fixed points the default didn't reverse, the change averages −1.68 in the 15 seed runs where `orient_cube`'s offset is lower, +2.91 in the 9 where it is higher, and −0.01 in the other 276. sample_00392, 00411 and 00108 lose mainly in those lower-offset runs. sample_00229's −3.56 at seed 25, with the same offset, is unexplained.

**The option does two things, and most of the gain sits in cubes where one of them dominates.** On disputed sheets of at least 500 samples:

| | Sheets | Samples | Trusted, default → `orient_cube` | Fibre positive | Curvature agrees |
|---|---|---|---|---|---|
| `orient_cube` restores the sheet's own vote, which the default overrode | 67 | 1,187,348 (85%) | 1.56% → 6.86% | 58 of 64 (91%) | 46 of 51 (90%) |
| `orient_cube` overrides the sheet's own vote | 33 | 207,531 (15%) | 4.81% → 3.09% | 14 of 19 (74%) | 22 of 24 (92%) |
| For comparison, undisputed sheets | 681 | 11,913,234 | 6.69% → 6.54% | 556 of 633 (88%) | 381 of 481 (79%) |

- The 44 cubes where restoring dominates gain +13.94 on consistency, 84% of the total. The 7 where overriding dominates gain −0.01, and both consistency losses are among them. The other 49 have no disputed sheet of that size and gain +2.33, 16% of the total, including the largest gain, sample_00114 (+34.41). Under drift the default reverses 13.47% of the points on overridden sheets, against 81.39% on restored ones, so there is less to gain on them. This doesn't say their signs are wrong.
- The evidence on the overrides is mixed. Trust uses CT polarity, so it isn't independent of the vote. Fibre counts sheets with at least 300 samples of data. Curvature, whether a sheet's normals point toward the centre of its own fitted curve, is a judge added here and not validated: it counts only sheets with a clear fit, and agrees with the shared sign on just 79% of undisputed sheets.

**The two consistency losses differ.**
- sample_00469 loses on three disputed sheets whose strong votes the option overrides (75.3 → 56.4, 79.4 → 64.2 and 79.0 → 53.2). On its disputed points, the CT step across the final band has the expected sign 70.5% of the time under `orient_cube` and 89.9% under the default, a reading tied to the vote. Fibre is neutral there (−0.009, t 0.9).
- sample_00397 loses most on undisputed sheets (81.0 → 78.4, 78.4 → 75.8 and 77.9 → 75.8), so other sheets' signs appear to reach them. The snapper's learned edge profile carrying that is an inference, not measured.

**One real cube appears to have a fold, which one orientation per cube couldn't fit and the 70% guard doesn't see.** In sample_00198, two stacks of sheets lie 188 voxels apart with what looks like a fold between them. On one stack (80,197 of 188,642 samples, 43%), fibre (−0.156, t −5.9) and curvature (t +7.5) both read reversed. On the other, both read right (fibre +0.22, t 7.3; curvature t −12.5). The cube's axis share is 0.83 at its lowest. On the first stack only one sheet (20,800 samples) is disputed, so the default gives the rest the same sign. Of the 40 cubes with stacks at least 60 voxels apart, it is the only one where they disagree on both judges. The reading rests on those judges and one look at three slices, with no expert and no ground truth.

**The fibre guard's 41 of 46 is partly weak, and its 54 uncounted cubes mostly have few disputed samples on the clean labels, though they still gain.** Of those 54, 49 have under 0.5% of their samples disputed; the other 5 have 1.8–7.2% (sample_00032, 00484, 00492, 00506 and 00774). The 54 still hold 17% of the consistency gain and 50% of the drift gain (sample_00114 alone has +34.41 and +41.91), and the default reverses 3.62% of their points under drift. Using the tool's t over blocks of 40 voxels, 29 of the 41 positive cubes reach t ≥ 2, and sample_00411 is only +0.005. One negative cube is clear: sample_00680, −0.129, t −6.0, on one disputed sheet. Other evidence on that sheet goes the other way (own vote +8.8; trusted on 12.8% of samples against 0.7% under the default; curvature agrees with `orient_cube`), so it is unresolved.

**As point estimates, the gain is larger than on the development cubes, but neither difference is clear.** Consistency is +7.27 here against +5.06, and drift +2.25 against +1.51 (differences about −1.34 to +5.76 and −1.28 to +2.76). That doesn't rule out tuning: the default reverses more here (10.61% against 6.52%), and per point of reversal the gain is 0.685 (0.631 to 0.742) against 0.776. The development cubes are 20 contiguous ids, not a random draw.

## What this doesn't show

- **Accuracy.** There is no ground truth here. Nothing shows the labels are more correct, or that the signs changed on disputed sheets are right. That needs a blind review, which hasn't been done for `orient_cube` or the third arm.
- **Anything near the scroll centre.** The 70% guard fired on 0 of 400 inputs, so the real data never exercised it, and that case still rests on phantoms. The lowest axis share is 0.016 above the guard, on sample_00392.
- **The cubes not drawn.** Axis share, sign margin and the share turned against their vote have been computed only on the 20 development cubes and, in a review of the tool, on sample_00146 and sample_00147, not on the other 770. If the draw stands for all 892, 0 guard firings in 100 cubes still allows a rate up to about 3.0%, about 26 cubes, and 1 fold in 100 up to about 4.7%, about 42 (one-sided 95%). The one 384-cubed cube, sample_00147, wasn't drawn.
- **Independence of the measurements.** Band positions, normals, trusted sets, the fibre signature, orientation change and merger events come from one code base. Apart from the band measure on the 3 cubes above, they were repeated, not measured another way, and merger events were repeated only on those 3 cubes.
- **Ink, meshes or surfaces.** None was tested. Only labels use this option: the mesh tools refuse it (`scroll_label_snap/mesh.py`).

## What follows

- **For the labels.** The plan says "adopt only if", and that condition is met. Published labels stay as they are unless the owner adopts the option and re-snaps all 892 cubes. Going by these 100 cubes, that would reverse the normals on 10.70% of a cube's samples on average, and move the clean-label band by more than 1 voxel at 9.21% of measured points, pooled: 83.00% of disputed points and 0.78% of the others.
- **For any release.** It can say that reversal under drift fell from 10.61% to 0.013% and that consistency rose 7.27 points. It can't say the labels are 7% more accurate, that the drift guard confirms the changed signs, or that the third arm is safe, and what's added afterwards above should be labelled as such.
- **Before any re-snap,** computing axis share, sign margin and the share turned against their vote on the other 770 cubes would show how common the low-margin cubes are.
- **A variant,** such as keeping only the part that restores a sheet's own vote, is not what this test tested. It would need its own plan and a fresh draw, since these 100 have now been used.
- **The third arm** waits on its blind review.
- **The owner decides.**
