# Pre-registered test: one orientation per cube

Written and committed before any cube in this test is run with the option. The tool has been run on no drawn cube. It has been run on development cubes sample_00018 and sample_00002, to show that it works and to check it against its verification, and in a review of the tool on sample_00146 and sample_00147, which are not in the draw.

## Why

`orient_normals` ends with a cross-sheet step that flips any sheet disagreeing with the samples around it. `Params.orient_cube` replaces that step: each sheet takes the side of the cube's dominant normal axis, the votes of all sheets together choose which way the axis points, sheets across the axis keep their vote, and every sheet keeps its vote if the axis holds under 70% of the normals' squared projection. On the 20 development cubes, over 3 drift seeds each, the share of samples whose orientation changes between a clean and a drifted run goes from 6.52% on average (34.8% at most) to 0.00% (0.01% at most). Consistency within 1 voxel goes from 81.73% to 86.79% (+5.06 points, 95% interval 2.07 to 8.35 by cube; 9 cubes better and 0 worse at a 0.5-point margin). Drift within 1 voxel, at points trusted under both orientations, goes from 89.99% to 91.50% (+1.51 points, interval 0.12 to 3.57; 5 cubes better, 1 worse, sample_00014 at -1.3). The fibre signature, which uses no label polarity and no CT step sign, favours the new orientation on all 20 cubes (+0.226 to +0.551), and on the samples where the two orientations disagree it is positive on 7 of 8 cubes (sample_00009 is the exception, at -0.11). These are development-cube evidence only: the step was chosen on these cubes, and no other cube has been run with it. All 20 are far from the scroll centre, so the case near the centre rests on phantoms: without the 70% guard the step gets 61% right with the axis at the cube's centre. The sign margin was about 0.90 at its lowest (0.8996 on sample_00018) and the axis share at least 0.797 on the 80 development inputs. The option turns sheets against their own vote on 1.89% of those samples (12.9% on sample_00014). Whether that holds on fresh cubes is what this test asks.

## Cubes

100 cubes, drawn once with `numpy.random.default_rng(20261103).choice(pool, 100, replace=False)` and sorted. `pool` is the sorted ids of the 892 cubes in `D:/vesuvius/all`, without:
- the 20 development cubes (sample_00001-00020);
- the held-out 100 (`results/heldout_selection.json`);
- the 200 cubes of the convention audit (the 20 development cubes and 180 random ones);
- the 40 cubes of the orientation prevalence check.

That removes 331 cubes (9 are in both of the last two lists) and leaves 561. `tests/test_eval_orient.py` repeats the draw from the lists below. 13 of the 100 are in the id stretch 161-280.

An earlier audit measured the topology of all 892 cubes without tuning anything. It and its verification read the published labels, confidence and flags, and the CT at merge sites. Nothing else in the repository or in this review's working files has looked at these ids, apart from the following, none of which tuned anything:
- All 892 are in the published full run (`results/full_run_summary.csv`), and an integrity check read their published labels (values, shape, label 2).
- sample_00180 and sample_00193 were among the topology audit's worst cubes. It tracked them voxel by voxel, re-solved them with remedy prototypes, and counted neighbouring sheets with opposite normals under the default step. That count is the only orientation measure on any of these ids.
- The same audit's work also read more of a few others: a sheet's split on sample_00198; a merge site rendered and looked at on sample_00229; repair and connectivity checks on sample_00070, 00114, 00254, 00469 and 00680; bright-run length at merge sites on sample_00032, 00125, 00192, 00194, 00229, 00340, 00481, 00527, 00577, 00599, 00648, 00668, 00762 and 00862.
- sample_00367 was re-snapped once in the speed audit, to compare bytes and time.

```text
drawn:
sample_00028 sample_00032 sample_00034 sample_00057 sample_00063 sample_00070 sample_00075 sample_00088
sample_00097 sample_00098 sample_00108 sample_00114 sample_00123 sample_00125 sample_00140 sample_00143
sample_00148 sample_00168 sample_00180 sample_00192 sample_00193 sample_00194 sample_00198 sample_00208
sample_00214 sample_00229 sample_00238 sample_00249 sample_00254 sample_00267 sample_00284 sample_00287
sample_00293 sample_00300 sample_00309 sample_00326 sample_00332 sample_00340 sample_00354 sample_00361
sample_00366 sample_00367 sample_00392 sample_00397 sample_00411 sample_00412 sample_00424 sample_00456
sample_00466 sample_00469 sample_00470 sample_00481 sample_00484 sample_00488 sample_00492 sample_00505
sample_00506 sample_00511 sample_00517 sample_00527 sample_00536 sample_00563 sample_00571 sample_00573
sample_00577 sample_00599 sample_00609 sample_00610 sample_00618 sample_00619 sample_00641 sample_00647
sample_00648 sample_00668 sample_00669 sample_00672 sample_00680 sample_00686 sample_00690 sample_00702
sample_00711 sample_00723 sample_00733 sample_00755 sample_00759 sample_00762 sample_00763 sample_00774
sample_00793 sample_00816 sample_00822 sample_00828 sample_00833 sample_00851 sample_00862 sample_00867
sample_00869 sample_00878 sample_00883 sample_00893

convention audit:
sample_00023 sample_00025 sample_00026 sample_00029 sample_00033 sample_00059 sample_00066 sample_00072
sample_00077 sample_00079 sample_00092 sample_00101 sample_00103 sample_00104 sample_00111 sample_00119
sample_00126 sample_00128 sample_00138 sample_00139 sample_00146 sample_00149 sample_00152 sample_00153
sample_00166 sample_00169 sample_00174 sample_00176 sample_00182 sample_00183 sample_00187 sample_00189
sample_00191 sample_00200 sample_00201 sample_00202 sample_00211 sample_00213 sample_00217 sample_00218
sample_00220 sample_00231 sample_00232 sample_00237 sample_00242 sample_00245 sample_00256 sample_00264
sample_00271 sample_00273 sample_00280 sample_00282 sample_00295 sample_00303 sample_00310 sample_00313
sample_00316 sample_00320 sample_00322 sample_00327 sample_00335 sample_00337 sample_00342 sample_00345
sample_00353 sample_00357 sample_00362 sample_00363 sample_00370 sample_00372 sample_00374 sample_00375
sample_00383 sample_00391 sample_00393 sample_00395 sample_00398 sample_00407 sample_00414 sample_00418
sample_00419 sample_00423 sample_00427 sample_00431 sample_00437 sample_00440 sample_00441 sample_00450
sample_00453 sample_00458 sample_00459 sample_00460 sample_00465 sample_00468 sample_00474 sample_00480
sample_00494 sample_00495 sample_00497 sample_00504 sample_00508 sample_00509 sample_00515 sample_00516
sample_00520 sample_00526 sample_00541 sample_00544 sample_00547 sample_00548 sample_00552 sample_00555
sample_00559 sample_00560 sample_00565 sample_00566 sample_00568 sample_00585 sample_00592 sample_00593
sample_00596 sample_00602 sample_00611 sample_00616 sample_00620 sample_00624 sample_00627 sample_00633
sample_00635 sample_00640 sample_00650 sample_00654 sample_00658 sample_00659 sample_00661 sample_00663
sample_00676 sample_00679 sample_00689 sample_00692 sample_00693 sample_00696 sample_00698 sample_00700
sample_00728 sample_00739 sample_00743 sample_00748 sample_00749 sample_00750 sample_00756 sample_00761
sample_00769 sample_00770 sample_00771 sample_00777 sample_00780 sample_00786 sample_00791 sample_00799
sample_00801 sample_00807 sample_00812 sample_00827 sample_00831 sample_00837 sample_00838 sample_00846
sample_00847 sample_00863 sample_00865 sample_00874 sample_00879 sample_00891 sample_00896 sample_00897
sample_00899 sample_00909 sample_00911 sample_00915

prevalence check:
sample_00079 sample_00101 sample_00120 sample_00130 sample_00178 sample_00191 sample_00202 sample_00207
sample_00210 sample_00213 sample_00215 sample_00233 sample_00253 sample_00288 sample_00308 sample_00311
sample_00359 sample_00416 sample_00446 sample_00513 sample_00544 sample_00561 sample_00576 sample_00607
sample_00611 sample_00612 sample_00620 sample_00642 sample_00645 sample_00666 sample_00679 sample_00695
sample_00706 sample_00718 sample_00747 sample_00805 sample_00820 sample_00840 sample_00854 sample_00871
```

## Arms

All at one commit containing this plan, with the other parameters at their defaults. The tool records the commit and this file's SHA-256 with each cube, and refuses to mix them in one output folder:
1. default;
2. `orient_cube=True`;
3. `orient_cube=True, keep_topology=True`.

## Run

`python tools/eval_orient_test.py D:/vesuvius/all OUT_DIR`, which runs each arm on a cube's clean labels and on three drifted versions of them. Cube k (0 to 99, in the order listed) uses drift seeds k, 100+k and 200+k, with the drift of `tools/eval_snap.py` (std 1.5, correlation 40). The drifted labels are built once, along the default arm's normals, so all arms start from the same input.

Every position is measured on one fixed set of points: the clean labels' samples at least 10 voxels from every face, along the same lines for all arms.

## Primary measure

Paired consistency: among those points where both bands exist, the share where the band from drifted labels is within 1 voxel of the band from clean labels, in the same arm. The share of points with no band to measure (`missing`) is reported per arm and does not count against it. Per cube, the mean over the three seeds, for `orient_cube` minus default, in points. The 95% interval is by resampling cubes (5,000 resamples, seed 0).

**Adopt only if the interval's lower end is above 0, and all the guards hold.**

## Guards

All are required to adopt:
- **Drift.** Within 1 voxel of the original label, on one fixed point set: the points trusted under both arms on the clean labels, where the band exists. For `orient_cube` minus default, the 95% interval's lower end must be above -1 point.
- **Orientation change.** The share of points whose nearest sample in the drifted run, within 6 voxels, has a reversed normal. For `orient_cube`, the mean across seeds and cubes must be at most 1%.
- **Fibre signature.** On the disputed samples, those where the two arms' normals disagree on the clean labels, under `orient_cube`'s normals. It uses no label polarity and no CT step sign, though the samples' positions and normal lines come from the labels, and it assumes the scroll axis is scan z. It must be positive in at least 75% of the cubes that have at least 1,000 disputed samples with fibre data. If no cube has that many, nothing independent supports the changed signs, and this guard is not met.

Reported per cube: axis share, sign margin, guard triggers (on how many of the four inputs the 70% guard fired), the share of samples turned against their own sheet's vote, and the numbers above. All of it is also reported for the id stretch 161-280 and for the other ids separately, since the topology audit found that the CT in that stretch rarely resolves gaps. The decision uses all 100 cubes.

**Otherwise it stays off, and that is reported.** If it passes, adopting would change published labels, so it would need a re-snap of all 892 cubes.

## The third arm

It passes if, on the clean labels' output, Betti numbers equal the original's at both connectivities and merger events are 0, on all 100 cubes (`tools/eval_topology.py`). That merger count is a rule, not a tracking of voxels, and can count a small sheet whose voxels end up nearest a neighbouring sheet even if the two stay separate. It gave 0 on the `keep_topology` bands of the 20 development cubes, with the default orientation and with `orient_cube`. Any event is reported with its sheets' sizes and the component counts. Also reported: drift on the same fixed set, consistency, and contested samples (samples that end within 3 voxels of a moved sample of another input sheet, at least 0.5 voxel closer than before), with how far its band differs from arm 2's at those samples and elsewhere.

It is not adopted from this test, whichever way it comes out, because the drift test barely scores the samples it changes. On the development cubes, of the 3,655 samples it moved more than 1 voxel from the plain warp, 21 were trusted. That needs a blind review at contested sites.

## Reporting

Results go in `results/orient_test.md` whichever way they come out, with the tool's `orient_test.json` and `orient_test_cubes.csv`.

## Amendments

Made before any drawn cube was run:
- The arms run at one recorded commit, not "the commit that adds this plan", because the tool and its tests changed after that commit.
- The primary measure and the drift guard are stated as the tool computes them, at points where both bands exist. The share of points missing is reported per arm. On sample_00018 at seed 17, the tool's reading gives the default 65.80% and "all those points" would give about 65.55%.
- The decision comes only from the list above: with any other list, the tool reports no decision.
- Corrected: the sign margin's floor (it read "at least 0.90", and sample_00018 has 0.89957), what the fibre signature uses, which cubes the tool has been run on, and the third arm's development result.
