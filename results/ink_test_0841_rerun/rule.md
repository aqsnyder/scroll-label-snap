# PHerc. 0841 rerun with a corrected depth transfer: rule fixed before these maps exist

Status: exploratory, post-hoc. The plan's as-run numbers (results/ink_test_0841_plan.md, results/ink_test_0841.md) remain its outcome and are not replaced.
Disclosed in advance: on one pair (s214732 x w056_fr0b, seed 42) an earlier diagnosis already scored V0 and V1 on B1-B4. Every other new number below is unseen when this rule is written. The rule was written after that one pair was seen, so it can't be fully blind.

## Surfaces and pairs
A arm per target (s213127, s214732, s022814): A1 traced, A2 snapped dense.
B arm per pair: B1 winding as received, B2 drop-in grid, B3 snapped dense, B4 offset 0 smooth 0.1 dense.
Pairs: s213127 x w056_fr0b, s214732 x w055_fr0, s214732 x w056_fr0b, s022814 x w058_fr0, s022814 x w063_fr0b.
Layer direction: forward, as the plan selected (check direction.json).
Model and settings: as the plan, ink_9um hybrid_3d2d step-075000, villa deb1b69, CPU, --no-compile --overlap 0.5 --blend-mode hann --batch-size 4 --direction forward. Seeds 42 and 43.

## Transfer variants
V0: as run at 627d058 (refine 0, normal sigma 0). Seed 42 maps already exist; seed 43 needs inference only.
V1: fixed (refine 3, normal sigma 10). The frozen tool the Validate stage reports.
V2: refine only (refine 3, normal sigma 0), seed 42, B arms. A sensitivity check on the post-hoc smoothing.
V3: exact ray-mesh intersection written independently by the Validate stage, seed 42, B arms. A check that the answer doesn't depend on this one tool.
PUNCH: B2 as run with B3's as-run holes zeroed through all 21 layers, seeds 42 and 43, all five pairs.
FILL: B3 as-run depth kept wherever it exists, holes filled from B3 V1 depth, seeds 42 and 43, all five pairs. The converse of PUNCH: same depths, no false holes.

## Run order (stop at the cap in this order, and report what was dropped)
1 V0 control maps, 2 V1 seed 42, 3 V1 seed 43, 4 V0 seed 43, 5 PUNCH and FILL seed 42, 6 PUNCH and FILL seed 43, 7 V2, 8 V3.

## Scoring
tools/fine_labels.py score semantics: ROC AUC with ties at half; best F1; paired bootstrap over 128 x 128-pixel blocks, 1000 resamples, seed 0.
Primary pixels: the plan's rule applied inside each variant and seed: supervised, valid in every render of that comparison, non-zero in every map of it.
Secondary pixels: the owner's as-run scored pixels for that pair, intersected with validity in the variant, so that a change in the pixel set can't explain a change in AUC.
Contrasts at each seed: B2-B1, B3-B1, B4-B1, B3-B2 per pair; A2-A1 per target. PUNCH-B2 and FILL-B3(V0) per pair.
Plan reading per contrast: an interval that excludes 0 is a gain or a loss; otherwise no detectable difference.
Replicates: same sign with the interval excluding 0 at both seeds.
Pooled: mean weighted by scored pixels, with an interval from block resampling that shares draws between pairs on the same target.

## Readings, fixed now
R1, the artefact claim. SUPPORTED if, under V1 at both seeds, the pooled B3-B2 interval includes 0, at most one of the five pairs has a B3-B2 interval below 0, and pooled B3-B2 under V1 is at least 0.05 higher than under V0 at the same seed. PARTLY if pooled B3-B2 improves by at least 0.05 at both seeds but one of the other conditions fails. NOT SUPPORTED otherwise. If the seeds fall in different categories, report SEEDS DISAGREE.
R2, the mechanism. Among the pairs whose V0 B3-B2 interval excludes 0 downward at that seed: PUNCH-B2 must exclude 0 downward with a point estimate at least half the V0 B3-B2 gap, and FILL-B3(V0) must exclude 0 upward. SUPPORTED if both hold on all but at most one of those pairs at both seeds. Otherwise report the counts without a verdict word.
R3, snapped against as received under V1. Report every B2-B1, B3-B1, B4-B1 per pair and pooled, at both seeds, with the plan reading and whether it replicates. The summary sentence for each output is one of: loss at both seeds, gain at both seeds, no detectable pooled difference at both seeds, seeds disagree.
R4, the A arm. Report A2-A1 under V1 next to V0 at each seed. Flag any point estimate that moves by more than 0.02.
R5, smoothing sensitivity. If the R1 category at seed 42 differs between V1 and V2, say the R1 result depends on the post-hoc smoothing.
R6, tool independence. If the R1 category at seed 42 differs between V1 and V3, or any pair's B3-B2 differs by more than 0.05 between V1 and V3, say the result depends on the transfer method, and name the pairs.
R7, noise floor. Report seed-to-seed differences in each contrast, and any sham or small-move AUC differences available from the counterfactual job, next to the effects. Effects smaller than that floor are not to be described as effects, whatever their interval says.
No other reading is added afterwards without being labelled as added afterwards.

