# Pre-registered follow-up review: are snapped ink maps less legible, or only dimmer?

Written and pushed before the review set below was built. It follows the pre-registered w052 review ([`ink_test_plan.md`](ink_test_plan.md), [`ink_review.md`](ink_review.md)), whose outcome stands as recorded: two blind reviewers preferred the original map 75 times and the snapped map 31 times. Nothing here can change that outcome. This review asks why it came out that way. Changes after this point will be listed as amendments at the end, with the reason.

## Why

- **Snapped maps are dimmer.** On w052 and w035, at both seeds, mean ink probability falls by about 0.015, 98–100% of 1024-pixel tiles go down, and the strongest pixels fall most ([`review_2026_11.md`](review_2026_11.md)).
- **Reviewers followed how strong a panel looked**, even where the extra strength was only seed noise.
- **Whether strength explains the preference isn't established.** At equal strength, the snapped share is estimated at 43% (22–65%) going by the panel's standard deviation, and 53% (28–79%) going by its mean. Neither can be told apart from the observed 29% or from 50%.

So the first review fits two explanations. Either the snapped map is only dimmer, and the level change alone lost the comparisons on the fixed display, or its structure is less legible as well. Showing each pair at a matched level and contrast separates the two. If the preference goes away, it went with the level. If it stays, it is in the structure.

The dimming comes from moves finer than about 30–40 voxels ([`move_scale.md`](move_scale.md)). This review doesn't test that, and it doesn't test a smooth snap: [`smooth_snap_plan.md`](smooth_snap_plan.md) does.

## Maps

These are the two w052 ink maps of the first review:
- the original mesh and the snapped drop-in output, through `ink_9um` seed 42;
- files `D:\vesuvius\ink\w052\original_seed42.tif` and `snapped_seed42.tif`;
- valid pixels from the renders `original_flipped_c1.zarr` and `snapped_flipped_c1.zarr`, over the whole render (rows 0–7681, columns 0–9741).

Practice items come from w035's original map, as before.

## Items

Built by `tools/make_ink_review.py`, at the commit that adds this plan:

```
python tools/make_ink_review.py OUT_DIR KEY_JSON --original w052/original_seed42.tif --snapped w052/snapped_seed42.tif
    --renders w052/original_flipped_c1.zarr w052/snapped_flipped_c1.zarr
    --practice-map w035/original_seed42.tif --practice-renders w035/original_flipped_c1.zarr w035/snapped_flipped_c1.zarr
    --seed 1104 --display midway --n-comparisons 60 --n-controls 0 --fixed-arm 20 --level-only 20 --weak-catch 4 --weak-sigma 2
```

| Arm | Items | The two panels | Display | What it is for |
|---|---|---|---|---|
| Comparisons at matched level | 60 | original and snapped map, same place | matched | the decision |
| Comparisons, fixed scale | 20 | original and snapped map, same place | fixed, as in the first review | whether this session reproduces the first review |
| Level only | 20 | the original map, and the same crop given the snapped crop's histogram | fixed | what the level change costs with structure identical |
| Catch | 8, 4 from each map | one map, and the same crop blurred (Gaussian, σ = 4) | matched | that a reviewer can see a structure difference at matched level |
| Weak catch | 4, 2 from each map | as catch, σ = 2 | matched | a harder version of the same check |
| Practice | 4, with feedback | w035 sharp and blurred (σ = 4) | matched | not scored |

That is 112 scored items. There are no controls this time. The first review's random-place controls don't bear on this question, and leaving them out keeps the session short.

**Places.**
- **The draw.** Crop centres are drawn with seed 1104 from the first review's pool: pixels where the higher of the two maps is at least 0.5, with the top 1% as the fallback. Crops are 512 × 512 pixels, lie inside the map, are at least 90% valid in both renders, and don't overlap. The arms are drawn in this order: comparisons, catch, fixed scale, level only, weak catch.
- **If too few places are free,** the tool stops. The seed is then raised by 1 until it builds, with the counts unchanged, and the run log records each try.
- **The draw is new.** A new draw can still overlap crops of the first review. In the review's dry run on these maps at a throwaway seed (7), 27% of the comparison crops overlapped one by more than half.
- **Why not the first review's places** (`--reuse-places`)? Reviewers 1 and 2 have seen those crops. Pairing items would also gain little, since the first review's reviewers agreed on direction but not item by item (kappa 0.12).

**Balance and order.** In exactly half of each arm, the original map, or the sharp panel, is A. Items are shuffled.

**Display.**
- **Matched level.** Each crop goes through a monotone look-up table onto the midway of the two crops' histograms, taken over the crop's valid pixels. It then gets the first review's display scaling, (p − 0.25) / 0.5 clipped to 0–1. Both panels end up with almost the same histogram (ties are averaged within a grey level), so level and contrast match and only where the values sit differs.
  - The review's dry run (seed 7) measured the difference in mean grey level between the panels. It was 0.0003 at matched level against 0.029 on the fixed arm, on a 0–1 scale.
  - The difference in the 99th percentile was 0.003 against 0.110.
- **Fixed scale:** the first review's display.
- **Level only:** the original's structure at the snapped crop's level. In the dry run its panels differ in mean by −0.028, as the fixed-scale arm's do (−0.028).
- **Off-mesh pixels** are black in both panels.

**Page and question.** These are the first review's page, `review/ink/index.html`, and its question, unchanged:
- The question: "Which shows the ink more clearly: strokes or letter shapes that are easier to make out?"
- The answers: A, B, About the same, and Can't tell.

Reviewers aren't told what the arms are or which map is which, and reviewer 3 isn't told what the first review found.

## Reviewers

- **Reviewer 1** (the repository owner) and **reviewer 2** (someone who hadn't worked on the tool), as in the first review.
- **Reviewer 3:** someone who hasn't worked on the tool, and hasn't seen the first review's result or this repository's analysis.
- **More reviewers** can join only before the key is opened, and only if they meet reviewer 3's conditions.
- **The session.** Each reviewer works alone, in one sitting: 4 practice items, then the 112. At the first review's median times of 1.9 and 3.2 s per item, that is about 4–6 minutes.
- **The blinding question.** After the session, and before the key is opened, each reviewer is asked: "On the pairs that weren't plainly blurred, could you tell which side came from which mesh, and how?" The answer goes in the run log.
- **Catch items.** A reviewer with fewer than 6 of the 8 catch items right is reported, and their answers still count, as in the first review.

## Sealing and opening

The key is sealed as for the first review:
- Before any reviewer starts, its SHA-256 and an encrypted copy are committed (`openssl enc -aes-256-cbc -pbkdf2 -iter 200000`).
- The passphrase is committed when the key is opened.
- The set id is the first 12 hex digits of the SHA-256 of `json.dumps(key, sort_keys=True)`.
- Nobody who takes the review sees the key before answering.

The key is opened once reviewers 1–3 have answered. If reviewer 3 hasn't answered by 20 November 2026, it is opened with the answers in hand, and the write-up says so.

## Decision

`tools/score_ink_review.py`, at the commit that adds this plan.

- **Measure.** The decisive answers (A or B) on the 60 comparisons at matched level, pooled over reviewers. The measure is the share that prefer the snapped map, with a 95% interval clustered by item: a delta-method standard error of the ratio, since every reviewer answers every item. On the first review's answers, this interval is 20–39%.
- **Reading.**
  - **Less legible at matched level** if the interval lies below 50%.
  - **More legible at matched level** if it lies above 50%.
  - **Dimmer, not shown less legible** if it includes 50% and lies above 29%, the first review's share (31 of 106).
  - **Inconclusive** if it includes both 50% and 29%.
- **A smaller loss.** If the reading is "less legible" and the interval also lies above 29%, the write-up adds that the loss is smaller than on the fixed display.
- **What "dimmer, not shown less legible" doesn't show.** It doesn't show that the structure is as legible. If the true share at matched level were 40%, this design would give that reading 27–60% of the time with three reviewers (see Power).

## Reported alongside, not part of the decision

- **The fixed-scale arm:** its snapped share and interval.
  - If the interval includes 50%, the write-up says this session didn't reproduce the first review on its display.
  - If the share is still 29%, the chance of the interval lying below 50% is 0.83–0.89 with three reviewers deciding 88% of the time, as in the first review, and 0.69–0.76 with two.
- **The level-only arm:** the share of decisive answers that pick the copy at the snapped level, with its interval and its "about the same" answers.
- **Readings of the two together, fixed now.**
  - **Level only below 50% and the decision "dimmer, not shown less legible":** the level change alone can account for the first review's result.
  - **Level only including 50% and the decision "less legible":** the loss is in the structure, not the level.
  - **Any other combination** is reported as it comes out.
- **Catch and weak catch:** right answers per reviewer.
- **Per reviewer:** the first review's table, with reviewer 3 on their own row.
- **The first review's rule** (a pooled exact sign test) on the comparisons at matched level, for comparison only.
- **The answers to the blinding question.**

## Power

From `tools/ink_review_power.py`: 4000 simulations per design, seed 321, output in [`ink_review_matched/power.json`](ink_review_matched/power.json). In each design, a reviewer decides on a share of the items, and a decisive answer picks the snapped map at the arm's true share, with spread between items (sd on the logit scale) and between reviewers.

| Reviewers | Decisive | Item sd, reviewer sd | Share 29% gives "less legible" | Share 50% gives "dimmer, not shown less legible" | Share 40% gives "less legible" |
|---|---|---|---|---|---|
| 3 | 88% (as in the first review) | 0, 0 | 1.00 | 0.95 | 0.73 |
| 3 | 88% | 1, 0.3 | 0.95 | 0.87 | 0.47 |
| 3 | 50% | 0, 0 to 1, 0.3 | 0.85–0.99 | 0.85–0.93 | 0.32–0.50 |
| 2 | 50–88% | 0, 0 to 1, 0.3 | 0.75–0.99 | 0.78–0.94 | 0.26–0.55 |

- **The share isn't adjusted for the spread.** With an item sd of 1, a nominal 29% is about 32% pooled.
- **Reviewer spread.** The interval treats reviewers as fixed, so when they differ it is too narrow for reviewers in general. The readings are about the reviewers who took part.
- **No preference.** At a true share of 50%, the interval misses 50% about 5–6% of the time with no reviewer spread, split between the two directions. With a reviewer sd of 0.3, it misses 10–20% of the time.

## What this can't say

- It can't change the first review's outcome.
- **The matched display isn't how anyone views these maps.** Each pair is stretched to a common histogram, crop by crop. A "dimmer" reading would suggest that a display matched to each map, or a recalibrated model, recovers what was lost. That isn't tested here.
- **It covers one ink model, one segment and one seed's maps.** It says nothing about other scrolls or models.

## Reporting

All outcomes go into `results/ink_review_matched.md`, the README and the write-up, whichever way they come out. The answers, the key and its passphrase are committed when the key is opened.

## Run log

4 November 2026, on Lysander, with `run_ink_review_matched.ps1`, the tools pinned at b0831ba.

- **Built** at seed 1104 on the first try, in about 43 s.
  - 112 items: 60 comparisons at matched level, 20 on the fixed scale, 20 level only, 8 catch and 4 weak catch.
  - 4 practice items.
  - Set `b5713cdd210c`.
- **Sealed before any reviewer started.**
  - The key's SHA-256, taken as `json.dumps(key, sort_keys=True)`, is `b5713cdd210cde02a304e9a3a19d43079cf6c5959f14d4f852dbd00404204578`.
  - The key file's own SHA-256 is `eb475199ebe8503e57b1ba9c5faaf1bf58cfbbbc7ee0572dd85856481a605a61`.
  - An encrypted copy is `review/sealed_key_b5713cdd210c.enc` (AES-256-CBC, `openssl enc -pbkdf2 -iter 200000`).
- **The page** is `review/ink/index.html`, unchanged.
