# Ink legibility review: set `8a9f1f2a81c9`

**Result: snapping hurts.** On the 60 comparisons, pooled over 2 reviewer(s), the snapped map was preferred 31 times and the original 75 times (snapped 29% (21%-39%) of decisive answers; two-sided exact sign test p = 2.3e-05). 14 answers were "about the same" and 0 "can't tell".

| Reviewer | Answered | Snapped preferred | Original preferred | Same | Can't tell | Snapped share of decisive (95% CI) | Sign test p | Catch items right | A share | Median time |
|---|---|---|---|---|---|---|---|---|---|---|
| reviewer 1 (the repository owner) | 78 | 18 | 37 | 5 | 0 | 33% (22%-46%) | 0.0145 | 8 of 8 | 40% | 1.9 s |
| reviewer 2 (hadn't worked on the tool) | 78 | 13 | 38 | 9 | 0 | 25% (16%-39%) | 0.000621 | 8 of 8 | 40% | 3.2 s |

Controls (random places, no ink condition):

- reviewer 1 (the repository owner): snapped 2, original 3, same 3, can't tell 2
- reviewer 2 (hadn't worked on the tool): snapped 2, original 6, same 2, can't tell 0

## Checking it

- The key was sealed before any answers came in. `review/sealed_key_8a9f1f2a81c9.enc` decrypts to `results/ink_review_key.json` byte for byte:

  ```
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in review/sealed_key_8a9f1f2a81c9.enc -pass file:results/ink_review_key_passphrase.txt
  ```

- The set id is the start of the key's SHA-256, taken as `json.dumps(key, sort_keys=True)`. The full hash is in the plan's run log (`results/ink_test_plan.md`).
- Re-score with:

  ```
  python tools/score_ink_review.py results/ink_review_key.json out.md results/ink_review_answers_reviewer1.json results/ink_review_answers_reviewer2.json
  ```

