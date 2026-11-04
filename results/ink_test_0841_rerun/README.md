# PHerc. 0841 rerun with a corrected depth transfer

Post hoc and exploratory. The plan's as-run numbers remain its outcome. The results are under "Later check" in [`../ink_test_0841.md`](../ink_test_0841.md).

- `rule.md`: the rule, written before these maps existed but after one pair had been scored.
- `table.csv`: every contrast by variant, seed and pixel set (bootstrap seed 0, as the rule fixes). It was scored with `tools/fine_labels.py`'s semantics, and again with separately written code, which agreed on every AUC and pixel count; intervals differ only by bootstrap draw.
- `table.md`: the same, compact. Its paths are the rerun's working folders, which aren't committed.
- `rule_readings.json`, `r7.json`: R1–R3 (its V2 and V3 rows give R5 and R6) and R7, applied to `table.csv`. R4's rows are in `table.md`.
- `exact_transfer.py`: the exact ray-mesh intersection (V3), kept here as the record of that check, not a maintained tool. `PIN` near its top names a local checkout of commit 627d058, whose `tools/` and `scroll_label_snap/` it imports; change it to run this.

Maps and renders aren't committed. V1 is `tools/depth_transfer.py --refine 3 --normal-sigma 10`, V2 is `--refine 3`, and V0 is the plan's run; the plan's settings and scorer do the rest, on its seed 42 checkpoint and the seed 43 checkpoint it didn't run. In `rule.md`, the "Validate stage" and "counterfactual job" are steps of the automated review ([`../review_2026_11.md`](../review_2026_11.md)).
