# Current binary findings

Version: 16 October 2026. Primary score is semantic agreement AND short-answer format compliance. These are completed descriptive evaluations; no new training or human audit is claimed by this release.

## Original benchmark

Four models share 1,500 questions and 1,065 original image groups, with 441 condition entries/model. Full source counts are in `results/binary_v1/`. The strict41 population passes clean and all 40 singles under the given model and axis.

| Model | Primary qualified | Primary joint-affected | Failure events / qualified joint conditions |
|---|---:|---:|---:|
| BLIP-2 |145|89|608 /58000|
| LLaVA |550|117|660 /220000|
| Qwen2.5-VL |540|155|1111 /216000|
| Qwen3-VL |658|170|1098 /263200|

The 25 maximum corners miss 41/42/63/63 affected questions, respectively; the 100 diagonals miss 24/19/34/37. A coordinate in the omission map counts distinct questions missed by the chosen slice and failing at that coordinate across at least one pair; a question can occur at several coordinates. These cells cannot be summed as distinct questions.

At equal joint-query budget 25, maximum-first discovers 107 Qwen3-VL questions versus uniform expectation 49.74; budget 100 gives 144 versus 111.50. Uniform is exact sampling-without-replacement expectation on 400 observed cells, not a confidence interval or learned generalization result. Curves display percentages of full-grid affected questions, with model denominators 89/117/155/170. Qualification cost of 41 queries is excluded from joint budgets.

Reference Qwen2.5-VL all-441 success fixes 385 questions. Original Qwen3-VL preserves 307 and first fails 10 at clean, 32 at single and 36 at joint. A different common strict41 population contains 445 questions: 65 affected questions become joint-successful, 36 previously successful questions acquire joint failures, 37 remain affected and 307 stay successful. Condition-level repairs 593 and new failures 324 are not question counts. Comparisons change checkpoints/architectures/runtimes together and are not isolated architectural causal effects.

## Semantic-only sensitivity

| Model | Semantic qualified | Joint-affected | Failure events | Diagonal missed | Corner missed |
|---|---:|---:|---:|---:|---:|
| BLIP-2 |178|110|830|29|44|
| LLaVA |554|118|656|20|43|
| Qwen2.5-VL |571|170|1134|39|73|
| Qwen3-VL |659|170|1080|37|63|

Each event denominator is 400×that row's qualification count. Fixed primary task qualification instead leaves semantic-affected counts 85/145, 117/550, 152/540 and 170/658. The original Qwen2.5→Qwen3 primary 385-question success cohort still preserves 307 semantically. Recomputing the reference all-441 success cohort semantically gives 401 questions and 315 preserved, with first failures at clean (11), single (37) or joint (38). These are different populations.

Thus additional joint failures, slice omissions and success loss are not removed by dropping format. This is same-label sensitivity, not independent Judge validation or an answerability audit.

## Completed 200-image adaptation development diagnostic

Three frozen endpoints share 200 distinct original images, one question/image, 105 conditions/image: 1 clean, 40 singles and four complete 4×4 pair grids. The 25 pairs each occur on 32 images. This scope uses known-benchmark development images with training/other-role/sealed-test image identities excluded; it is not an untouched final test. All 63,000 answers and binary grades are complete.

| Endpoint | Task clean/200 | Single/8000 | Joint/12800 | All/21000 |
|---|---:|---:|---:|---:|
| Base |134|5232|8181|13547|
| Ordinary matched protection50 |141|5627|8895|14663|
| Matched constrained50 |142|5635|8902|14679|

Relative to Base, ordinary repairs 2,140 and introduces 1,024 failures over 21,000 conditions; constrained repairs 2,117 and introduces 985. Constrained versus ordinary repairs 142 and introduces 126: 16 net correct, +0.076 percentage points, with original 200-image paired 95% interval [-0.281, +0.467] pp. The original bootstrap uses 2,000 replicates, seed 17, carrying all conditions of each of 200 images together. This interval was computed in the completed experiment; it is not recomputed by this compact package and does not establish an incremental constraint advantage.

Fixed Base corresponding-single task qualification gives 7,552 joint cells/145 images, including 119 gaps/45 images and 7,433 successes. Ordinary repairs 71/new 389, leaving 7,115 correct; constrained repairs 70/new 385, leaving 7,118. Constrained qualified accuracy falls 98.42%→94.25%, despite overall gains. Original primary net difference −4.171 pp has image-paired 95% interval [-7.220, −1.689] pp.

Matched constrained training accepted 50 slots without projection; 0 blocked and 1 zero-parameter-movement slot, with 5 post-update reference-CE increases. A local sampled gradient constraint is not a hard guarantee on all old successful answers; absence of activation is observed, not a proven cause of the similar methods.

Semantic-only all-condition correct counts are 14,138/14,744/14,765. Semantic Base corresponding-single eligibility gives 8,018 cells/150 images, 121 gaps and 7,897 prior successes. Ordinary repairs 74/new 473, constrained repairs 74/new 466, leaving 7,498/7,505 correct. Constrained accuracy falls 98.49%→93.60%. Retaining the exact original 7,552-cell task mask instead gives constrained semantic repairs 57/new 356. No semantic-only CI is computed; 21 extra correct outcomes versus ordinary is a descriptive 0.100 pp difference, not an established method benefit.

The root/condition analysis units, format axis and fixed reference masks must accompany any reuse of these summaries. A model's loss or fitting on selected training questions is not a claim about accuracy on unseen data.

