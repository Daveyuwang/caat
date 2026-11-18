# Benchmark figures

Run `python scripts/figures.py` after installing Matplotlib 3.10.7 and producing the WildBench judge replay. The script reads the saved tables and exports SVG and PDF. [Judge replay instructions](../research/judge-study.md)

## When can a labeler stop?

Each vertical segment spans the smallest and largest final error counts still possible after a block of WinoGrande grades. After 90 labels the range is 28 to 38. Even the lowest possible count exceeds the 27-error decision boundary. The diamond shows the separate 100-label verification. [SVG](winogrande-audit.svg) · [PDF](winogrande-audit.pdf) · [Frozen plan](../../results/winogrande/plan.json) · [Returned blocks](../../results/winogrande/receipts.csv) · [Full draw](../../results/winogrande/full-result.json)

## Held-out error at four coverage targets

Read across a row to see how a score trades coverage for accepted error. Compare the same cell across graders to see the effect of grading the same outputs differently. Each cell gives the percentage of accepted outputs judged wrong, averaging over ties at the budget boundary. [Source table with achieved coverage and counts](../../results/module_study/ranking.csv)

| Held-out archive | Vector figures |
| --- | --- |
| IFEval, Qwen 2.5 7B | [SVG](risk-coverage-ifeval-qwen.svg) · [PDF](risk-coverage-ifeval-qwen.pdf) |
| IFEval, Llama 3.1 8B | [SVG](risk-coverage-ifeval-llama.svg) · [PDF](risk-coverage-ifeval-llama.pdf) |
| MMLU-Pro, Qwen 2.5 7B | [SVG](risk-coverage-mmlu-pro-qwen.svg) · [PDF](risk-coverage-mmlu-pro-qwen.pdf) |
| MMLU-Pro, Llama 3.1 8B | [SVG](risk-coverage-mmlu-pro-llama.svg) · [PDF](risk-coverage-mmlu-pro-llama.pdf) |

The CLI policy plot gives the number of errors and accepted rows for each grader at one cutoff. The dashed line marks the target. Generate it with the README’s `caat policy --plot` example. [Policy plot](policy-example.svg) · [Example](../../README.md#try-a-review-policy)

## WildBench

The grader-shift plot compares two graders on the same outputs at a 50% coverage target. A point measures the second grader’s change in accepted error relative to its change on the whole batch. Whiskers give paired 95% bootstrap intervals. The scale is linear between −10 and 10 percentage points and logarithmic beyond that range. Parentheses give the paired row count. The data come from `replay/judge_study/gate_cells.csv`. [SVG](wildbench-grader-shifts.svg) · [PDF](wildbench-grader-shifts.pdf)

The AUROC matrices show how five scores rank passes for each model under three judges. A judge score of 7 or higher counts as a pass. Blue cells exceed 0.5, brown cells fall below 0.5 and `n/a` marks an undefined AUROC. The data come from `replay/judge_study/auroc.csv`. [SVG](wildbench-auroc.svg) · [PDF](wildbench-auroc.pdf)

## Calibration

Compare each raw score (top) with its Platt calibration (bottom) on the same held-out archive. Each circle represents an occupied bin. Larger circles contain more rows. The diagonal marks perfect agreement. Rebuild these panels with `python scripts/figures.py --calibration-only`. [SVG](calibration-replay.svg) · [PDF](calibration-replay.pdf) · [Source bins](../../results/calibration/reliability.csv)

The ticket example shows one dot per occupied bin on equal probability and accuracy scales. Add `--reliability-plot FILE.svg` to the README’s evaluate command to generate it. [Ticket diagram](calibration-example.svg) · [Example](../../README.md#try-caat-on-five-predictions)
