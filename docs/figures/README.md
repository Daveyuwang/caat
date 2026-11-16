# Benchmark figures

Run `python scripts/figures.py` from the repository after generating the
judge replay. The script uses the saved tables and exports SVG and PDF
without network access. [Judge replay](../research/judge-study.md)

## Held-out risk and coverage

Each plot shows the risk at the four recorded review budgets on one
archive. The two panels use different graders on the same outputs. At a
budget boundary the plotted risk averages over ties. The source is
[`results/module_study/ranking.csv`](../../results/module_study/ranking.csv).

| Archive | Files |
| --- | --- |
| IFEval Qwen 2.5 7B | [SVG](risk-coverage-ifeval-qwen.svg) · [PDF](risk-coverage-ifeval-qwen.pdf) |
| IFEval Llama 3.1 8B | [SVG](risk-coverage-ifeval-llama.svg) · [PDF](risk-coverage-ifeval-llama.pdf) |
| MMLU-Pro Qwen 2.5 7B | [SVG](risk-coverage-mmlu-pro-qwen.svg) · [PDF](risk-coverage-mmlu-pro-qwen.pdf) |
| MMLU-Pro Llama 3.1 8B | [SVG](risk-coverage-mmlu-pro-llama.svg) · [PDF](risk-coverage-mmlu-pro-llama.pdf) |

## WildBench

**Grader shifts.** Each point gives the difference between the accepted-set
and whole-set grader shifts at the 50% coverage target. Bars show paired
95% bootstrap intervals, in percentage points. The signed direction is
second grader minus first. The symmetric logarithmic axis is linear
between −10 and 10 points. The source is `replay/judge_study/gate_cells.csv`.
[SVG](wildbench-grader-shifts.svg) · [PDF](wildbench-grader-shifts.pdf)

**Score AUROC.** Each point measures how well a score ranks passes for
one model and judge, using a judge score of 7 as the pass threshold.
The dashed line marks 0.5. Single-class combinations have no point.
The source is `replay/judge_study/auroc.csv`.
[SVG](wildbench-auroc.svg) · [PDF](wildbench-auroc.pdf)

The [policy example](policy-example.svg) is produced separately by
`caat policy --plot` using the command in the [README](../../README.md).

## Calibration

Run `python scripts/figures.py --calibration-only` to plot the bundled
held-out calibration bins. The diagonal indicates perfect agreement, and
larger marks indicate more rows in the bin.
[SVG](calibration-replay.svg) · [PDF](calibration-replay.pdf)

The [ticket reliability example](calibration-example.svg) comes from the
README's `caat evaluate` command with `--reliability-plot FILE.svg`.
