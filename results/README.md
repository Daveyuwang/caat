# Results

Saved tables from the benchmark replays. Each one can be regenerated on a
CPU with the commands in the linked document.

| Directory | Contents | Protocol |
| --- | --- | --- |
| [`baselines/`](baselines/) | Confidence rankings, threshold policies, and the paired IFEval grader replay on the Qwen HELM archive. `course.csv` replays the saved course predictions. | [CPU benchmark replay](../docs/benchmarks.md) |
| [`module_study/`](module_study/) | Score ablations, single-grader and two-grader threshold policies, and grader disagreements for two models on IFEval and MMLU-Pro. | [Module study](../docs/module-study.md) |

The [judge replay](../docs/judge-study.md) summarizes the WildBench gate
and its controls. Its per-model tables regenerate under `replay/judge_study/`.

## Course archive

The other directories hold saved predictions, annotations, metrics, and
plots from the original course project: `deberta_predictions/`,
`llm_annotations_7b/`, `trust_scorer/`, `offline_eval/`, and
`answer_token_pilot/`. `python scripts/compare_baselines.py --course` reads
`trust_scorer/*_lr_predictions.csv` as a historical baseline.
[Course results](../docs/archive/course-results.md) ·
[Course datasets](../docs/archive/course-datasets.md) ·
[Course project](../docs/archive/course-project.md)
