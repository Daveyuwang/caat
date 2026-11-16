# Results

Saved tables from the benchmark replays. Each one can be regenerated on a
CPU with the commands in the linked document.

| Directory | Contents | Protocol |
| --- | --- | --- |
| [`course/`](course/) | Historical classifier predictions and the course baseline replay. | [Course project](../docs/archive/course-project.md) |
| [`module_study/`](module_study/) | Score ablations, single-grader and two-grader threshold policies, and grader disagreements for two models on IFEval and MMLU-Pro. | [Module study](../docs/research/benchmarks.md#module-study) |

The [judge replay](../docs/research/judge-study.md) summarizes the WildBench gate
and its controls. Its per-model tables regenerate under `replay/judge_study/`.

## Course archive

The course directory keeps the trust scorer's saved predictions for
IMDb, Jigsaw, and FEVER. `python scripts/study_modules.py --course`
replays them and writes `replay/baselines/course.csv`. The
other course outputs (DeBERTa predictions, Qwen annotations, trained models,
plots, and the answer-token pilot) are available in the
[original results tree](https://github.com/Daveyuwang/caat/tree/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/results).
[Course results](../docs/archive/course-results.md) ·
[Course datasets](../docs/archive/course-datasets.md) ·
[Course project](../docs/archive/course-project.md)
