# Benchmark results

The [benchmark guide](../docs/research/benchmarks.md) lists the archived inputs, graders, score definitions and replay commands. The [WildBench judge study](../docs/research/judge-study.md) writes its per-model results under `replay/judge_study/`.

The [course baseline](course/course.csv) compares scores for Qwen labels on IMDb, Jigsaw and FEVER. Run `python scripts/study_modules.py --course` to regenerate it from the included predictions. The Jigsaw annotation sample draws from the original training, development and test splits. [Sampling record](course/jigsaw-sampling.json) · [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) · [Jigsaw](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) · [FEVER](https://fever.ai/dataset/fever.html)
