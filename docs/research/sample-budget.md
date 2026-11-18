# Confidence quality at different sampling budgets

The replay uses one saved greedy answer and the first 2, 5, or 10 of its
saved samples. All budgets grade the same greedy answers for Qwen2.5-7B,
Phi-4-mini, and SmolLM2-1.7B on TriviaQA and MMLU-Pro. Each task reuses the
200 design and 400 holdout questions from the three-family benchmark.
[Runs and splits](model-families.md)

## Protocol

For zero additional generations, choose from mean token log probability,
summed token log probability, shorter response, and random ordering. For
each sample budget choose from exact and normalized sample agreement,
semantic entropy, UQLM exact match, and LM-Polygraph ROUGE-L. Use the
highest normalized AURC gain on design questions separately for each
model, task, and budget. Keep the chosen score fixed on holdout. Semantic
entropy uses the archived cluster labels for the sample prefix. The
external scorers run on each identical prefix and match by generation hash.
[Scorer versions and definitions](black-box-benchmark.md#expanded-comparison)

Accept 300 of 400 holdout answers per model using the chosen ranking.
Average over tied scores at the acceptance boundary and weight models
equally within each task. Use 2,000 paired bootstrap draws with seed
20261115, resampling the same questions across all models and budgets.
Report each budget's error difference from zero additional generations
and from the next smaller budget.

Each prefix of size `k` represents `k` extra prompt evaluations. When
semantic entropy is selected on TriviaQA, also count the directional NLI
pairs among distinct nonempty normalized samples. MMLU-Pro groups choices
directly. The archive records measured elapsed time for the original
ten-sample collection. The prefix replay uses those saved generations and
measures quality at the smaller budgets.

## Held-out results

Across the three model families, the zero-additional-generation score had
the lowest observed error on both tasks at 75% coverage. Each cell shows
the error rate and its difference from that baseline in percentage points.
The brackets give a paired 95% interval for the difference.

| Added prompt evaluations per answer | TriviaQA | MMLU-Pro |
| ---: | ---: | ---: |
| 0 | 57.44% | 79.00% |
| 2 | 60.23% · +2.78 [+1.86, +3.75] | 80.72% · +1.72 [+0.97, +2.77] |
| 5 | 58.59% · +1.15 [+0.46, +2.02] | 79.98% · +0.98 [+0.25, +2.00] |
| 10 | 59.16% · +1.72 [+0.86, +2.59] | 79.88% · +0.88 [+0.29, +1.95] |

Increasing from two to five samples lowered TriviaQA error by 1.64 points
[−2.37, −0.78]. The intervals for increasing from five to ten included
zero on both tasks. These comparisons use the recorded sample prefixes
and the three frozen model runs. [Task results](../../results/sample_budget/task_results.csv) ·
[Per-model selections](../../results/sample_budget/model_results.csv)

The original ten-sample stages each evaluated 10,000 prompts on the RTX
5080. The sampling stage ranged from 64.09 to 625.62 seconds across the
six model and task combinations, with 157 or 625 batched invocations.
The collection table records sampling-stage time separately from greedy
generation and confidence checks. [Recorded collection cost](../../results/sample_budget/collection_cost.csv)

## Replay

Use the bundled score table to reproduce the design selections and
holdout comparisons with the base CAAT installation.
[Prefix scores](../../results/sample_budget/prefix_scores.csv) ·
[Comparison script](../../scripts/study_sample_budget.py)

```bash
python -m pip install .
python scripts/study_sample_budget.py
```

To rebuild the score table from the saved generations, prepare the three
model-family archives, then export the fixed prefixes. Set `UQLM_PYTHON`
and `POLYGRAPH_PYTHON` to Python interpreters with CAAT and the pinned
external scorers installed. The join verifies each scorer's prefix digest
and reproduces the ten-sample columns from the original benchmark.
[Archive preparation](model-families.md#replay-from-saved-scores) ·
[Prefix builder](../../scripts/prepare_sample_budget.py)

```bash
python scripts/prepare_sample_budget.py prepare
for budget in 2 5 10; do
  prefix="replay/sample_budget/prefix-$budget"
  "$UQLM_PYTHON" scripts/study_external_scores.py uqlm "$prefix.jsonl" --output "$prefix-uqlm.csv"
  "$POLYGRAPH_PYTHON" scripts/study_external_scores.py lm-polygraph "$prefix.jsonl" --output "$prefix-lm-polygraph.csv"
done
python scripts/prepare_sample_budget.py join
python scripts/study_sample_budget.py --input replay/sample_budget/prefix_scores.csv
```
