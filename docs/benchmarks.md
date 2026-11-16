# CPU benchmark replay

The [module study](module-study.md) extends this replay to two model
families and compares score features and grader changes on frozen outputs.

The bundled samples contain 24 rows each. Run `python scripts/replay_benchmarks.py`
from an installed checkout to check the evaluator on instruction following,
multiple-choice reasoning, and open-ended factual recall.

| Sample | Score | Cutoff | Accepted | Accepted errors |
| --- | --- | ---: | ---: | ---: |
| [IFEval](../examples/ifeval_strict.csv) | Mean token log probability | −0.35 | 12 / 24 | 4 |
| [MMLU-Pro](../examples/mmlu_pro.csv) | Mean token log probability | −0.30 | 12 / 24 | 5 |
| [TriviaQA](../examples/triviaqa.csv) | Archived response score | 0.50 | 13 / 24 | 0 |

## Full HELM archives

The comparison uses all 541 IFEval and 1,000 MMLU-Pro responses in the
Qwen2.5-7B-Instruct-Turbo HELM Capabilities v1.0.0 runs. The MMLU-Pro archive
is a 1,000-question sample of the official test split. Both evaluations
freeze the model outputs. The script sorts by ID, shuffles with seed 42,
then splits about 20% for threshold design, 40% for calibration, and 40% for
held-out replay. Both IFEval graders use the same split.

```bash
python -m pip install .
python -m pip install -r scripts/requirements.txt
python scripts/prepare_ifeval.py --full
python scripts/prepare_mmlu_pro.py --full
python scripts/study_modules.py --baselines
```

Preparation downloads and caches the pinned public artifacts. Comparison
reuses the prepared CSVs. It writes `ranking.csv`,
`policies.csv`, and `paired_ifeval.csv` under `replay/baselines/`. Omit
`--full` to regenerate the 24-row samples.

[`examples/ifeval_graders.csv`](../examples/ifeval_graders.csv) carries the
`id`, `mean_token_logprob`, `strict_pass`, and `helm_strict_pass` columns of
the full Qwen IFEval file, `replay/ifeval/predictions_full.csv`, for the
[`caat policy` example](../README.md#choose-and-check-a-cutoff-in-one-command).

### Confidence rankings

The two log-probability scores implement the rankings used by LM-Polygraph's
Maximum Sequence Probability and Perplexity estimators using the archived
token log probabilities. Summing the log probabilities preserves the ranking
by sequence probability. Averaging them preserves the ranking by increasing
confidence (decreasing perplexity). Random accepts a uniform subset of the
same size, with expected risk equal to the full-set error rate. Accept-all
is the shared endpoint at 100% coverage.
[LM-Polygraph benchmark](https://aclanthology.org/2025.tacl-1.11/) ·
[Sequence probability](https://lm-polygraph.readthedocs.io/en/v0.4.0/lm_polygraph.estimators.html#lm-polygraph-estimators-max-probability-module) ·
[Perplexity source](https://lm-polygraph.readthedocs.io/en/latest/_modules/lm_polygraph/estimators/perplexity.html)

| Held-out evaluation | Accepted / total | Random | Sequence probability | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| IFEval, Google full pass | 163 / 217 | 25.35% | 25.15% | 24.54% |
| IFEval, HELM full pass | 163 / 217 | 34.56% | 34.97% | 33.13% |
| MMLU-Pro | 300 / 400 | 47.25% | 45.00% | 44.33% |

The table reports accepted error at approximately 75% coverage. The full
table uses the same budget for each score at 25%, 50%, 75%, and 100%,
rounding accepted counts up. Ties at the budget boundary use the expected
error under uniform selection. Each deployed score threshold still accepts
or defers the whole tie group. On MMLU-Pro at 50% coverage, sequence
probability has 39% risk and perplexity has 41%.
The `--baselines` replay writes these values to `ranking.csv`.

### Threshold selection

The design subset fixes up to ten candidate thresholds at the 10% through
100% acceptance quantiles. The empirical rule chooses the most permissive
threshold whose calibration error is at most the target. The Learn then Test
rule uses a one-sided binomial test on accepted calibration examples and
applies Bonferroni correction at family-wise level 0.05 for each benchmark,
score, and risk target. Both rules apply the selected cutoffs unchanged on
the held-out rows. A rule with no eligible threshold accepts zero rows.
[Learn then Test, §3.2](https://arxiv.org/pdf/2110.01052)

Learn then Test found no eligible threshold at either the 20% or the 30%
risk target in this replay. The empirical MMLU-Pro sequence-probability rule
at a 30% target accepted 139 of 400 held-out rows with 47 errors (33.81%).
These are retrospective within-archive splits. Any risk guarantee for
future data depends on the sampling assumptions and grader used.
The `--baselines` replay writes these values to `policies.csv`.

For IFEval, the two full-pass outcomes differ on 40 of 541 instances. The
Google-based empirical rule at the 30% target chose a mean token log
probability cutoff of −0.43964836263206303. Holding the cutoff and the 144
accepted responses fixed gives 33 errors (22.92%) under the Google regrade
and 45 errors (31.25%) under the published HELM grades. The comparison
measures sensitivity to the two grading pipelines. All 40 disagreements
occurred among the 95 prompts with an English capitalization, lowercase,
or language check. The other 446 prompts agreed. This breakdown locates the
disagreement without attributing it to one implementation detail.
The `--baselines` replay writes these values to `paired_ifeval.csv`.

The risk–coverage counts were cross-checked against CalibRoute 0.4.0 at all
1,671 held-out threshold points, using a rank-preserving map for its
`[0, 1]` input range. Direct masks independently checked the matched budgets.
SciPy's binomial CDF and direct acceptance counts reproduced the 24 policy
rows. Recomputed MMLU-Pro grades matched all 1,000 published grades.
[CalibRoute release](https://pypi.org/project/calibroute-ai/0.4.0/)

## Sources and grading

- **IFEval.** Match each archived key, stripped prompt, and instruction
  arguments to Google's dataset at the revision used by HELM. The local
  Google grader at `e6890f8` records a pass when every instruction passes.
  Python and language detection use seed 0. For the published HELM result,
  a response passes when `ifeval_strict_accuracy` is exactly 1, meaning
  full instruction credit. [HELM completions](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/ifeval:model=qwen_qwen2.5-7b-instruct-turbo/scenario_state.json) ·
  [HELM grades](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/ifeval:model=qwen_qwen2.5-7b-instruct-turbo/per_instance_stats.json) ·
  [Dataset revision](https://huggingface.co/datasets/google/IFEval/tree/966cd89545d6b6acfd7638bc708b98261ca58e84) ·
  [Google grader](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval) ·
  [HELM metric source](https://github.com/stanford-crfm/helm/blob/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/metrics/ifeval_metrics.py)
- **MMLU-Pro.** Verify each question, ordered options, and answer key against
  TIGER-Lab's test split at `3373e0b`. Use HELM's chain-of-thought extraction
  patterns to grade the saved choice. An unparsed choice counts as an error.
  [HELM completions](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false,model=qwen_qwen2.5-7b-instruct-turbo/scenario_state.json) ·
  [HELM grades](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false,model=qwen_qwen2.5-7b-instruct-turbo/per_instance_stats.json) ·
  [Test split](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/3373e0b32277875b8db2aa555a333b78a08477ea/data) ·
  [Answer extraction](https://github.com/stanford-crfm/helm/blob/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/metrics/gpqa_chain_of_thought_metric.py)
- **TriviaQA.** Match the first 24 questions from Xia's Llama 8B DPO
  archive to unique questions in the official Web validation split without
  context. Check the archived target against the official aliases and grade
  each response by normalized exact match. The saved `response_prob` is a
  ranking score. Run `python scripts/prepare_triviaqa.py` with PyArrow to
  regenerate the sample. [Saved responses and scores](https://huggingface.co/datasets/yuxixia/triviaqa_llama8b-dpo_test/tree/c73c578fd7302bec3c4ecc69d75691cced11aa4a) ·
  [Validation split](https://huggingface.co/datasets/mandarjoshi/trivia_qa/blob/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/rc.web.nocontext/validation-00000-of-00001.parquet) ·
  [Official evaluator](https://github.com/mandarjoshi90/triviaqa/blob/51b53595131828a00a1fed09cca9c87f512090e2/evaluation/triviaqa_evaluation.py)
