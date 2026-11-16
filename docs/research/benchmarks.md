# CPU benchmark replay

The [module study](#module-study) extends this replay to two model
families and compares score features and grader changes on frozen outputs.

The bundled samples contain 24 rows each. Run `python scripts/replay_benchmarks.py`
from an installed checkout to check the evaluator on instruction following,
multiple-choice reasoning, and open-ended factual recall.

| Sample | Score | Cutoff | Accepted | Accepted errors |
| --- | --- | ---: | ---: | ---: |
| [IFEval](../../examples/ifeval_strict.csv) | Mean token log probability | −0.35 | 12 / 24 | 4 |
| [MMLU-Pro](../../examples/mmlu_pro.csv) | Mean token log probability | −0.30 | 12 / 24 | 5 |
| [TriviaQA](../../examples/triviaqa.csv) | Archived response score | 0.50 | 13 / 24 | 0 |

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

[`examples/ifeval_graders.csv`](../../examples/ifeval_graders.csv) carries the
`id`, `mean_token_logprob`, `strict_pass`, and `helm_strict_pass` columns of
the full Qwen IFEval file, `replay/ifeval/predictions_full.csv`, for the
[`caat policy` example](../../README.md#choose-and-check-a-cutoff-in-one-command).

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

## Module study

The study tests two choices that affect users of the CLI. A fixed acceptance
policy may pass under one grader and fail under another. A task-specific
score may prioritize review better than a generic confidence score, using
the same saved response.

## Replay

The input is the full HELM Capabilities v1.0.0 archive for Qwen2.5-7B-Instruct-Turbo
and Llama 3.1 8B Instruct Turbo on IFEval and MMLU-Pro. The models share the
same 541 IFEval prompts and the same 1,000 MMLU-Pro questions. The preparation
scripts verify prompts, correct choices or instructions, and output IDs.
Each prepared row carries a SHA-256 hash of the saved response text.
[Qwen sources and dataset revisions](#sources-and-grading) ·
[Llama IFEval run](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/ifeval:model=meta_llama-3.1-8b-instruct-turbo/scenario_state.json) ·
[Llama IFEval grades](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/ifeval:model=meta_llama-3.1-8b-instruct-turbo/per_instance_stats.json) ·
[Llama MMLU-Pro run](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false,model=meta_llama-3.1-8b-instruct-turbo/scenario_state.json) ·
[Llama MMLU-Pro grades](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false,model=meta_llama-3.1-8b-instruct-turbo/per_instance_stats.json)

```bash
python -m pip install .
python -m pip install -r scripts/requirements.txt
python scripts/prepare_ifeval.py --full
python scripts/prepare_ifeval.py --model llama --full
python scripts/prepare_mmlu_pro.py --full
python scripts/prepare_mmlu_pro.py --model llama --full
python scripts/study_modules.py
```

The run writes the aggregate tables and aligned held-out CSVs to
`replay/module_study/`. For example, reproduce the 30% Qwen IFEval audit
with the cutoff recorded in `policies.csv`.

```bash
caat audit replay/module_study/ifeval_qwen_heldout.csv --score-column mean_token_logprob --correct-column strict_pass --correct-column helm_strict_pass --threshold -0.43964836263206303 --risk-target 0.30
```

## Design score screen

Run `python scripts/study_features.py` after preparing the four complete
archives. For each archive, the script uses the original seed-42 design
split and its primary grader. It compares each score with mean token log
probability at 75% coverage. The matched-budget calculation averages over
ties at the boundary. A candidate enters the default registry if it lowers
risk in at least three of the four archives.

| Candidate | Direction for higher confidence | Design archives with lower risk |
| --- | --- | ---: |
| Token log probability population standard deviation | Less variation | 0 / 4 |
| Share of tokens with probability below 0.1 | Lower share | 1 / 4 |
| Mean log probability of the final tenth of tokens | Higher mean | 1 / 4 |
| Distinct adjacent word-pair ratio | Higher ratio | 0 / 4 |

The four candidate scores remain in the research script. The output at
`replay/feature_screen.csv` records each observed risk difference.

## Grader changes

IFEval compares Google's pinned strict grader with the published HELM
full-pass outcome. MMLU-Pro compares HELM answer extraction with the
TIGER-Lab local evaluator's extraction rules at commit `26d41d0`. For both
extractors an unparsed output counts as incorrect. The local evaluator's
separate random-guessing step would change the outcome without a change to
the response, so the paired comparison uses its extraction rules alone.
The recomputed HELM outcomes agree with all 2,000 published MMLU-Pro grades.
[TIGER-Lab answer extraction](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/26d41d02c9f6e1794e74a2740ed378cb0562faea/evaluate_from_local.py#L102-L125)

The archives show 40 and 57 grader disagreements on Qwen and Llama IFEval,
respectively. All 97 involve language or English letter-case checks.
MMLU-Pro has 8 disagreements on Qwen and 30 on Llama. The counts reflect
the behavior of these graders and the specified extraction rules.
[All and held-out disagreement counts](../../results/module_study/disagreements.csv)

For each archive, order IDs, shuffle with seed 42, and split about
20%/40%/40% into design, calibration, and held-out sets. The two model
archives for a task share the same question IDs per split. Fix the score to
mean token log probability. Choose up to ten cutoffs from design quantiles.
Evaluate at 20%, 30%, and 40% target risk, using empirical selection or
Bonferroni Learn then Test at 0.05 per archive and target. For two graders,
empirical selection requires both accepted error rates to meet the target.
Learn then Test uses the higher accepted error count in the one-sided
binomial test. Each policy applies its selected cutoff unchanged on the
held-out set. A missing cutoff corresponds to zero automatic acceptance.
[Learn then Test](https://arxiv.org/pdf/2110.01052)

| Archive | Accepted, primary empirical rule at 30% | Primary grader risk | Alternate grader risk |
| --- | ---: | ---: | ---: |
| IFEval, Qwen | 144 / 217 | 22.92% | 31.25% |
| IFEval, Llama | 18 / 217 | 27.78% | 44.44% |
| MMLU-Pro, Qwen | 88 / 400 | 31.82% | 28.41% |
| MMLU-Pro, Llama | 0 / 400 | Undefined | Undefined |

For Qwen IFEval, requiring both graders under the empirical 30% rule
reduced acceptance from 144 to 92. Held-out risk was 20.65% and 25.00%
under the two graders. At a 40% target, Learn then Test on both graders
also accepted 92, down from 144 for the primary grader alone. For Llama
IFEval at 40%, Learn then Test accepted 215 with the primary grader and
found no eligible threshold with both. These are coverage costs at the
available calibration sample sizes and candidate cutoffs. The empirical
two-grader rule can still exceed the target on held-out data, as it did
for Qwen MMLU-Pro at 30%.
[Policy table and paired intervals](../../results/module_study/policies.csv)

Independent masks reproduced all 48 policy rows. CalibRoute 0.4.0
reproduced the risk and coverage for every selected cutoff with nonzero
coverage. The installed `caat audit` CLI also reproduced the paired
counts on all four held-out archives from separate, reordered grader
exports.
[CalibRoute release](https://pypi.org/project/calibroute-ai/0.4.0/)

## Confidence scores

The comparison reuses the saved generation for every score. Raw features
include mean and sum of token log probabilities, mean of the lowest tenth
of token log probabilities, and the negative log of token count. The
fusion model fits an L2-regularized logistic regression on standardized
mean log probability, low-token log probability, and log token count
using design rows and the primary grader. A fitted mean-only logistic
model and the design-set base rate provide probability baselines.

For MMLU-Pro the answer-token score uses the archived log probability
of the token containing the explicitly marked option. Token text must
reconstruct the saved response up to the end of that response. Missing
or unaligned spans rank last. The presence-only control assigns a binary
score to whether the span is available. We also fit two fusion models
adding presence alone and presence plus the log probability. These
comparisons use zero additional model calls and generated tokens.

| Archive, graded by HELM | Mean token log probability | Answer present | Answer-token log probability | Fusion with answer token |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro, Qwen | 44.33% | 45.29% | 41.33% | 42.00% |
| MMLU-Pro, Llama | 53.33% | 50.15% | 46.67% | 46.00% |

The table reports error among 300 accepted responses per model. Answer
token scoring improved on presence alone by 3.95 and 3.48 percentage
points. The paired 95% bootstrap intervals for the change were
−6.03 to −1.08 and −4.99 to −1.26 percentage points, respectively.
Against mean token log probability, the reduction was clearer for Llama
(−6.67 points, interval −10.00 to −3.11) than for Qwen (−3.00 points,
interval −6.47 to 1.33). The ranking intervals resample held-out
questions and recompute the matched budget for both scores. Policy
intervals resample the accepted set at the frozen threshold. Both use 2,000
paired bootstrap replicates, seed 20261115, and percentile intervals.
The intervals are per comparison from a retrospective study.

The generic three-feature fusion gave mixed results on IFEval, and the
plain sequence score was stronger than that fusion for Llama IFEval at
75% coverage. Missing answer spans also differed by model (51/1,000
Qwen and 181/1,000 Llama). Keep the score configurable. Use the
answer-token score when a multiple-choice pipeline records aligned token
probabilities, and evaluate its presence rate as part of onboarding.
[Risk–coverage and Brier tables](../../results/module_study/ranking.csv) ·
[Paired score differences](../../results/module_study/ranking_deltas.csv)
