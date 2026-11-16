# Evaluating review modules on archived outputs

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
[Qwen sources and dataset revisions](benchmarks.md#sources-and-grading) ·
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
[All and held-out disagreement counts](../results/module_study/disagreements.csv)

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
[Policy table and paired intervals](../results/module_study/policies.csv)

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
[Risk–coverage and Brier tables](../results/module_study/ranking.csv) ·
[Paired score differences](../results/module_study/ranking_deltas.csv)
