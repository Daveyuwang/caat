# Qwen2.5-7B confidence benchmark

CAAT collected greedy and sampled answers from Qwen2.5-7B-Instruct on a
local RTX 5080. This benchmark measures accepted-set error at fixed
coverage budgets and records the model work needed for each score. Within
each run, all methods rank the same greedy answers. The original run uses
128 questions and five samples per question. The expanded run uses 1,000
questions and ten samples per question on each task.

The model-family benchmark reuses these 2,000 questions and adds a shared
holdout for score selection checks. [Three-family comparison](model-families.md)

## Inputs and grading

Each task samples questions without replacement with seed 42 after
sorting IDs. Both runs use the same pinned dataset splits. Each expanded
question set includes the original 128 questions.

| Task | Split and population | Grader |
| --- | --- | --- |
| TriviaQA short answer | Web no-context validation, 9,951 rows | Exact match after TriviaQA normalization against the published normalized aliases |
| MMLU-Pro multiple choice | Test, 12,032 rows | One valid option letter, allowing surrounding whitespace, parentheses, or a trailing `.` or `)`. Compare it with the published answer key. |

The model sees the question, the choices for multiple choice, and an
instruction to return a short answer or one option letter. Greedy and
sampled answers use the same prompt. Their token limits are 16 on TriviaQA
and 4 on MMLU-Pro. Samples use temperature 0.7, top-p 0.9, and seed 42.
The replay evaluates the fixed scores on these questions.
[TriviaQA evaluator](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py) ·
[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) ·
[Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28)

## Scores and baselines

Random acceptance is the chance baseline, and shorter answers give a
length-only control. Mean and summed token log probabilities give the
rankings of inverse perplexity and maximum sequence probability,
respectively. MMLU-Pro also includes the greedy option's answer-span
log probability. These scores reuse the greedy generation.
[LM-Polygraph benchmark](https://aclanthology.org/2025.tacl-1.11/)

P(True) reads the logits for two correctness choices in a separate forward
pass. Verbalized confidence parses a separate numeric self-report. Exact
and normalized sample agreement use the relative frequency of the modal
form among the sampled answers. The latter follows CAAT's response
normalization. These measures still score the original greedy answer.
[P(True)](https://arxiv.org/abs/2207.05221) · [CAAT score definitions](../guide/features.md)

Discrete semantic entropy uses the frequencies of sample clusters.
On TriviaQA, exact normalized matches share a cluster. For distinct
answers, the pinned DeBERTa NLI model checks entailment in both directions,
with the question supplied on both sides. Each sample joins the
first matching representative. Lexical entropy uses normalized matches
alone for a control at the same Qwen sampling cost. On MMLU-Pro, samples
share a cluster when they choose the same option. Entropy scores are
negated so higher scores mean more confidence. The NLI inputs use
questions and samples, and the graders use the dataset keys.
[Discrete semantic entropy](https://doi.org/10.1038/s41586-024-07421-0) ·
[Pinned NLI checkpoint](https://huggingface.co/cross-encoder/nli-deberta-v3-small/tree/fa2804872c3b4bd748f38c0185cc85775361e735)

## Expanded comparison

At 75% coverage, summed token log probability has the lowest observed
error on TriviaQA (43.60%). Semantic entropy has the lowest observed
error among the black-box scores in this replay (44.29%). Its paired
interval for the difference from mean token log probability spans
−2.67 to −0.13 percentage points. Mean token log probability yields
53.33% error on MMLU-Pro and ties the summed and answer-token scores.

| Score | TriviaQA error | TriviaQA delta vs mean (95% interval, pp) | MMLU-Pro error | MMLU-Pro delta vs mean (95% interval, pp) |
| --- | ---: | ---: | ---: | ---: |
| Mean token log probability | 45.73% | 0.00 | 53.33% | 0.00 |
| Summed token log probability | 43.60% | −2.13 [−3.33, −1.07] | 53.33% | 0.00 [0.00, 0.00] |
| P(True) | 46.53% | +0.80 [−0.93, +2.40] | 55.20% | +1.87 [−0.14, +4.29] |
| Exact sample agreement | 45.14% | −0.60 [−1.79, +0.61] | 54.72% | +1.39 [+0.61, +2.81] |
| Discrete semantic entropy | 44.29% | −1.45 [−2.67, −0.13] | 54.72% | +1.39 [+0.61, +2.81] |
| UQLM exact match | 45.25% | −0.48 [−1.66, +0.80] | 54.72% | +1.39 [+0.61, +2.81] |
| LM-Polygraph lexical similarity (ROUGE-L) | 44.59% | −1.14 [−2.40, +0.13] | 54.72% | +1.39 [+0.61, +2.81] |

The expanded run scored 434 of 1,000 short answers and 410 of 1,000
multiple-choice answers correct. The common score comparison uses all
1,000 TriviaQA answers and 999 MMLU-Pro answers. One MMLU-Pro output
lacked an answer span. The 75% target accepts 750 rows in both tasks.
Intervals pair 2,000 question resamples against mean token log probability
and use a per-comparison 95% level. The reference run's TriviaQA semantic
entropy interval spanned −5.35 to +1.33 percentage points.
[Built-in scores and full comparison](../../results/black_box_1k/comparison.csv) ·
[External score comparison](../../results/black_box_1k/external_comparison.csv)

UQLM 0.6.8 compared each saved greedy answer with its ten samples.
LM-Polygraph 0.7.0 measured mean ROUGE-L similarity between sample
pairs. The comparison orients both scores so higher values favor
acceptance. Both methods used the same saved generations and made zero
additional model calls. The join matched the question ID and a hash of
the greedy text and ordered samples before calling `caat scores`.
[UQLM scoring API](https://cvs-health.github.io/uqlm/latest/_notebooks/examples/black_box_demo.html) ·
[LM-Polygraph estimators](https://github.com/IINemo/lm-polygraph#overview-of-methods)

| Task | Batch size | Prompt evaluations | Model invocations | Collection wall time | Clustering wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| TriviaQA | 64 | 13,000 | 205 | 572.25 s | 112.54 s |
| MMLU-Pro | 16 | 13,000 | 814 | 914.69 s | 0.0043 s |

The expanded run used PyTorch 2.11.0+cu130 and Transformers 5.16.0.dev0
with batching at every collection stage. The 128-question reference
used serial generation and a different runtime. The runs describe score
rankings at their recorded settings. The expanded runs evaluate each fixed
score ranking and share the generated answers across methods.
[Operation costs](../../results/black_box_1k/costs.csv) ·
[Expanded run metadata](../../results/black_box_1k/run.json)

## 128-question reference results

The greedy exact matches are 46/128 on TriviaQA and 51/128 on MMLU-Pro.
All methods have scores for all 128 questions per task. At 75% coverage
each policy accepts 96 answers.

| Score | TriviaQA error at 75% | MMLU-Pro error at 75% |
| --- | ---: | ---: |
| Random selection | 64.1% | 60.2% |
| Mean token log probability | 55.2% | 54.2% |
| Summed token log probability | 54.2% | 54.2% |
| P(True) | 56.3% | 56.3% |
| Verbalized confidence | 56.8% | 62.2% |
| Exact sample agreement | 52.2% | 57.4% |
| Discrete semantic entropy | 53.5% | 57.4% |

On TriviaQA, exact sample agreement changes the 75%-coverage risk by
−2.97 percentage points versus mean token log probability. The paired
95% interval is −6.25 to 0.00 points. Summed token log probability has the
highest AUROC (0.885) on that task. TriviaQA lexical entropy has 52.5%
risk at 75% coverage with the same five Qwen samples. On MMLU-Pro, the
answer-span and token baselines coincide because the greedy outputs
contain one token.

The comparison holds the evaluated questions fixed for all methods on a
task. `scored_rows` records score availability before this match, and
`matched_rows` records the common denominator. Risk at 75% coverage uses a
fixed number of accepted rows and averages over ties at the boundary.
The paired intervals resample the same questions for the candidate and
mean log-probability baselines (2,000 replicates, seed 20261115).
Brier and ECE accompany the two explicit correctness probabilities.
The research verifier checks AUROC and Brier against scikit-learn and
recomputes the budgeted risks with NumPy.
[Score comparison](../../results/black_box/comparison.csv)

One Qwen call is one unbatched generation or one P(True) forward pass.
Timings are wall-clock seconds on the recorded workstation. The RTX 5080
uses bfloat16 with a 13 GiB weight limit and CPU offload. Model loading
and warmup are reported separately. Exact agreement, normalized agreement,
lexical entropy, and semantic entropy reuse the same five samples. The
CPU NLI column counts one batched forward per question with distinct
answers and records the number of directional pairs in those batches.

| Operation | Qwen calls per task | TriviaQA wall time | MMLU-Pro wall time |
| --- | ---: | ---: | ---: |
| Greedy answer | 128 | 128.33 s | 54.48 s |
| P(True) | 128 | 27.47 s | 28.60 s |
| Verbalized confidence | 128 | 90.78 s | 117.84 s |
| Five sampled answers | 640 | 655.62 s | 271.54 s |

TriviaQA clustering adds 12.62 seconds on CPU for 68 NLI forwards
covering 550 directional pairs. MMLU-Pro clusters by option letter.
[Operation costs](../../results/black_box/costs.csv) ·
[Run metadata](../../results/black_box/run.json)

## Replay and recollect

The checked-in score tables contain IDs, correctness outcomes, and scalar
scores. Recompute the comparison from them with the base installation.

```bash
python scripts/study_black_box.py replay --output replay/black_box/comparison.csv
python scripts/study_black_box.py replay --scores-dir results/black_box_1k --output replay/black_box_1k/replayed-comparison.csv
```

For a new collection, install the research dependencies and the
`models` extra with a CUDA-enabled PyTorch build. Cache the Qwen and
DeBERTa checkpoints at the revisions above. The preparation step fetches
the pinned Parquet files if needed. Generation and clustering read the
cached weights.

```bash
python -m pip install -r scripts/requirements.txt
python -m pip install '.[models]'
python scripts/study_black_box.py prepare
OMP_NUM_THREADS=4 python scripts/study_black_box.py collect
OMP_NUM_THREADS=4 python scripts/study_black_box.py cluster
python scripts/study_black_box.py summarize --results replay/black_box/new_results
```

Use `--work-dir DIR` before a subcommand to keep another run's inputs,
responses, and timings in its own directory.

To recollect the expanded run and produce its score tables, use the recorded
batch sizes and sampling budget.

```bash
python scripts/study_black_box.py --work-dir replay/black_box_1k prepare --size 1000
python scripts/study_black_box.py --work-dir replay/black_box_1k collect --dataset triviaqa --batch-size 64 --samples 10
python scripts/study_black_box.py --work-dir replay/black_box_1k collect --dataset mmlu_pro --batch-size 16 --samples 10
OMP_NUM_THREADS=8 python scripts/study_black_box.py --work-dir replay/black_box_1k cluster
python scripts/study_black_box.py --work-dir replay/black_box_1k summarize --results replay/black_box_1k/summary
```

For an external score check, install `uqlm==0.6.8` and
`lm-polygraph==0.7.0` in their own environments. Both also need the local
CAAT package for the research script. Run each of the first two commands
in its corresponding environment. Repeat with the MMLU-Pro outputs for
the other task.

```bash
python scripts/study_external_scores.py uqlm replay/black_box_1k/triviaqa/responses.jsonl --output replay/external-scores/triviaqa-uqlm.csv
python scripts/study_external_scores.py lm-polygraph replay/black_box_1k/triviaqa/responses.jsonl --output replay/external-scores/triviaqa-polygraph.csv
python scripts/study_external_scores.py join replay/black_box_1k/triviaqa/responses.jsonl --builtins replay/black_box_1k/summary/triviaqa_scores.csv --uqlm replay/external-scores/triviaqa-uqlm.csv --lm-polygraph replay/external-scores/triviaqa-polygraph.csv --output replay/external-scores/triviaqa-joined.csv
caat scores replay/external-scores/triviaqa-joined.csv --correct-column correct --baseline mean_token_logprob --score-column mean_token_logprob --score-column uqlm_exact_match --score-column lm_polygraph_rouge_l --score-cost uqlm_exact_match=external --score-cost lm_polygraph_rouge_l=external --split all --bootstrap 2000 --seed 20261115
```
