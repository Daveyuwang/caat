# Qwen2.5-7B confidence benchmark

CAAT collected one greedy answer and five sampled answers per question
from Qwen2.5-7B-Instruct on a local RTX 5080. All methods rank the same
greedy answers. The benchmark measures accepted-set error at a fixed
coverage budget and the extra calls each score needs.

## Inputs and grading

Each task uses 128 questions sampled without replacement with seed 42
after sorting IDs. The sample comes from the pinned dataset split.

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
form among the five samples. The latter follows CAAT's response
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

## Read the results

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
