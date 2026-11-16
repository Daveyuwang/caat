# Grader changes under a frozen review policy

On WildBench, the 95% interval for excess grader shift included zero in
30 of 48 model and judge-pair comparisons at the primary cutoff. The plan
set the stopping point at more than half. The result ends the broader
study and gives users a reproducible benchmark for the paired audit.

## Archived inputs

The replay uses saved responses and grades from HELM Capabilities v1.15.0.
The release map identifies the source suite for each run. The generated
`replay/helm_judges/inventory.csv` records the exact suite and run name.
[HELM release run specs](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.15.0/run_specs.json) ·
[Release-to-suite map](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.15.0/runs_to_run_suites.json)

| Benchmark | Dataset revision and split | Saved outputs and grading |
| --- | --- | --- |
| WildBench v2 | `allenai/WildBench` at `7c05c1b4550282b2ed6a2e6ac5db069f1e07df5c`, test | All 68 release runs scanned. Sixteen had at least one response with three 1–10 grades and genuine token log probabilities. [HELM scenario](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/scenarios/wildbench_scenario.py) |
| Omni-MATH | `KbsdJames/Omni-MATH` at `40ba231d8f16e29ecd40e6407e2c8640145a8f62`, test | Qwen2.5-7B-Instruct-Turbo in suite v1.0.0, 999 complete responses with Boolean answer-equivalence judgments from all three judges. [HELM scenario](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/scenarios/omni_math_scenario.py) · [Saved grades and prompts](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/runs/v1.0.0/omni_math:model=qwen_qwen2.5-7b-instruct-turbo/display_predictions.json) |
| IFEval and MMLU-Pro | HELM Capabilities v1.0.0 test archives | Qwen2.5-7B-Instruct-Turbo and Llama 3.1 8B Instruct Turbo. IFEval uses the pinned Google strict regrade and HELM's full pass. MMLU-Pro uses HELM's answer parser and the pinned TIGER-Lab parser. [Dataset revisions, runs, and grader definitions](module-study.md#replay) |

At HELM commit `b1bc37e9223645ce91070e4fffd86dd46a04ca75`, `gpt` is
GPT-4o (`2024-05-13`), `llama` is Llama 3.1 405B Instruct Turbo
(Together), and `claude` is Claude 3.5 Sonnet (`20241022`). Both
annotators use temperature 0. The named templates are
`eval_template.score.v2.md` for WildBench v2 and
`gpt_evaluation_zero_shot_template.txt` for Omni-MATH. The public display
records include the rendered judge prompt and saved verdicts. Rendering
the pinned templates reproduced the prompt for every retained response
in both benchmarks (14,763 total). The deployment IDs above come from
the pinned source.
[WildBench annotator](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/annotation/wildbench_annotator.py) ·
[WildBench rubric](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/annotation/wildbench/eval_template.score.v2.md) ·
[Omni-MATH annotator](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/annotation/omni_math_annotator.py) ·
[Omni-MATH rubric](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/annotation/omni_math/gpt_evaluation_zero_shot_template.txt)

For each included run the script matches state and display rows one to one
by instance ID, verifies the saved response text, and hashes it with
SHA-256. Across all retained rows, the archived WildBench score agrees with
`(mean(raw judge scores) - 1) / 9`, and Omni-MATH accuracy agrees with the
mean of the three Boolean verdicts. Of the other 52 WildBench runs, 49
have only missing or placeholder log probabilities. Amazon Nova Lite,
Llama 3.1 405B Instruct Turbo, and Mixtral 8x22B Instruct have display
records but no `scenario_state.json` at their mapped suite in the public
bucket. Two OLMo 2 runs, 7B and 13B, each contribute two rows found by
scanning beyond the first response. The exclusion log lists the 3,000
display IDs from the runs with no public state.
[WildBench metrics](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/metrics/wildbench_metrics.py) ·
[Omni-MATH metrics](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/benchmark/metrics/omni_math_metrics.py)

CAAT checks the visible response by joining token strings, allowing
archived end markers, reversing ByteLevel encoding, and extracting the
final answer from the archived channel or thinking markers. The check
reconstructs 12,824 of 13,764 WildBench responses and all 999 Omni-MATH
responses. Of the remaining 940 WildBench responses, 916 contain replacement
characters in the saved token strings and 24 differ in other ways. The
public token records contain text and log probabilities. `token_alignment`
records the check for each row. The gate scores use the whole archived
token stream, which can include thinking tokens.
[ByteLevel mapping](https://github.com/openai/gpt-2/blob/e5c5054474f583d6d9499624649353995d63c70a/src/encoder.py) ·
[GPT-OSS channel format](https://github.com/openai/harmony) ·
[HELM token record](https://raw.githubusercontent.com/stanford-crfm/helm/b1bc37e9223645ce91070e4fffd86dd46a04ca75/src/helm/common/request.py)

## Protocol

For each grader pair `g → h`, the shift is the error rate under `h` minus
the error rate under `g` on the same accepted responses. Excess shift is the
accepted-set shift minus the whole-set shift. WildBench passes at score
7 or above, with 6 and 8 as sensitivity cutoffs. Omni-MATH passes when the
judge's equivalence verdict is true.

The primary policy accepts the top 50% by mean token log probability;
all ties at the boundary are accepted. The replay also uses 25% and 75%
targets and compares summed token log probability, the mean of the lowest
tenth of token log probabilities, negative token count, and negative mean
token log probability. Each 95% percentile interval uses 2,000 paired row
resamples and recomputes the cutoff in every resample. The two-sided
permutation check shuffles scores against paired outcomes 10,000 times at
the observed accepted count. Both use seed `20261115`.

The analysis is retrospective and restricted to the saved rows with the
required fields. The intervals describe row-level uncertainty relative to
the recorded judges. No human adjudication was performed.

## Gate result

| WildBench judge pair | 95% intervals containing zero | Cells |
| --- | ---: | ---: |
| Llama → GPT | 8 | 16 |
| Claude → GPT | 11 | 16 |
| Llama → Claude | 11 | 16 |
| **Total** | **30** | **48** |

Excluding the two OLMo runs with two rows each gives 24 of 42 intervals
containing zero. Limiting the gate to the eight runs with at least 90%
literal token-text matches gives 14 of 24. Recomputing the cutoff using
only rows whose responses could be reconstructed gives 31 of 48. All
three checks meet the same stopping rule.

For mean token log probability, the table gives the number of WildBench
intervals containing zero out of 48 at each pass score and coverage target.

| Judge score required to pass | 25% | 50% | 75% |
| ---: | ---: | ---: | ---: |
| 6 | 34 | 30 | 28 |
| 7 | 32 | 30 | 32 |
| 8 | 36 | 32 | 35 |

The AUROC gate passed. Negative mean token log probability
reached 0.60 under at least one judge on 10 of 16 models. Negative token
count reached it on seven. Mean, sum, and lowest-tenth token log
probability each reached it on four.

The three Omni-MATH judges disagreed on 17 of 999 outputs. Their
GPT/Llama/Claude pass rates were 29.53/29.53/28.93%. All shifts below are
in percentage points at the 50% coverage target.

| Archive | Grader change | Whole-set shift | Accepted-set shift | Excess shift [95% CI] |
| --- | --- | ---: | ---: | ---: |
| WildBench Qwen2.5 7B, 998 rows | Llama → GPT | 15.23 | 21.64 | 6.41 [3.61, 8.72] |
| WildBench Llama 3.1 8B, 995 rows | Llama → GPT | 20.10 | 24.30 | 4.20 [1.49, 6.90] |
| Omni-MATH Qwen2.5 7B | Llama → GPT | 0.00 | 0.40 | 0.40 [−0.20, 1.10] |
| Omni-MATH Qwen2.5 7B | Claude → GPT | −0.60 | −0.60 | 0.00 [−0.70, 0.80] |
| Omni-MATH Qwen2.5 7B | Llama → Claude | 0.60 | 1.00 | 0.40 [−0.30, 0.99] |
| IFEval Qwen2.5 7B | Google → HELM | 7.39 | 5.17 | −2.23 [−4.63, −0.015] |
| IFEval Llama 3.1 8B | Google → HELM | 10.54 | 11.07 | 0.53 [−1.92, 3.11] |
| MMLU-Pro Qwen2.5 7B | HELM → TIGER-Lab | −0.60 | −0.80 | −0.20 [−0.80, 0.40] |
| MMLU-Pro Llama 3.1 8B | HELM → TIGER-Lab | −3.00 | −3.80 | −0.80 [−1.89, 0.20] |

The two WildBench probe runs reproduce the original sample sizes and
GPT/Llama/Claude pass rates (71.14/86.37/80.56% for Qwen and
62.61/82.71/73.67% for Llama). The four deterministic controls have
small excess estimates, with the Qwen IFEval interval narrowly below
zero at 50% coverage. The same IFEval run had a disagreement AUROC of
0.412 (95% CI 0.328 to 0.499) in the initial probe.

## Reproduce the gate

Prepare the four full deterministic control CSVs using the installation
and preparation commands in the [module study](module-study.md#replay).
The following commands cache the HELM judge archives and run the CPU
analysis. The replay was verified with Python 3.12 and NumPy 2.5.5.
The installed `caat` commands only require the standard library.

```bash
python -m pip install -e .
python -m pip install -r scripts/requirements.txt
python scripts/prepare_helm_judges.py --scenario wildbench --release v1.15.0 --all-with-logprobs
python scripts/prepare_helm_judges.py --scenario omni_math --release v1.15.0 --model qwen_qwen2.5-7b-instruct-turbo
python scripts/study_judges.py --release v1.15.0 --deterministic-controls replay
```

`replay/helm_judges/` holds the per-run CSVs, source cache, inventory, and
row exclusion log. Under `replay/judge_study/`, `gate_cells.csv` gives
all 48 primary comparisons with AUROC for both judges in each pair.
`reconstructed_gate_cells.csv` reruns those comparisons on the verified
rows. `cells.csv` includes the other scores, cutoffs, and controls,
`auroc.csv` gives ranking results by judge, and `gate.json` records the
decision and sensitivity counts.

## Toolkit cross-checks

After preparing the module CSVs and the judge replay, run
`python scripts/verify_toolkit.py` in the research environment above.

- All 48 policy rows in the [module study](../results/module_study/policies.csv)
  reproduced their thresholds and held-out risk counts through `caat select`
  and `caat audit`. The original fixtures still give 12 accepted with 4
  errors, 12 with 5, and 13 with 0.
- Scikit-learn 1.7.2 matched the 743 archive AUROC entries, including
  undefined single-class cases. SciPy 1.18.1 matched the 480 candidate
  binomial tails checked on module study calibration data.
- MAPIE 1.4.1 matched the first-pass, second-fail sequence on Llama
  IFEval with the Google strict grader at a 42% target. Both selected
  `-0.14207283069189022`. Rank mapping put the saved scores in `[0, 1]`
  and preserved every accepted set. At a 39% target on Qwen MMLU-Pro,
  MAPIE rounded the second candidate's floating-point error count up from
  28 to 29 of 95. Its p-value was 0.05450. CAAT and SciPy gave 0.03435
  from the integer counts, so CAAT continued one threshold further.
  [MAPIE risk control](https://mapie.readthedocs.io/en/stable/api/risk-control/)
