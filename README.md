# Confidence-Aware Annotation Toolkit

[![CI](https://github.com/Daveyuwang/caat/actions/workflows/ci.yml/badge.svg)](https://github.com/Daveyuwang/caat/actions/workflows/ci.yml)

CAAT scores, calibrates, evaluates, and triages saved model predictions.
Compare confidence signals, measure correctness probabilities, and choose
which outputs to accept automatically or send for review. The analysis
commands run on a CPU with Python 3.10 or later.

```bash
git clone https://github.com/Daveyuwang/caat.git
cd caat
python -m pip install .
caat evaluate examples/ticket_triage.csv --id-column ticket_id --prediction-column model_route --reference-column expected_route --score-column confidence --group-column inbox --threshold 0.75 --probability --output-dir replay/tickets
```

The example accepts 3 of 5 tickets and makes 1 error. CAAT writes `metrics.json`
with overall and per-group counts, AUROC, area under the risk–coverage curve
(AURC), and normalized AURC gain. It writes `risk_coverage.csv` with the
results at each distinct score threshold. A normalized gain of 0 matches
random ordering and 1 matches ordering all correct predictions first.

## CSV input

Each row needs a unique `id` and a finite `score`. Larger scores indicate
greater confidence. Supply `prediction` and `reference` for exact label
matching, or use `--correct-column NAME` for a grader's 0/1 outcome. Map
different headers with the flags above. An optional `group` column gives
per-group metrics.

CAAT accepts every row with a score at or above `--threshold`. Use
`--probability` to report Brier score, ECE, ACE, and reliability bins for
probabilities of correctness.
The review commands print a short summary. Add `--json` for machine-readable
stdout or `--output FILE` to save the JSON report. The CLI and Python API
use the standard library. [CSV input](https://github.com/Daveyuwang/caat/blob/main/docs/guide/csv-format.md) · [Report fields](https://github.com/Daveyuwang/caat/blob/main/docs/guide/reports.md)

Use `caat calibrate fit` to save a temperature, Platt, isotonic, or histogram
map from labeled calibration rows. Use `caat calibrate apply` to add its
probabilities to a held-out CSV. Add `--html report.html` to `evaluate`,
`audit`, or `policy` for a self-contained report.
[Calibration examples](https://github.com/Daveyuwang/caat/blob/main/docs/guide/calibration.md)

![Reliability diagram for the ticket-triage example](https://github.com/Daveyuwang/caat/raw/main/docs/figures/calibration-example.svg)

## Choose and check a cutoff in one command

```bash
caat policy examples/ifeval_graders.csv --score-column mean_token_logprob --correct-column strict_pass --check-column helm_strict_pass --risk-target 0.4 --plot replay/policy.svg
```

`caat policy` splits one labeled CSV into design, calibration, and held-out
rows. It selects a cutoff on the calibration rows and applies it unchanged
to the held-out rows. The example holds 541 saved IFEval responses graded by
Google's strict grader and by HELM. The command selects
`-0.43964836263206303` under the Google grader. On the 217 held-out rows the
policy accepts 144 responses, with 33 errors (22.9%) under Google and 45
(31.2%) under HELM. Both meet the 40% target.

![Held-out IFEval risk for the Google and HELM graders](https://github.com/Daveyuwang/caat/raw/main/docs/figures/policy-example.svg)

A `--check-column` grader is reported on held-out rows and takes no part in
selection. Pass it as a second `--correct-column` to require the target
under both graders. The cutoff then rises to `-0.3327795407672605` and the
policy accepts 92 held-out responses, with 19 and 23 errors.

The summary reports AUROC for each grader on the calibration rows and says
when the score ranks no better than random. The split sorts IDs, shuffles
with `--seed` (default 42), and assigns about 20%, 40%, and 40% of the rows.
`--rule` and `--failure-rate` work as in `caat select`. Add `--json` for the
full report. The command returns 0 if the held-out rows meet the target
under every listed grader, 1 if no cutoff qualifies or a grader exceeds the
target, and 2 for invalid input.

## Compare graders on the same outputs

Run a frozen threshold against separate grader exports. CAAT matches rows
by ID and checks that each grader scored the same saved output.

```bash
caat audit examples/audit/predictions.csv --output-key-column output_key --grader-file reference=examples/audit/reference.csv --grader-file rubric=examples/audit/rubric.csv --threshold 0.75 --risk-target 0.4
```

The sample accepts three tickets. The reference grader counts one error,
and the rubric grader counts two. Use `--json` for the changed
decision, discordant IDs, and per-group counts. For each alternate grader,
`overall_risk_change` measures the shift on all outputs and
`excess_risk_change` subtracts that shift from the accepted-set shift.
Add `--bootstrap 2000 --seed 42` for paired 95% intervals at the frozen
cutoff. The sample exits 1 because the rubric grader exceeds the 40%
risk target. A policy with zero accepted rows also exits 1 if you supply
a target. Invalid input returns 2.

For separate grader files, include `id`, `output_sha256`, and a 0/1
`correct` column in each export. The prediction file supplies the same
IDs and output hashes plus the score. Use `--output-key-column` if your
pipeline already records another output identity. If the results are
already aligned in one CSV, repeat `--correct-column NAME` for each grader.
The first grader is the reference for directional differences.

If a prediction CSV contains multiple runs for the same IDs, select one
with `--prediction-filter COLUMN=VALUE`.
[Real routing example and CI setup](https://github.com/Daveyuwang/caat/blob/main/docs/guide/ci-audit.md)

## Select a threshold on calibration data

```bash
caat select examples/ifeval_strict.csv --score-column mean_token_logprob --correct-column strict_pass --thresholds -0.20 -0.35 -0.60 --risk-target 0.4 --rule empirical
```

The example selects `-0.35`, accepting 12 of 24 responses with four
errors on the calibration sample. Supply a fixed list with `--thresholds`
or use `--design DESIGN.csv` to derive candidates from score deciles on a
separate file. Repeat `--correct-column` to require the target under every
grader. The JSON includes per-candidate counts and `selected_threshold`.
The command returns 0 if it selects a cutoff, 1 if none qualifies, and 2
for invalid input.

For statistical risk control, use `--rule ltt-bonferroni` (default),
`--rule ltt-holm`, or `--rule ltt-fixed-sequence`. These rules assume
independent calibration rows from the target population and a candidate
grid fixed before calibration. Holm's rule tests candidates in order of
increasing p-value. [Holm's step-down procedure](https://www.jstor.org/stable/4615733)
`--failure-rate` defaults to 0.05. Fixed-sequence testing proceeds from
the strictest threshold and stops at the first failure. The empirical rule
has no statistical guarantee. Apply the selected cutoff unchanged to
held-out data with `caat evaluate` or `caat audit`.

## Score saved responses

```bash
caat features examples/token_scores.jsonl --output replay/token_scores.csv
```

The input supplies response text and token log probabilities. The CSV
contains one row per response, with scores you can use as `--score-column`
values. [JSONL format and available scores](https://github.com/Daveyuwang/caat/blob/main/docs/guide/features.md)

Use `caat scores` to compare candidate columns at matched coverage budgets
and pick a score on the design split before calibrating a threshold.
[Score comparison guide](https://github.com/Daveyuwang/caat/blob/main/docs/guide/scores.md)

Use `caat queue` to order saved predictions for review under a row budget.
It can cap the number of rows per group and compare error yield with a
random queue. [Queue guide](https://github.com/Daveyuwang/caat/blob/main/docs/guide/queue.md) ·
[Course replay](https://github.com/Daveyuwang/caat/blob/main/docs/research/benchmarks.md#review-queues)

## CPU benchmarks

```bash
python scripts/replay_benchmarks.py
```

The command replays the bundled 24-response samples for IFEval, MMLU-Pro, and
TriviaQA and writes `replay/benchmarks/summary.csv`. The larger HELM comparison
evaluates sequence probability and perplexity rankings and tests empirical
threshold selection against Learn then Test.
[Sources, results, and reproduction](https://github.com/Daveyuwang/caat/blob/main/docs/research/benchmarks.md) ·
[WildBench and Omni-MATH grader replay](https://github.com/Daveyuwang/caat/blob/main/docs/research/judge-study.md)

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

The tests include the bundled benchmark replay. CI runs them on Python 3.10
through 3.14 on Linux and on 3.10 and 3.14 on Windows, runs the README
examples, and builds and installs the wheel.

[Evaluation method and related work](https://github.com/Daveyuwang/caat/blob/main/docs/research/method.md)

`caat plugins` lists built-in and installed components. The optional
Transformers backend collects model outputs for later scoring and replay.
[Plugin guide](https://github.com/Daveyuwang/caat/blob/main/docs/guide/plugins.md) ·
[Local model collection](https://github.com/Daveyuwang/caat/blob/main/docs/guide/collect.md)

[Contributing](https://github.com/Daveyuwang/caat/blob/main/CONTRIBUTING.md) ·
[Roadmap](https://github.com/Daveyuwang/caat/blob/main/docs/ROADMAP.md) ·
[Changelog](https://github.com/Daveyuwang/caat/blob/main/CHANGELOG.md)

## License

MIT. See [LICENSE](https://github.com/Daveyuwang/caat/blob/main/LICENSE).
