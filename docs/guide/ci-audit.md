# Audit a grader change on a routing export

This walkthrough replays a frozen review policy from the public CFPB ticket
routing project. It uses 750 ModernBERT predictions. The upstream policy
keeps the encoder's output when confidence is at least
`0.7980151891708375` and escalates the rest. The source project selected
that cutoff on a separate split for a 10% escalation budget.
[Upstream project](https://github.com/ryansloan/cfpb-routing/tree/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f) ·
[Predictions](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/followup/encoder_eval.csv) ·
[Frozen policy report](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/followup/stats.json)

## Reproduce the audit

Run from the CAAT checkout in a Python virtual environment.

```bash
python -m pip install .
python scripts/prepare_cfpb_audit.py
caat audit replay/cfpb/sources/encoder_eval.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --group-column product --output-key-column output_text --grader-file archived=replay/cfpb/archived.csv --grader-file regraded=replay/cfpb/regraded.csv --threshold 0.7980151891708375 --risk-target 0.10
```

Preparation downloads the pinned prediction export and policy report and
writes two grader CSVs. The archived grader uses the published exact-match
outcome. For this walkthrough, the candidate grader parses the same stored
score map and requires a unique highest-scoring label matching the same
reference. The candidate is a local rubric change for the audit. The 10%
accepted-error limit is an example CI target. The upstream 10% escalation
budget determines the threshold.

The prediction export has four tiers with overlapping complaint IDs. The
filter scopes the audit to the final encoder tier before CAAT checks ID
uniqueness. The two grader exports use the raw `output_text` value as the
output key. Separate grader files must cover exactly the selected IDs and
carry the same key for each. Their row order can differ. CAAT reads the
original prediction CSV directly.

## Results

| Measure | Archived grader | Candidate grader |
| --- | ---: | ---: |
| Accepted | 671 / 750 | 671 / 750 |
| Errors among accepted | 62 | 62 |
| Accepted error rate | 9.24% | 9.24% |
| Meets the example 10% target | Yes | Yes |

The two changed outcomes are complaints `22203333` and `23779143`. Each
has a tie at the top of its saved score map. Both fall below the frozen
cutoff, so the audit shows them as reviewed. Preparation writes
`replay/cfpb/reference.json` with direct counts and IDs. An independent
replay matched the report, including per-product counts, and reproduced
the upstream encoder-to-cheap cascade's 650 correct outcomes out of 750
after fallback. [Upstream fallback predictions](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/full_run.csv) ·
[Upstream cutoff semantics](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/src/followup/report.py#L176-L191)

On one Windows machine with Python 3.14, the median over nine runs with
cached inputs was 0.053 s for a purpose-built direct counting script,
0.089 s to produce the two grader exports and reference JSON, and 0.081 s
for the CAAT command. These timings cover execution and exclude writing
the integration. The walkthrough has one preparation command and one
audit command. The prediction input needs no filtered copy, renamed
columns, or manual grader join.

## Route and evaluate the same tier

Use the same `--prediction-filter tier=encoder` to route and evaluate the
original CSV. The file has 3,000 rows for 750 complaint IDs. CAAT selects
the encoder rows before checking ID uniqueness. Both reports record the
filter in `input.prediction_filter`.

```bash
caat route replay/cfpb/sources/encoder_eval.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --group-column product --output-key-column output_text --threshold 0.7980151891708375 --output replay/cfpb/decisions.csv
caat evaluate replay/cfpb/sources/encoder_eval.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --group-column product --correct-column correct --threshold 0.7980151891708375 --risk-target 0.10
```

Add `--batch-decisions replay/cfpb/decisions.csv --holdout-confidence 0.95`
to the paired audit above to check this fixed batch. Both graders label all
671 accepted predictions, so the exact batch error rate is 62/671 (9.24%).
With a 9% assumed error rate and 80% requested power, the label planner
calls for 599 sampled labels and at most 56 errors at the 10% target.
The seed-42 sample contains 57 errors under each grader, putting both
one-sided 95% upper bounds at 10.28%.

The 599 highest-scoring accepted rows contain 32 errors. Treating those
rows as a random draw gives a 5.96% bound, below the 9.24% census error
rate. `--sample-seed 42` rejects those rows because their IDs differ from
the random draw.
[Fixed-batch walkthrough](queue.md#check-the-error-rate-of-a-fixed-batch)

## Fit a cutoff on the encoder tier

Use the original combined export as input to `policy`. The filter supplies
750 encoder rows before the default seed-42 split of 150 design, 300
calibration, and 300 holdout rows.

```bash
caat policy replay/cfpb/sources/encoder_eval.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --correct-column correct --output-key-column output_text --track-fit-outputs --risk-target 0.10 --holdout-confidence 0.95 --holdout-output replay/cfpb/fitted-holdout.csv --output replay/cfpb/fitted-policy.json
```

At a 10% risk target, the fitted cutoff accepts 235 of 300 holdout rows
with 14 errors and a 9.16% one-sided 95% upper bound. The holdout export
keeps the original columns and contains only the selected tier's holdout
rows. Use `--prediction-filter tier=encoder` again when routing the original
mixed export with this cutoff report.
[Input filters](csv-format.md#select-a-run-from-a-combined-export)

## Send disagreements to review

Use `--disagreements-output` to save one CSV row for each sample whose
grades differ. The rows follow ID order and carry `id`, `score`,
`decision`, and `group`. The file includes `output_key` when the audit
uses a key column, and each grader's binary result appears in a column
named `grade_<grader>`. `decision` is the result at the frozen cutoff.
A completed audit writes the file after a pass or a target miss. An
audit with zero disagreements writes the header.

```bash
caat audit examples/audit/predictions.csv --threshold 0.75 --risk-target 0.70 --output-key-column output_key --grader-file reference=examples/audit/reference.csv --grader-file rubric=examples/audit/rubric.csv --disagreements-output replay/disagreements.csv
caat queue replay/disagreements.csv --prediction-filter decision=accept --output-key-column output_key --budget 1 --output replay/disagreement-review.csv
```

The queue example sends ticket `2` for review with its output key. Set
the budget to the number of accepted disagreements you can review.
[Queue options](queue.md) · [Audit report fields](reports.md#audit)

## Run as a CI check

Add these GitHub Actions steps after checking out CAAT. The action installs
the CAAT version from the referenced checkout into its own environment.

```yaml
- name: Prepare grader exports
  run: python scripts/prepare_cfpb_audit.py
- name: Check the frozen review policy
  id: audit
  uses: ./
  with:
    predictions: replay/cfpb/sources/encoder_eval.csv
    prediction-filter: tier=encoder
    id-column: complaint_id
    score-column: confidence
    group-column: product
    output-key-column: output_text
    grader-files: |
      archived=replay/cfpb/archived.csv
      regraded=replay/cfpb/regraded.csv
    threshold: "0.7980151891708375"
    risk-target: "0.10"
    report-path: replay/cfpb/audit.json
    disagreements-path: replay/cfpb/disagreements.csv
- name: Keep the audit artifacts
  if: ${{ always() && steps.audit.outputs.report-path != '' }}
  uses: actions/upload-artifact@v7
  with:
    name: caat-audit
    path: |
      ${{ steps.audit.outputs.report-path }}
      ${{ steps.audit.outputs.disagreements-path }}
```

From another repository, use `Daveyuwang/caat@v1.0.0` in place of `./`,
or pin the commit identified by the release tag. Paths resolve from
the calling repository's checkout. Provide either `grader-files`, with
one name and CSV or JSONL path per line, or `correct-columns`, with one inline
correctness column per line. List the reference grader first.
[Action inputs](../../action.yml)

To check a cutoff chosen by CAAT, set `threshold-report: replay/cutoff.json`
in the audit step. Give the path to a `select`, `policy`, or `run` report.
The action reads the full cutoff precision, ID column, score column, risk
target, and any saved holdout confidence. Explicit options override those
settings. Set `raw-score-column` to read a renamed original score column
using the saved direction. With `plan-report`, use the risk target and
confidence from the plan. For a numeric cutoff, supply `threshold` and
either `risk-target` or `plan-report`.
[Saving a cutoff](reports.md#reuse-a-saved-cutoff)

For a numeric cutoff with partially scored predictions, pass
`review-missing-scores: "true"` in the action's `with` block. The audit
keeps unscored rows in the coverage denominator and marks their decisions
as `review` in the disagreements export. A saved cutoff also supplies
its missing-score rule when it recorded one.

The `status` output preserves the CLI exit code. Both a missed target
(`1`) and an invalid input (`2`) fail the action. A passing audit (`0`) or
a target miss writes `report-path` for later steps. The default report
filename is `caat-audit.json`. Set `disagreements-path` to keep the CSV
alongside the report. The action exposes its path on status 0 or 1.
Set `holdout-confidence` to `"0.95"` if
the gate should use a one-sided upper bound on the accepted-set error rate.
A statistical check with a tracked cutoff returns status 2 if any holdout
row reuses a design or calibration output. The action inherits the tracked
key column when `output-key-column` is omitted.
[Track fit outputs](reports.md#track-outputs-used-to-choose-the-cutoff) · [Audit report fields](reports.md#audit) · [GitHub artifact action](https://github.com/actions/upload-artifact)

The routing example passes at 10% and misses at 9%. Changing a candidate
output key to the key from another tier for the same complaint produces
an input error. The report step keeps both passing and target-miss reports.

For a first run on your own exports, use your existing cutoff and error
target. Record any field conversions you needed and check one discordant
ID against its source output. The audit report and those notes are enough
to identify the next integration change.

### Check a labeled sample from a fixed batch

Set `batch-decisions` to the complete CSV from `caat route`. The action
counts its accepted rows and verifies the sample against that file.
Give the action the labeled sample drawn uniformly without replacement
from those accepted predictions. Pass the cutoff report and the saved
`plan-report` to reuse the plan's seed, risk target, and confidence level
and check the label count. The saved plan supplies the ID, score, and key
headers for the generated sample. Set `sample-seed` for a separate random queue
draw when the plan has no recorded seed.
[Sampling and labeling walkthrough](queue.md#check-the-error-rate-of-a-fixed-batch)

```yaml
- name: Check the labeled sample
  id: audit
  uses: ./
  with:
    predictions: replay/label-sample.csv
    threshold-report: cutoff.json
    grader-files: |
      primary=primary-labels.csv
      alternate=alternate-labels.csv
    batch-decisions: replay/new-decisions.csv
    plan-report: replay/label-plan.json
    report-path: replay/batch-audit.json
    disagreements-path: replay/batch-disagreements.csv
```

For full Inspect grader exports, add `allow-extra-grades: "true"` and
`grader-output-key-column: output_sha256` to the inputs above. The gate
matches each sampled ID and output key to one grade per file.
Set `grader-id-column` when the grader exports use a different ID header
from the predictions.

The gate compares each grader's exact upper error bound for the fixed
batch to the risk target. The JSON includes each upper error count and
the accepted population size. The same output paths are available when
a grader misses the target. If you have only the count, use
`accepted-population` with `overall.accepted` from the routing JSON.
[Fixed-batch report fields](reports.md#fixed-accepted-batch)
