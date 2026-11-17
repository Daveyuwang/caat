# Use Inspect evaluations with external confidence scores

From the CAAT checkout, install the Inspect extra to read a completed
`.eval` or `.json` log. Choose one epoch so each question can be assigned
to a single CAAT split. Both formats include samples added while the
evaluation was running. CAAT reads indexed `.eval` logs in batches and
loads a fresh snapshot of a `.json` log for each command.

```bash
python -m pip install '.[inspect]'
caat inspect export run.eval --epoch 1 --output scoring-input.csv
```

On Python 3.10, the extra installs Inspect 0.3.252 or newer and keeps its
Agent Client Protocol dependency below version 1. On Python 3.11 and
later, the extra uses Inspect 0.3.289 or newer with the updated imports.
[Inspect 0.3.252 metadata](https://pypi.org/project/inspect-ai/0.3.252/) · [Upstream import issue](https://github.com/UKGovernmentBEIS/inspect_ai/issues/6417)

The export contains `eval_id`, `id`, `epoch`, and `output_sha256`, along
with `input` and `completion`. The `input` field stores chat messages as
a JSON string and includes media retained as attachments in the log. Use
`--output scoring-input.jsonl` for media or long prompts that exceed a CSV
reader's field limit.

Run your confidence scorer on the exported inputs and completions. Return
the four identity columns and a finite `confidence` column with larger
scores for answers you would prefer to accept. The hash identifies the exact
saved text completion.
[Inspect log API](https://inspect.aisi.org.uk/eval-logs.html)

```bash
caat inspect join run.eval --epoch 1 --scores scored.csv --grader primary --grader alternate --output graded.csv
caat policy graded.csv --score-column confidence --correct-column primary --check-column alternate --risk-target 0.20 --holdout-confidence 0.95 --output policy.json
```

Use the scorer names from the log in `--grader`. The importer accepts
binary grades encoded as `C`/`I`, 0/1, or booleans. It outputs one grade
column per listed scorer. Score rows may be in any order. The importer checks
that each selected sample has exactly one score and that its completion
hash matches. Pass `--score-column` on both commands when using a different
score header.

`--check-column alternate` applies the frozen threshold to the alternate
grader on the holdout. `--holdout-confidence 0.95` bases the holdout's
target decision on the one-sided upper error bound and includes the observed
error rate in the same report. Save `policy.json` to reuse its full cutoff
with `--threshold-report`.
[Report fields](reports.md)

## Compare several score files

Use a `--score-file COLUMN=FILE` argument for each confidence column to
compare. Each CSV or JSONL score file keeps the four identity columns
and covers every sample in the selected epoch. The joined table uses the
named score columns in argument order.

```bash
caat inspect join run.eval --score-file confidence=scored.csv --score-file token_count=lengths.jsonl --lower-is-better token_count --grader primary --output compared.csv
caat scores compared.csv --score-column confidence --score-column negative_token_count --correct-column primary --coverage 0.75 --output comparison.json
```

This example compares an external confidence score with a length control
that assigns higher scores to shorter completions. `--lower-is-better`
keeps `token_count` and adds `negative_token_count` by negating it.
Repeat the option for other scores whose smaller values favor acceptance.
Use the generated columns with `scores` and `policy`. `caat scores` uses
the design split to recommend a score and reports both candidates on the
same held-out rows. Use the same seed for the subsequent policy run.
[Score comparison](scores.md)

## Keep answers with missing scores

Add `--review-missing-scores` to `join` when an external scorer omits
answers or uses empty CSV cells or JSONL `null` scores. The join keeps
every selected answer in log order and writes empty cells for each
missing score. Each supplied record still needs the four identity fields
and the matching completion hash. The same option works with several
`--score-file` inputs, including columns listed in `--lower-is-better`.

```bash
caat inspect join run.eval --scores scored.jsonl --grader primary --review-missing-scores --output graded.csv
caat policy graded.csv --score-column confidence --correct-column primary --risk-target 0.20 --holdout-confidence 0.95 --review-missing-scores --output policy.json
```

The policy counts unscored answers in total coverage and sends them to
review. You can also set `review_missing_scores = true` in a TOML workflow.
[TOML workflow](workflow.md#review-unscored-outputs)

## Check a rescored log at a frozen cutoff

Join the original scorer with the external confidence scores. Export the
holdout when selecting the cutoff. The export keeps the four Inspect
identity columns alongside the scores and grades.

```bash
caat inspect join run.eval --scores scored.csv --grader primary --output graded.csv
caat policy graded.csv --score-column confidence --correct-column primary --risk-target 0.20 --holdout-confidence 0.95 --output-key-column output_sha256 --track-fit-outputs --output policy.json --holdout-output holdout.csv
```

After the policy selects a cutoff, use `inspect grades` to extract the
holdout samples from the original and rescored logs. When a `.eval` index
covers the holdout, CAAT reads the sample records by ID. Both logs must
contain the same evaluation ID, sample IDs, epoch, and completion hashes
for the holdout. Set `--epoch` on the Inspect commands for an epoch other
than 1.

```bash
caat inspect grades run.eval --predictions holdout.csv --grader primary --output primary.csv
caat inspect grades rescored.eval --predictions holdout.csv --grader alternate --output alternate.csv
caat audit holdout.csv --threshold-report policy.json --output-key-column output_sha256 --grader-file primary=primary.csv --grader-file alternate=alternate.csv --holdout-confidence 0.95 --output audit.json
```

The grader exports contain `id`, `output_sha256`, and `correct`. Keep
the holdout separate from the rows used to select the cutoff.
[Separate grader exports](csv-format.md#separate-grader-exports) · [Track fit outputs](reports.md#track-outputs-used-to-choose-the-cutoff)

Use `caat calibrate fit --grader-file` with the calibration scores and
their grade export to fit a correctness probability map.
[Calibration walkthrough](calibration.md#fit-from-a-separate-grader-export)
