# Use Every Eval Ever sample files

Download a per-sample JSONL file from the public datastore. Use a file
with one saved text response per question and a boolean correctness
grade. The same command reads local exports with this layout. [Datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) · [Instance schema](https://github.com/evaleval/every_eval_ever/blob/main/every_eval_ever/schemas/instance_level_eval.schema.json)

```bash
caat eee export samples.jsonl --output replay/to-score.csv
```

CAAT exports `evaluation_id`, `result`, `id`, and `output_sha256` for the
score join. `result` is `evaluation_result_id` when present and the
evaluation name in older logs. `id` is `sample_id`. The export uses the
formatted prompt when supplied and the raw prompt otherwise. It writes
the single value in `output.raw` as `completion`. Select a result with
`--result NAME` for files that carry several metrics.

Run your confidence scorer on the export. Write the four identity columns
and a finite `confidence` score for each response. Larger values favor
acceptance. The native `evaluation.is_correct` supplies the binary
`correct` column after the join. `evaluation.score` is a grade in the
Every Eval Ever schema. The confidence comes from your score file.

```bash
caat eee join samples.jsonl --scores replay/scored.csv --output replay/graded.csv
caat policy replay/graded.csv --score-column confidence --correct-column correct --risk-target 0.10
```

The join checks that every selected sample has one score with the matching
output hash. Pass the same `--result` to export and join when selecting a
metric. Use `--score-column` if the confidence file has a different header.
Add `--lower-is-better COLUMN` for a score where smaller values favor
acceptance. The joined file includes the original column and its negation
as `negative_COLUMN`. Use the latter for score comparison and fitting.

For incomplete score exports, use `--review-missing-scores` on `join`
and `policy`. The join keeps all selected answers and checks the identity
of every supplied score record. [Missing scores](inspect.md#keep-answers-with-missing-scores)

Repeat `--score-file COLUMN=FILE` to align several scores to the same
samples. Each CSV or JSONL file needs its named score column and the four
identity fields from `eee export`.

```bash
caat eee join samples.jsonl --score-file confidence=replay/scored.csv --score-file perplexity=replay/perplexity.jsonl --lower-is-better perplexity --output replay/compared.csv
caat scores replay/compared.csv --score-column confidence --score-column negative_perplexity --correct-column correct
```

The native import was checked on 1,319 saved GSM8K answers from
Grok 2. The 1,253 correct labels reproduced the archived aggregate
accuracy of 95.00%. The replay verified the bytes against the fixed
content index. The legacy aggregate lists a different checksum.
[Pinned sample log](https://huggingface.co/datasets/evaleval/EEE_datastore/blob/4df3db0f3268bab9282f9f354b01debbee63c81e/data/gsm8k/grok/grok-2-1212/a2be81d9-d344-4a6c-91ed-653303edf43a_samples.jsonl) · [Aggregate result](https://huggingface.co/datasets/evaleval/EEE_datastore/blob/4df3db0f3268bab9282f9f354b01debbee63c81e/data/gsm8k/grok/grok-2-1212/a2be81d9-d344-4a6c-91ed-653303edf43a.json) · [Content index](https://huggingface.co/datasets/evaleval/EEE_datastore/blob/4df3db0f3268bab9282f9f354b01debbee63c81e/flat/indexes/by_collection/gsm8k.jsonl)
