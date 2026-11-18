# Audit a grader change in CI

Compare graders on the same outputs at one cutoff. In the bundled ticket example the reference grader finds one error among three accepted predictions. The alternate grader finds two, exceeding the 40% target.

```bash
caat audit examples/audit/predictions.csv --output-key-column output_key --grader-file reference=examples/audit/reference.csv --grader-file rubric=examples/audit/rubric.csv --threshold 0.75 --risk-target 0.40 --disagreements-output replay/disagreements.csv
```

The CLI returns 1 when a grader misses the target. The disagreement CSV identifies the affected outputs.

## Run as a CI check

After checking out the repository that contains your predictions, add this step to your GitHub Actions workflow. Paths are relative to that checkout.

```yaml
- name: Audit accepted predictions
  uses: Daveyuwang/caat@v1.1.2
  with:
    predictions: examples/audit/predictions.csv
    output-key-column: output_key
    grader-files: |
      reference=examples/audit/reference.csv
      rubric=examples/audit/rubric.csv
    threshold: "0.75"
    risk-target: "0.40"
    report-path: replay/audit.json
    disagreements-path: replay/disagreements.csv
```

Supply `threshold-report` to reuse a saved policy. The action exposes `status`, `report-path` and `disagreements-path`. [Action inputs and outputs](../../action.yml) · [Audit report fields](reports.md#audit)
