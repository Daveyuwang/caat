# Monitor a frozen policy

Supply predictions in arrival order. The monitor counts error among accepted predictions with grades and can alert when coverage falls below a floor.

```bash
caat monitor results/monitoring/heldout.csv --threshold-report results/monitoring/selection.json --correct-column correct --coverage-window 50 --min-coverage 0.80 --json
```

The example uses the saved Omni-MATH policy. A full 50-row window with coverage below 80% triggers a coverage alert. Ungraded and reviewed predictions count toward the window.

## Sequential test

The error test uses the accepted grades in order. It controls the probability of any false alert at `--failure-rate` when each accepted prediction’s conditional error probability stays at or below the target. The first alert remains in the report after later windows recover.

## Read arriving labels from a separate file

```bash
caat monitor predictions.csv --threshold-report cutoff.json --grader-file human=labels.csv --output monitor.json
```

Supply the batch IDs and output keys in the grader file. Unlabeled predictions contribute to coverage. They enter the error test when their grade is available.

## Monitor a routed batch

```bash
caat monitor --batch-decisions decisions.csv --threshold-report cutoff.json --grader-file human=labels.csv --output monitor.json
```

Use `coverage_check` for the coverage window and `first_alert` for the error test. An alert returns exit code 1. [Report fields](reports.md#monitor)
