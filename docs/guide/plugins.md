# Extend CAAT with a plugin

`caat plugins --json` lists the installed components and their source
packages. CAAT uses Python entry points, so a component can ship with its
own dependencies in a separate package. An entry point exposes an
instance of the stage's component type. Choose a unique name for each
stage. Built-in names are reserved.

| Group | Instance | CLI |
| --- | --- | --- |
| `caat.features` | `caat.features.Feature` | `features --feature NAME` |
| `caat.calibrators` | `caat.calibrate.Calibrator` | `calibrate fit --method NAME`, `policy --calibrate NAME` |
| `caat.metrics` | `caat.metrics.Metric` | `evaluate --metric NAME` |
| `caat.rules` | `caat.rules.Rule` | `select --rule NAME`, `policy --rule NAME` |
| `caat.adapters` | `caat.adapters.Adapter` | `evaluate`, `policy`, `select`, or `audit --adapter NAME` for one input file |
| `caat.renderers` | `caat.renderers.Renderer` | `--render NAME --render-output FILE` on the same four report commands |

## Calibrator

Create a package containing `extension.py`. This example fits the
calibration set's correctness rate as a constant probability baseline.

```python
from statistics import mean
from caat.calibrate import Calibrator


def fit(scores, outcomes, *, bins=10):
    return {"rate": mean(outcomes)}


def restore(parameters):
    return lambda score: parameters["rate"]


calibrator = Calibrator(fit, restore, monotone=True)
```

Add the entry point to the package's `pyproject.toml`. The package should
depend on a CAAT release supporting this contract.

```toml
[project.entry-points."caat.calibrators"]
base-rate = "extension:calibrator"
```

Install the package in the same environment as CAAT, then run
`caat plugins`. Select the component with `--method base-rate`. CAAT
stores the parameters returned by `fit` in JSON and calls `restore` to
replay the map. The map takes one score and returns a probability.
Declare `monotone=True` when the map is nondecreasing.

## Feature

```python
from caat.features import Feature

score = Feature(lambda response: float(len(response.text)),
                direction="lower", requires=("text",), cost="no-extra-call")
```

Register `short-response = "extension:score"` under `caat.features`.
CAAT supplies a saved response and writes the returned finite score.
Declare the fields the scorer requires and the direction of confidence.
[Response format and costs](features.md)

## Metric

```python
from caat.metrics import Metric

accuracy = Metric(lambda rows: sum(row.outcomes[0] for row in rows) / len(rows),
                  direction="higher")
```

Register `accuracy = "extension:accuracy"` under `caat.metrics`.
Return a number or JSON-compatible table from the graded rows.
Set `requires_probability=True` if the score must be a probability.
The result appears under `extra_metrics` in the evaluation report.

## Selection rule

```python
from caat.rules import Rule, empirical


def half_target(candidates, risk_target, failure_rate):
    return empirical(candidates, risk_target / 2, failure_rate)


rule = Rule(half_target)
```

Register `half-target = "extension:rule"` under `caat.rules`.
The rule returns a `RuleResult` containing one decision per cutoff in
descending cutoff order. Use `True` for a qualifying cutoff, `False`
for a failed test, and `None` for an untested cutoff. CAAT selects the
lowest qualifying cutoff. Declare the sampling assumptions when the
result claims finite-sample risk control.

## Input adapter

```python
import json
from caat import GradedRow
from caat.adapters import Adapter


def read(path, options):
    for row in json.loads(path.read_text(encoding="utf-8")):
        yield GradedRow(row["id"], float(row["score"]), (row["correct"],))


def fixture(directory):
    path = directory / "rows.json"
    path.write_text('[{"id":"1","score":0.9,"correct":true}]', encoding="utf-8")
    return path, {}


adapter = Adapter(read, source_format="json-array", source_version="1",
                  fixture=fixture)
```

Register `json-array = "extension:adapter"` under `caat.adapters`.
The adapter yields rows with unique IDs, finite scores, and Boolean
outcomes. Multiple outcomes follow the order in the `correct_columns`
option. `fixture` supplies the fixture path and options for the reusable
contract test.

## Renderer

```python
from caat.renderers import Renderer

accepted = Renderer(lambda report: str(report["overall"]["accepted"]) + "\n",
                    format="text", commands=("evaluate",))
```

Register `accepted = "extension:accepted"` under `caat.renderers`.
The renderer receives a report mapping and returns text for its
declared format. It declares which commands it supports.
[Report fields](reports.md)

## Run the contract suite

From a CAAT checkout with the plugin installed, run
`python -m unittest discover -s tests/contracts`. The suite uses the
registered names and checks both built-ins and installed plugins.
Build one representative end-to-end example for a new component's
own tests.
