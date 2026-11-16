# Contributing to CAAT

Install Python 3.10 or later, clone the repository, and run these commands
from the checkout in a virtual environment.

```bash
python -m pip install -e ".[dev]"
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
```

The tests execute the README examples, compare their reports with the saved
fixtures, and replay the three bundled CPU benchmarks. A change to a report
or a numerical method needs the corresponding test and an explanation of
the affected fields or results.

## Add a selection rule

Add a function to [`src/caat/rules.py`](src/caat/rules.py). It receives the
candidate table in descending threshold order, a risk target, and a
failure rate. Return a `RuleResult` with one decision per candidate.
`True` marks a qualifying cutoff, `False` marks a failed test, and
`None` marks a candidate the rule skipped. CAAT selects the least
restrictive qualifying threshold.

For example, an empirical rule that divides the target by two can reuse
the existing rule.

```python
def half_target(candidates, risk_target, failure_rate):
    return empirical(candidates, risk_target / 2, failure_rate)


RULES["half-target"] = half_target
```

Add its name to the README and a hand-checkable case to
[`tests/test_caat_selection.py`](tests/test_caat_selection.py). The CLI
reads its choices from the registry. A statistical rule should set
`controls_risk=True` only when its test controls the stated risk under
the sampling assumptions. Set `per_candidate_level` when the same
significance level applies to all candidates.

## Add an input adapter or benchmark

An adapter yields `caat.GradedRow` records with stable IDs, finite scores,
and Boolean outcomes in grader order. Include output identities when the
source records them. An adapter can live in `caat/adapters/` and consume
the existing public API without changes to the policy or metric code.
[CSV input contract](docs/guide/csv-format.md)

A benchmark preparation script belongs in `scripts/`. Reuse cached downloads
and CSV writing from `scripts/common.py`. Document the dataset revision,
split, saved output run, score definition, grader, and a replay command.
Fit score functions on design data and choose thresholds on calibration
data before evaluating held-out rows.
[Benchmark sources and preparation](docs/research/benchmarks.md)

## Build a release

Run the suite from a clean checkout. Build both distribution files and
check their metadata.

```bash
python -m pip install build==1.6.1 twine==7.0.0
python -m build
python -m twine check --strict dist/*
```

Install the wheel in an empty environment and run the README examples.
Update the [changelog](CHANGELOG.md) and tag the tested commit. A version
tag runs the [release workflow](.github/workflows/release.yml) and produces
the wheel and source archive.

After the maintainer approves PyPI publication, set up a Trusted Publisher
for project `caat-annotation` with owner `Daveyuwang`, repository `caat`,
workflow `release.yml`, and environment `pypi`. Run the Release workflow
from the version tag with `publish` enabled. Then install the exact version
in a fresh environment with `python -m pip install caat-annotation==0.1.0`
and run the README examples.
[Create a PyPI project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
