# Contributing to CAAT

Install Python 3.10 or later, clone the repository, and run these commands
from the checkout in a virtual environment.

```bash
python -m pip install -e ".[dev]"
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
python -m unittest discover -s tests/properties -v
```

The tests execute the README examples, compare their reports with the saved
fixtures, and replay the three bundled CPU benchmarks. A change to a report
or a numerical method needs the corresponding test and an explanation of
the affected fields or results.

The installed-wheel check also replays an upgrade. The PyPI 0.7.0 wheel
produced the IFEval policy, routed IDs and ANLI R2 plan in
`tests/fixtures/upgrade/`. The 0.8.0rc1 wheel built from `03311b2` issued
the first ten R2 rows and recorded one returned grade. The SQL dump
restores that journal so the candidate can finish the original draw.

The installed wheel includes `caat/py.typed`. CI type-checks the public
imports and verifies that invalid arguments still produce diagnostics.
Python report payloads use dictionaries with the fields documented in
the [report reference](docs/guide/reports.md).
[Packaging type information](https://typing.python.org/en/latest/spec/distributing.html#packaging-type-information)

```bash
python -m pip install ".[dev]"
python -m mypy --strict --python-version 3.10 tests/types/public_api.py
```

## Build the documentation

Build from the checkout root and serve the generated pages locally.

```bash
python -m pip install ".[docs]"
python scripts/build_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory replay/docs/site
```

Open `http://127.0.0.1:8000`. CI runs the same strict build and provides a
downloadable `caat-docs` artifact with the pages and bundled examples.

## Extend a pipeline stage

Implement a component, register it, and add a hand-checkable case
and documentation. Use the examples to test a plugin in a separate
package or register a built-in in the stage's module. The contract
suite runs against each installed component.
[Registry contracts and six worked examples](docs/guide/plugins.md) ·
[Contract tests](tests/contracts/test_components.py)

For a new core component, include an independent reference or a
hand-checked result and a saved replay on two task types. Document the
comparison in [benchmark results](docs/research/benchmarks.md).

## Add a benchmark

A benchmark preparation script belongs in `scripts/`. Reuse cached downloads
and CSV writing from `scripts/common.py`. Document the dataset revision,
split, saved output run, score definition, grader, and a replay command.
Fit score functions on design data and choose thresholds on calibration
data before evaluating held-out rows.
[Benchmark sources and preparation](docs/research/benchmarks.md)

## Public interface

CAAT versions the names exported from `caat`, the six plugin contracts,
CLI flags, exit codes, and documented report fields. A flag or report
field stays available through at least one minor release after
deprecation. A reporting change that alters the meaning of an existing
field increments `schema_version`. Keep the [report guide](docs/guide/reports.md),
[golden reports](https://github.com/Daveyuwang/caat/tree/main/tests/golden), and [changelog](CHANGELOG.md) aligned.

## Build a release

Run the suite from a clean checkout. Use the same build command as the
release workflow.

```bash
python -m pip install build==1.6.1 twine==7.0.0
python scripts/build_release.py
```

The script sets the archive timestamp from the checkout commit, builds the
wheel from the finalized source archive, and checks both files with Twine.
Set `SOURCE_DATE_EPOCH` to reuse a timestamp or to build from a source tree
without Git metadata.

Install the wheel in an empty environment and run the README examples.
Update the [changelog](CHANGELOG.md) and tag the tested commit. A version
tag runs the [release workflow](https://github.com/Daveyuwang/caat/blob/main/.github/workflows/release.yml) and produces
the wheel and source archive.

After the maintainer approves PyPI publication, set up a Trusted Publisher
for project `caat-annotation` with owner `Daveyuwang`, repository `caat`,
workflow `release.yml`, and environment `pypi`. Run the Release workflow
from the version tag with `publish` enabled. Then install the exact version
in a fresh environment with `python -m pip install caat-annotation==VERSION`
and run the README examples.
[Create a PyPI project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
