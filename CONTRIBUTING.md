# Contributing to CAAT

From a checkout with Python 3.10 or later, install the development tools and run the checks for your change.

```bash
python -m pip install -e ".[dev,workflow]"
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests
python -m unittest discover -s tests/properties
python -m mypy --strict --python-version 3.10 tests/types/public_api.py
```

## Documentation

```bash
python -m pip install ".[docs]"
python scripts/build_docs.py
python -m http.server 8000 --directory replay/docs/site
```

For a new component, document its inputs and outputs and add a small example with a known result. For a benchmark, record the dataset, split, model, score, grader and replay command. [Plugin contracts](docs/guide/plugins.md) · [Benchmark examples](docs/research/benchmarks.md)

## Public interface

CAAT 1.x supports the documented CLI flags and exit statuses, JSON report fields, six plugin contracts, and GitHub Action inputs and outputs. The public Python exports are `GradedRow`, `load_csv`, `evaluate`, `select`, `policy` and `audit`.

Breaking changes require a major version. Deprecated flags and fields remain available through at least one subsequent minor release. A change in field meaning increments `schema_version`. Saved reports retain their producer and schema versions. [Report reference](docs/guide/reports.md)

## Release

Update the version and changelog, then tag the tested commit as `vX.Y.Z`. The release workflow builds a wheel and source distribution. To publish them, run the workflow on that tag with `publish` enabled in the `pypi` environment. [Release workflow](https://github.com/Daveyuwang/caat/blob/main/.github/workflows/release.yml)
