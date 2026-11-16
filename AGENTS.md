# CAAT

This folder is the working checkout for [Daveyuwang/caat](https://github.com/Daveyuwang/caat).
CAAT evaluates confidence-based review policies for saved model predictions.
Use [README.md](README.md) for public setup and [docs/benchmarks.md](docs/benchmarks.md)
for dataset provenance and replay commands.

## Current direction

- Prioritize the Python evaluator, its CSV interface, and reproducible CPU benchmarks.
  Keep the installed toolkit compatible with Python 3.10+ and the standard library.
- Prioritize useful open-source workflows and reproducible examples. Use
  benchmark evidence to choose features and support a technical report.
- Keep confidence scores configurable by task. Use risk–coverage for ranking
  scores and Brier score for probabilities of correctness. Follow CS546
  Lecture 18 as methodological guidance.
- Use the saved course trust scorer as a historical baseline.

## Working files

- [`src/caat/`](src/caat/) contains the evaluator and `caat evaluate`.
- [`examples/`](examples/) contains the verified CSV fixtures.
  IFEval, MMLU-Pro, and TriviaQA each have 24 rows for checking the pipeline.
  `ifeval_graders.csv` has all 541 Qwen IFEval rows with both graders for
  the `caat policy` example.
- [`scripts/`](scripts/) contains benchmark preparation and replay entry points.
  [`results/`](results/) contains the benchmark comparisons and saved course results.
- `.venv/` is the local Python environment. `replay/` holds generated reports
  and cached inputs. Git ignores both directories.

## Local checks

Run from this folder in a Python virtual environment. On Windows, use
`./.venv/Scripts/python.exe` if the environment is inactive.

```bash
python -m pip install -e .
python -m unittest discover -s tests
python scripts/replay_benchmarks.py
```

The full test suite includes a replay check. [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs the same tests, the README examples, and a wheel build on every pull request.

The replay writes `replay/benchmarks/summary.csv`, plus a metrics JSON and a
risk–coverage CSV per benchmark. With the included scores and cutoffs, IFEval
accepts 12 rows with 4 errors, MMLU-Pro accepts 12 with 5 errors, and TriviaQA
accepts 13 with 0 errors.

## YU Skills

Before starting work, read the matching `SKILL.md` and any references needed
for the task. Use the original files under `D:/YU-skills/skills/` on Windows
or `/mnt/d/YU-skills/skills/` in WSL.

Use `yu-skills:pragmatic-coding` for implementation,
`yu-skills:clear-writing` for documentation, and `yu-skills:oss-ideas` for
decisions about CAAT's open-source direction or a report based on the project.

## Working conventions

- Put model-specific preparation in `scripts/` and reuse the shared evaluator.
  Keep the existing CLI examples runnable when changing the CSV interface.
- For new benchmarks, record the dataset version, split, archived model run,
  score definition, and grader. Verify row alignment before reporting results.
  Fit calibration and choose thresholds on validation data. Report results
  on held-out test data.
- Start with `git status --short --branch` and preserve existing edits.
  Keep the current refactor local until asked to commit or push. Use short
  commit messages.
