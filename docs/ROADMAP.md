# CAAT roadmap

CAAT covers the confidence pipeline on saved model outputs: score,
calibrate, evaluate, and use. The local collection extra can produce those
inputs. [Sampling-budget replay](research/sample-budget.md)

Version 0.3.0 added inputs from Inspect, lm-evaluation-harness, Every Eval
Ever, and HELM, a one-file workflow, label planning, monitoring, and
summaries that state why a result holds.
[Changelog](../CHANGELOG.md)

| Stage | Open work |
| --- | --- |
| Score | A tunable confidence scorer that combines signals for a user's task, compared against single signals on held-out data. Model integration for collecting those signals. |
| Calibrate | Calibration references and fixtures for score types proposed by users. |
| Evaluate | More standard LLM evaluation sets beyond IFEval, MMLU-Pro, and TriviaQA. |
| Use | Label columns in the TOML workflow. A queue file without the empty group column. `caat route` naming the rows that chose the cutoff. Recheck proportional grader shifts on held-out WildBench archives. |
| Reports | Extend the offline report when a replay needs additional fields. |

Start with a saved input, an independent check or hand-calculated
example, and an acceptance case. [Contributor guide](../CONTRIBUTING.md) ·
[Plugin contracts](guide/plugins.md)
