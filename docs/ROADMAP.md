# CAAT roadmap

CAAT covers the confidence pipeline on saved model outputs: score,
calibrate, evaluate, and use. The local collection extra can produce those
inputs. [Sampling-budget replay](research/sample-budget.md)

Version 0.5 applies a frozen scorer from routing through monitoring and supports independent calibration labels and ungraded reference scores. On BoolQ, reference quantiles improved stability at 320 labels, and an exploratory native score with the answer identity had the lowest accepted error among the reported 0.5 comparisons. [Frozen workflow](guide/workflow.md#calibrate-a-frozen-score) · [BoolQ results](research/boolq-results.md)

Version 0.3.0 added inputs from Inspect, lm-evaluation-harness, Every Eval
Ever, and HELM, a one-file workflow, label planning, monitoring, and
summaries that state why a result holds.
[Changelog](../CHANGELOG.md)

| Stage | Open work |
| --- | --- |
| Score | Compare complementary signals on a new task at matched inference cost. |
| Calibrate | Calibration references and fixtures for score types proposed by users. |
| Evaluate | More standard LLM evaluation sets beyond IFEval, MMLU-Pro, TriviaQA, and ARC-Challenge. |
| Use | Validate a more stable cutoff for learned scores on fresh data. Label columns in `caat calibrate fit`. A queue file without the empty group column. `caat route` naming the rows that chose the cutoff. Recheck proportional grader shifts on held-out WildBench archives. |
| Reports | Extend the offline report when a replay needs additional fields. |

Start with a saved input, an independent check or hand-calculated
example, and an acceptance case. [Contributor guide](../CONTRIBUTING.md) ·
[Plugin contracts](guide/plugins.md)
