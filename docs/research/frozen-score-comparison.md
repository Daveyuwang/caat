# Frozen scores at two calibration budgets

The table gives the percentage of held-out decisions that changed between paired refits. The examples use the same two archived score tables and error target (40%) as the routing stability study.

| Grid | Calibration size | Inspect TriviaQA | IFEval |
| --- | --- | ---: | ---: |
| Original design ranks | Small | 14.31% | 36.18% |
| Ten fixed values | Small | 16.89% | 42.45% |
| Ten fixed values | Large | 6.39% | 17.15% |
| Independent reference ranks | Small | 14.73% | 38.62% |
| Independent reference ranks | Large | 5.63% | 10.95% |

The frozen scalar needs only unlabeled predictions to propose its quantiles. Inspect uses 160 original design rows, 120 independent reference scores and calibration sizes of 320 and 480. IFEval uses 86, 66, 172 and 258 respectively. The script prints coverage and empty fits for every arm. For 200 matched splits, SciPy exact binomial tails reproduce CAAT's counts, p-values and decisions. At seed 42 the MAPIE 1.6.0 binary precision controller and CAAT differed on three of 100 candidate tests. MAPIE reconstructs its precision risk from a floating-point proportion and uses its native p-value implementation. CAAT tests integer error counts directly. The supplied rank encoding preserves boundary ties. [Reference comparison](../../scripts/study_mapie.py)

The Inspect verifier replay uses two disjoint 400-question TriviaQA cohorts, Qwen3.5-4B outputs, generator token likelihoods and Qwen3.5-2B P(True). A 200-label design selects P(True) and fits the standardized logistic scorer. At 75% coverage the fitted score reduces accepted error relative to P(True) by 3.47 and 3.78 points. Paired 95% intervals are [−5.43, −1.10] and [−6.13, −1.27]. Against summed log probability the intervals cross zero. The archived answers required teacher forcing to recover token likelihoods. The table includes the original response hashes, grades and splits. [Inspect features](../../results/routing_stability/inspect_verifier.csv) · [BoolQ confirmation protocol](boolq-protocol.md)

```bash
python -m pip install ".[scorer]" mapie==1.6.0
python scripts/study_frozen_allocation.py
python scripts/study_inspect_verifier.py
```
