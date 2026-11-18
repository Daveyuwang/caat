# Terms in CAAT reports

| Term | Meaning |
| --- | --- |
| Score | A number used to order predictions by confidence. Higher oriented scores enter the accepted set first. |
| Grader | A person, label set or rule that marks each prediction correct or incorrect. |
| Cutoff | The minimum score for eligibility. Ties at the cutoff qualify. |
| Coverage | Accepted predictions divided by all predictions. |
| Accepted error (risk) | Incorrect accepted predictions divided by accepted predictions. Undefined when none qualify. |
| Risk target | The maximum accepted error the policy aims to allow. |
| Design, calibration and held-out rows | Separate data for choosing the score and candidate cutoffs, selecting the cutoff and checking it. |
| Confirmed policy | A policy whose selected cutoff passed every required held-out check. |
| Withheld policy | A policy that did not qualify for routing. A selected cutoff can still define an audit sample. |
| Confidence bound | A one-sided statistical bound on accepted error at the requested confidence level. |
| Power | The chance of reaching a specified verdict at an assumed error rate. |

## Comparing scores

AUROC measures how often a correct prediction outranks an incorrect one, giving half credit for a tie. AURC averages error as coverage increases. Lower AURC is better. For a probability score, Brier loss is the mean squared difference from binary correctness. [Score comparison](scores.md) · [Report fields](reports.md)
