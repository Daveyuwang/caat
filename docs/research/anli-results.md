# Predicted classes and verification on ANLI

At 75% global coverage on ANLI R1, adding the predicted class to native likelihood features lowered accepted error from 29.08% to 28.50%. The paired difference was −0.58 percentage points (95% interval −1.57 to +0.31). Adding the verifier to the design-selected cheap score, the top-two gap, changed error by +0.24 points (−0.36 to +0.82). Both comparisons missed the prespecified two-point improvement. Class features and verifier calls remain opt-in. [Frozen protocol](anli-protocol.md) · [Design selection](../../results/anli/design.json) · [All intervals and class budgets](../../results/anli/summary.json)

| Score | R1 accepted error (635 of 846) | R3 transport error (846 of 1,128) |
| --- | ---: | ---: |
| Random | 35.93% | 49.91% |
| Predicted class alone | 32.19% | 47.02% |
| Native likelihood gap | 29.06% | 46.82% |
| Winning log probability and gap | 29.08% | 46.81% |
| Native pair and predicted class | 28.50% | 45.86% |
| All three option logits | 34.49% | 48.35% |
| Gap and verifier | 29.29% | 46.85% |
| Native pair, predicted class and verifier | 28.66% | 45.98% |

The class-aware native scorer sent 140 of the 428 predicted neutral relations to review, 36 of 156 contradictions and 35 of 262 entailments on R1. These groups accounted for 105, 23 and 53 errors respectively among the accepted answers. Neutral coverage fell from 70.56% to 67.29% relative to the numeric pair. None of the three classes lost ten coverage points. A 400-example design-only rotation of the option tokens changed 90 semantic predictions. [Per-class results](../../results/anli/summary.json) · [Keyed outputs](../../results/anli/scores.csv)

The RTX 5080 scored 3,014 original prompts in 189 batches (500,325 tokens, 130.28 s). The verifier added 3,014 calls in 189 batches (410,654 tokens, 27.50 s). The separate design rotation cost 400 calls, 25 batches and 17.79 s. Times exclude model loading. On R1 the paid gap model incurred 1.77 more expected accepted errors per 1,000 extra verifier calls, versus the gap alone. Boundary ties can give fractional errors at a fixed budget. [Collection measurements](../../results/anli/collection.json)

## The cutoff failed its independent audit

The design-selected gap cutoff of 4.75 accepted 380 calibration answers with 50 errors, enough to pass the calibration test. It accepted 588 R1 answers with 154 errors. The one-sided 95% upper bound was 29.34%, above the 20% target. Applied unchanged on R3, it accepted 790 answers with 367 errors and a 49.44% upper bound. The entire cohorts also became harder, with error rising from 21.04% on calibration to 35.93% on R1 and 49.91% on R3. The workflow marks the policy unconfirmed. Routing with that report is blocked. Replaying the explicit numeric cutoff for analysis reproduces the candidate IDs in the evaluation, audit and monitor checks. [Candidate IDs](../../results/anli/accepted.csv) · [Scores and grades](../../results/anli/scores.csv)

To repeat the CPU analysis from the checkout, install the scorer extra and run the study. The manifest fixes one hypothesis per premise. The test grades were opened after the design selection and analysis were committed. The 160 reference rows carry no grades. [Manifest](../../results/anli/manifest.csv)

```bash
python -m pip install ".[scorer,workflow]"
python scripts/study_anli.py
```

To repeat collection, use the pinned revisions and the library versions in the collection measurements. Install CUDA PyTorch, Transformers, Accelerate, PyArrow and huggingface-hub in the collection environment.

```bash
python scripts/prepare_anli.py results/anli/manifest.csv
python scripts/collect_anli.py generate --output replay/anli
python scripts/collect_anli.py rotate --output replay/anli
python scripts/collect_anli.py verify --output replay/anli
python scripts/assemble_anli.py replay/anli --design-only --output replay/anli/design.csv
python scripts/study_anli.py --design-only --table replay/anli/design.csv
python scripts/assemble_anli.py replay/anli --output results/anli/scores.csv
python scripts/study_anli.py
```
