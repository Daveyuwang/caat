# Predicted classes and verification on ANLI

Freeze one hypothesis per normalized premise at ANLI revision `8e4813d81f46d313dac7892e1c28076917cfcdf9`. Use the smallest SHA-256 of the normalized hypothesis, breaking ties by UID. Give test R1 priority, then test R3, then train R1. Sort by premise hash and shuffle each split with Python seed 20261118. Use 400 design labels, 160 ungraded reference rows and 480 calibration labels from train R1. This yields 846 R1 primary and 1,128 R3 transport groups. Publish the manifest before reading either test's grades. [ANLI dataset](https://huggingface.co/datasets/facebook/anli/tree/8e4813d81f46d313dac7892e1c28076917cfcdf9) · [Preparation](../../scripts/prepare_anli.py)

Use the pinned Qwen models, BF16, and chat template from the [BoolQ protocol](boolq-protocol.md). Score the generator's single next-token choices A, B and C. Use the largest logit, breaking ties in listed order. The semantic labels in order are entailment, neutral and contradiction. Use this user message, substituting the original text.

```text
Premise: {premise}
Hypothesis: {hypothesis}
Does the premise entail the hypothesis, leave it neutral, or contradict it?
A: entailment
B: neutral
C: contradiction
Reply with exactly A, B or C.
```

On the 400 design rows only, rotate the semantic labels to neutral, contradiction, entailment and compare decoded semantic answers. Charge this pass separately. For verification use the fixed original answer.

```text
Premise: {premise}
Hypothesis: {hypothesis}
Proposed relation: {semantic_label}
Reply with exactly True if the proposed relation is correct or False if it is incorrect.
```

Use standardized logistic regression with C = 0.1. Compare random ordering, predicted class alone, top-two gap, winning full-vocabulary log probability plus gap, that pair plus the semantic class, and the three raw option logits. Encode classes with full one-hot indicators in sorted order. Select the strongest nonrandom cheap arm by out-of-fold normalized AURC gain on design (five shuffled stratified folds by correctness, seed 42, ties in the listed order). Add P(True) to that exact arm and report its marginal gain. Also add P(True) to the class-aware pair. Keep answers fixed across arms. Record the design-only paired discordances and achievable precision before opening test grades.

The primary endpoint is accepted error at 75% global coverage, averaged over boundary ties. Compare class-aware pair minus numeric pair and paid minus strongest cheap. Resample R1 premises in 5,000 paired draws with NumPy seed 20261118, reranking each draw. Report two-sided 95% intervals and one-sided 97.5% upper limits for the two primary claims. A gain needs at least two observed error points and an upper limit below zero. Report class-level coverage, errors, reviews and declines above ten coverage points. Report verifier costs per 1,000 calls. Treat R3 as a separate descriptive transport test.

Deploy the design-selected cheap score. Choose one global threshold using ten ungraded reference quantiles, LTT Bonferroni at 5%, and a 20% accepted-error target on the 480 calibration labels. Report accepted IDs, counts, and the one-sided exact 95% R1 upper bound. If no threshold passes, report all rows for review. Apply the identical policy on R3. Rank-budget estimates remain available when no cutoff qualifies.
