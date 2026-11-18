# Frozen BoolQ comparison

The dataset is BoolQ revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`. Passage and question hashes use lowercasing by Unicode casefold and collapsed whitespace. Retain the smallest hashed question per passage and give validation passages precedence. The manifest fixes 2,938 evaluation passages from 3,270 rows and 1,520 training passages from 7,200 eligible groups (9,427 original rows). Seed 20261117 assigns 400 training labels to scorer design, 160 label-free examples to reference quantiles, and 960 rows to the calibration pool. The preparation code reads only passage and question. [Manifest](../../results/boolq/manifest.csv) · [Preparation](../../scripts/prepare_boolq.py)

Use BF16, left padding, no sampling and the pinned tokenizer's user chat template with the assistant generation prompt, thinking disabled. The generator is Qwen2.5-7B-Instruct at `a09a35458c702b33eeacc393d103063234e8bc28`. Its user message is the following template.

```text
Passage: {passage}
Question: {question}
Answer with exactly Yes or No.
```

Score the distinct next-token strings Yes and No. Choose Yes on a tie. Save the winning token's full-vocabulary log probability, the absolute difference in the two logits, and the two-choice softmax probability of the winner. The grader compares the predicted boolean with the dataset's answer. The verifier uses Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc` with the following user message and the normalized next-token probability of True among True and False.

```text
Passage: {passage}
Question: {question}
Proposed answer: {answer}
Reply with exactly True if the proposed answer is correct or False if it is incorrect.
```

Choose the best native single and the best of the native and verifier singles by normalized AURC gain on the 400 design labels. Fit standardized logistic regression with C = 0.1 on the winner log probability, gap and verifier score. Benchmark every score on the same outputs. At 75% coverage, average ties at the boundary, round the number accepted up and draw 5,000 paired passage bootstraps with seed 20261117. Require at least two points lower accepted error versus the design-selected single with an upper bound below zero. The extra-call claim also requires a reduction against the best native single. For the joint claim use two one-sided 97.5% upper bounds. Report the individual two-sided 95% intervals.

For the native winner confidence, compare fixed thresholds 0.50 to 0.95 by 0.05 against ten quantiles of the 160 reference scores. For the learned score use its own reference quantiles. Each fit tests frozen candidates with 5% LTT Bonferroni at a 20% accepted-error target. Shuffle the sorted calibration pool with Python seeds 0 to 199 and take nested prefixes of 320 and 480. Seed 42 is the primary fit for the independent one-sided 95% holdout audit. Record the accepted IDs, coverage, flips between adjacent paired fits and fits with no cutoff. A stability gain requires 25% fewer flips at equal cost, at most two points of lost coverage or added empty fits, and a nonempty passing audit. Compare reference/320 to numeric/480 using paired one-sided 95% upper limits for adverse changes in flips, empty fits and coverage loss, with a two-point margin for each. Resample the 100 seed pairs 5,000 times with seed 20261117. Uncertainty about refits is conditional on this cohort. Retain all outcomes, including a failed gate. [BoolQ data](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)
