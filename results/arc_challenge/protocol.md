# ARC-Challenge audit protocol

Score every four-option question in the pinned ARC-Challenge test split in source order. Preserve the order of the four options and display them as A, B, C and D. Grade by the position of the original answer label, including numeric labels. The cohort manifest records the source index and hashes of the question and each option.

Use the pinned Qwen2.5-7B-Instruct BF16 collector, last-token logits for IDs 32 to 35, batch size 16 and the standard CUDA backend with a 12 GiB GPU weight placement budget. Select the highest logit and apply the saved ANLI R2 policy to the top-two gap (cutoff 4.75). Freeze scores, offers and seed-42 draws before opening answers. The target is 20% error with 95% directional confidence, 15% and 25% design scenarios, 80% power, one label per request and a hard ceiling of 500.

Compare the smallest qualifying cap, the lowest modeled cost, the prior two-stage workflow at the minimum qualifying single-stage cap, and its best fixed block. Use prefixes of one random order. Record issued labels, requests and verdicts before grading the eligible census. Report modeled improvement of at least 2% over the prior two-stage design only if both directional powers qualify. Keep observed costs separate.
