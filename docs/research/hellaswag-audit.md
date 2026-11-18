# Two-stage audit on HellaSwag

## Frozen protocol

Use Qwen2.5-7B-Instruct at `a09a35458c702b33eeacc393d103063234e8bc28`
in BF16. Take at most 1,500 validation sources in SHA-256 order, choosing
one row per source by its hashed source, example index and file position.
Keep the four endings in source order and score the single A/B/C/D tokens
in one forward pass. Apply the saved ANLI candidate gap cutoff of 4.75
to the difference between the two highest logits. Keep its 20% error
target and both 95% one-sided limits. Pin the dataset to
`218ec52e09a7e7462a5400043bb9a69a41d06b76`. Collection reads only IDs,
contexts and endings. [HellaSwag dataset](https://huggingface.co/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76)

Before opening validation answers, choose the smallest single-stage cap
up to 500 that reaches 80% directional power at both 15% and 25% error.
Compare interim counts from 50 in steps of 25 below the cap, plus the
integer part of two thirds of the cap. Keep the qualifying checkpoint with
the lowest worst-scenario cost at one label per request, breaking ties by
the smaller interim count. If there is no qualifying design, record that
outcome. Freeze the eligible cohort and the shared seed-42 draw. Compare
bulk, the best constant 1.1 block size and the two-stage rule at this cap.
For all three record issued labels, requests and verdict. Open the eligible
census last. Also report costs at zero and five labels per request. The
milestone threshold is a 5% reduction in worst-scenario expected cost
against the best 1.1 block size, with the same power requirement and cap.
