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

The local run switched devices while the desktop was in use. Before
freezing the design, rescore every CPU-scored row on the RTX 5080 in
its original batch of eight. Use the GPU logits for the audit and record
how many choices and cutoff decisions changed.

## Reproduce the audit

The cohort is frozen. Collection stopped before the predictions were saved.
Once the logits and reference grades are available, the following CPU replay
will redraw the sample, restart a partially returned first request and
compare the three verdicts.

```bash
python -m pip install .
python scripts/study_hellaswag.py replay
```

To repeat collection, install the models extra, PyArrow and a CUDA build
of PyTorch. Run `scripts/collect_hellaswag.py freeze` and then `collect`.
The collector streams the pinned BF16 weights from disk, caps PyTorch's
GPU allocator at 2.25 GiB and saves progress after each batch of eight.
Repeating `collect` resumes from the last completed batch. Use `--device cpu`
to continue with two CPU threads. The collection report records each device
and its last row in scoring order. Use `--stop-after` to repeat a segment.
Run `scripts/study_hellaswag.py` with `design`, `audit` and `census` in
that order. The dataset supplies the reference ending for each issued row.
