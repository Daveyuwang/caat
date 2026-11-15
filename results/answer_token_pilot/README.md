# Jigsaw answer-token pilot

## Comparison

Does the probability of the saved label from a single-token A/B question
predict its correctness better than Qwen's saved self-reported confidence?
Use the existing logistic-regression trust score as the supervised reference.

- The pilot uses the [saved Jigsaw predictions](../trust_scorer/jigsaw_lr_predictions.csv).
  The file matches the one used from course checkout `78503f6`. The runner
  samples 400 rows without replacement with `random.Random(42).sample` before
  inference. All methods score the saved Qwen label against the gold label.
- The runner uses [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28)
  in BF16. The course code names the same model and leaves the local
  checkpoint revision unspecified.
- The A/B score normalizes the next-token logits over the two answer tokens.
  The runner scores the original label with both A/B mappings and averages
  the two probabilities.
- The comparison uses Brier score and risk–coverage for the raw A/B score,
  the saved numeric self-report, and the saved calibrated trust score. Each
  threshold accepts or rejects tied scores together. For the comparison at
  75% coverage, we compute expected risk under uniform tie-breaking. A
  2-point drop at 75% coverage means six fewer expected errors among 300
  accepted labels.
- A standalone follow-up against the self-report baseline is warranted if the
  raw answer-token score has lower Brier loss and at least 2 percentage points
  lower risk at 75% coverage. Report both outcomes separately.
- The pilot uses one RTX 5080 with host-memory offload. It compares the same
  saved labels on a sample drawn from the original train, development, and
  test splits.

## Reproduce

Use an environment with CUDA-enabled PyTorch. The recorded run used Python
3.12, PyTorch 2.13.0 with CUDA 13.2, and Transformers 5.16.1. The runner
downloads the pinned checkpoint on first use. Set `HF_HUB_OFFLINE=1` to use
a cached copy.

```bash
python -m pip install "transformers==5.16.1" accelerate "matplotlib>=3.7"
python scripts/run_answer_token_pilot.py --batch-size 16 --gpu-memory 10GiB --output-dir replay/answer_token_pilot
```

[Per-example scores](scores.csv) include both option-order probabilities and
the original label. [Run metadata and metrics](metrics.json) record the model
revision, timing, and GPU memory. The
[risk–coverage plot](risk_coverage.png) shows attainable threshold points
connected by lines. Open circles at 75% coverage show the expected error
rates under uniform tie-breaking.

## Results

Run on 2026-11-14. The original Qwen labels were correct on 319 of 400 rows
(79.75%). The sample included 31 rows with the gold label toxic. The pilot
used 37.34 seconds of inference on one RTX 5080, with 10.93 GiB peak PyTorch
reserved GPU memory and some weights stored in host memory. API cost was $0.

| Score | Brier score ↓ | Error rate at 75% coverage ↓ |
| --- | ---: | ---: |
| A/B answer-token probability | 0.2217 | 19.33% |
| Saved self-reported confidence | 0.1569 | 14.88% |
| Original trust scorer | 0.0380 | 0.56% |

The answer-token score missed both follow-up criteria. Among 295 rows whose
averaged answer-token probability was at least 0.95, 58 labels were incorrect
(19.66%). Swapping the A/B mapping changed the probability of the same
saved label by 20.3 percentage points on average in absolute value. The
trust scorer is a supervised reference from the original project, whose
Jigsaw sample spans the original training, development, and test splits.

Verification matched all 400 example IDs, text hashes, and original labels to
the original annotation export. Both metrics were independently recomputed
from the per-example CSV.
