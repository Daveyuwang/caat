# Collect responses from a local model

Install the optional local backend with `python -m pip install '.[models]'`
from a checkout. Choose a PyTorch build that supports your host and GPU.
Input JSONL records have a unique string `id` and a nonempty `prompt`.

```json
{"id":"qa1","prompt":"What is the capital of France? Answer in one word."}
```

```bash
caat collect prompts.jsonl --output replay/responses.jsonl --p-true --verbalized --samples 5 --metadata replay/collection.json
caat features replay/responses.jsonl --feature mean_token_logprob --feature p_true --feature verbalized_confidence --feature normalized_sample_agreement --output replay/scores.csv
```

The default model is a pinned revision of Qwen2.5-7B-Instruct. Use
`--model` and `--revision` to select another chat-template model with
safetensors weights. Use `--local-files-only` for a cached local run.
`--device cuda` requires a CUDA-enabled PyTorch build. `--device cpu`
can run smaller cached models. `--gpu-memory 10GiB` permits CUDA weight
offload to RAM.

CAAT saves the greedy answer and its token log probabilities. `--p-true`
makes one extra forward pass and normalizes the logits for the two choices
in a correctness check. `--verbalized` makes a separate generation call.
It saves the raw check response and also saves a probability if that response
parses. `--samples N` collects N additional answers per prompt using
temperature 0.7 and top-p 0.9. Generation uses `--seed`.

`--metadata` writes the model version, seed, device, model loading time,
and separate call counts and wall times for each collection method.
The JSONL can be scored later using the standard-library installation.
[Saved response fields](features.md)
