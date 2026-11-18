# Collect model responses

Input JSONL records have a unique string `id` and a nonempty `prompt`.

```json
{"id":"qa1","prompt":"What is the capital of France? Answer in one word."}
```

## Transformers

Install the optional local backend with `python -m pip install '.[models]'`
from a checkout. Choose a PyTorch build that supports your host and GPU.

```bash
caat collect prompts.jsonl --output replay/responses.jsonl --p-true --verbalized --samples 5 --metadata replay/collection.json
caat features replay/responses.jsonl --include-output-key --feature mean_token_logprob --feature p_true --feature verbalized_confidence --feature normalized_sample_agreement --output replay/scores.csv
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

`--batch-size N` groups up to N prompts or samples in each model call.
Greedy generation, correctness checks, verbalized confidence, and
sampling use this limit. The default of 1 preserves the original
per-prompt execution order. Use the same batch size and seed to repeat a
sampled run.

Batch shape can change model scores and greedy answers through
floating-point arithmetic. Use `--serial-greedy` to collect the primary
answer and its token log probabilities with the same per-prompt execution
as the serial run. The other stages still use `--batch-size N`.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

`--metadata` writes the model version, seed, device, batch size, greedy
mode, and model loading time. For each method, `calls` counts individual
prompt evaluations and `batches` counts model invocations. `seconds` is
the wall time spent on that method.
The JSONL can be scored later using the standard-library installation.
The output key lets CAAT match each score to a separate grade for the same response.
[Saved response fields](features.md)

## Chat Completions API

Use `--base-url` and `--model` with the standard CAAT installation. For
example, start a local server in a separate terminal. [Install vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/)

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen-smoke --host 127.0.0.1 --port 8000 --generation-config vllm
```

```bash
caat collect prompts.jsonl --base-url http://127.0.0.1:8000/v1 --model qwen-smoke --p-true --samples 3 --output replay/api-responses.jsonl --metadata replay/api-collection.json
caat features replay/api-responses.jsonl --include-output-key --feature mean_token_logprob --feature p_true --feature normalized_sample_agreement --output replay/api-scores.csv
```

Pass the API root, including `/v1` when the server uses it. CAAT appends
`/chat/completions`. For an authenticated endpoint, put the bearer token in
an environment variable and pass its name with `--api-key-env NAME`.
CAAT sends one prompt or sample per HTTP request. `--batch-size` must be 1.

CAAT saves token log probabilities when the endpoint returns a trace that matches
the visible answer. For `--p-true` it requests 20 alternatives at the first
token and normalizes the exact `A` and `B` probabilities when both are available.
A row without both probabilities has an empty `p_true` cell in `caat features`.
Use `caat route --review-missing-scores` to send those rows to review when
applying a P(True) cutoff to the batch.
[vLLM Chat Completions API](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)

If the endpoint rejects log probability options, CAAT keeps collecting text.
Use `--no-logprobs` to request text directly from a text-only endpoint.
`--verbalized` and `--samples N` still collect confidence checks and samples.
Choose the corresponding features to score those responses.

```bash
caat features replay/api-responses.jsonl --feature normalized_sample_agreement --output replay/api-scores.csv
```

For API runs, `--metadata` records `http_requests`, including rejected attempts
and retries, and `probability_rows`, the number of rows with each probability
signal. The per-method counters describe collection operations. Use
`http_requests` to count the actual requests sent to the server.
