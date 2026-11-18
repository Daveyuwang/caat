# Collect model responses

Supply one JSONL object per prompt.

```json
{"id":"qa1","prompt":"What is the capital of France? Answer in one word."}
```

## Transformers

Install `.[models]` from a checkout. Use `--model` and `--revision` for a chat-template model or use the default Qwen2.5-7B-Instruct. Limit GPU weight placement with `--gpu-memory 10GiB`.

```bash
caat collect prompts.jsonl --output replay/responses.jsonl --p-true --verbalized --samples 5 --metadata replay/collection.json
caat features replay/responses.jsonl --include-output-key --feature mean_token_logprob --feature p_true --feature verbalized_confidence --feature normalized_sample_agreement --output replay/scores.csv
```

`--p-true` queries the model about its own answer. `--verbalized` requests a stated confidence. `--samples N` collects additional answers for agreement scores.

## Chat Completions API

Pass the server’s API root, including `/v1` when required. For an authenticated endpoint, use `--api-key-env NAME`.

```bash
caat collect prompts.jsonl --base-url http://127.0.0.1:8000/v1 --model qwen-smoke --p-true --samples 3 --output replay/api-responses.jsonl --metadata replay/api-collection.json
```

For a text-only endpoint, use `--no-logprobs` with verbalized confidence or sampled agreement. `--metadata` records actual HTTP requests and how many rows have each probability signal. [Feature definitions](features.md)
