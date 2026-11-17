"""Collect replayable confidence signals with an optional local model."""

import json
from importlib.metadata import version
from itertools import chain, islice, repeat
from time import perf_counter

from .features import parse_confidence


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def read_prompts(path):
    prompts = {}
    with path.open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                item_id, prompt = record["id"], record["prompt"]
                if (
                    not isinstance(item_id, str)
                    or not item_id.strip()
                    or item_id in prompts
                    or not isinstance(prompt, str)
                    or not prompt.strip()
                ):
                    raise ValueError(
                        "each record needs a unique id and nonempty prompt"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"prompt JSONL line {line_no}: {exc}") from exc
            prompts[item_id] = prompt
    if not prompts:
        raise ValueError("at least one prompt is required")
    return prompts


def batches(items, size):
    items = iter(items)
    while batch := list(islice(items, size)):
        yield batch


def collect_responses(
    prompts,
    backend,
    *,
    max_new_tokens=32,
    p_true=False,
    verbalized=False,
    samples=0,
    batch_size=1,
    serial_greedy=False,
):
    if not prompts or max_new_tokens < 1 or samples < 0:
        raise ValueError(
            "collect needs prompts, a positive token budget, and a nonnegative sample count"
        )
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("collect needs a positive batch size")
    timings = {}

    def invoke(name, inputs, *args, **options):
        method = "p_true" if name == "p_true" else "generate"
        start = perf_counter()
        if len(inputs) == 1:
            result = [getattr(backend, method)(inputs[0], *args, **options)]
        else:
            generate_batch = getattr(backend, f"{method}_batch", None)
            if generate_batch is None:
                raise ValueError(f"the backend does not support batched {name}")
            result = generate_batch(inputs, *args, **options)
        if len(result) != len(inputs):
            raise ValueError(f"the backend returned the wrong number of {name} results")
        stats = timings.setdefault(name, {"calls": 0, "batches": 0, "seconds": 0.0})
        stats["calls"] += len(inputs)
        stats["batches"] += 1
        stats["seconds"] += perf_counter() - start
        return result

    collected = []
    greedy_size = 1 if serial_greedy else batch_size
    for group in batches(prompts.items(), batch_size):
        ids, inputs = zip(*group, strict=True)
        greedy = (
            result
            for block in batches(inputs, greedy_size)
            for result in invoke("greedy", block, max_new_tokens, tokens=True)
        )
        rows = []
        for item_id, (text, tokens) in zip(ids, greedy, strict=True):
            record = {"id": item_id, "text": text}
            if tokens:
                record["tokens"] = tokens
            rows.append(record)
        if p_true or verbalized:
            reviews = [
                f"Question or task:\n{prompt}\n\nProposed answer:\n{record['text']}\n\n"
                for prompt, record in zip(inputs, rows, strict=True)
            ]
        if p_true:
            values = invoke(
                "p_true",
                [
                    review
                    + "Is the proposed answer correct? Reply with A for correct or B for incorrect."
                    for review in reviews
                ],
            )
            for record, value in zip(rows, values, strict=True):
                if value is not None:
                    record["p_true"] = value
        if verbalized:
            values = invoke(
                "verbalized_confidence",
                [
                    review
                    + "Give your probability that the proposed answer is correct. Reply only with a number between 0 and 1 or a percentage."
                    for review in reviews
                ],
                12,
            )
            for record, (response, _) in zip(rows, values, strict=True):
                record["verbalized_response"] = response
                try:
                    record["verbalized_confidence"] = parse_confidence(response.strip())
                except ValueError:
                    pass
        if samples:
            sample_inputs = chain.from_iterable(
                repeat(prompt, samples) for prompt in inputs
            )
            answers = (
                text
                for draw in batches(sample_inputs, batch_size)
                for text, _ in invoke("sample", draw, max_new_tokens, sampled=True)
            )
            for record, draw in zip(rows, batches(answers, samples), strict=True):
                record["samples"] = draw
        collected.extend(rows)
    return collected, timings


class TransformersBackend:
    def __init__(
        self, model, revision, *, device, seed, local_files_only, gpu_memory=None
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "install caat-annotation[models] to run caat collect"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch installation")
        if gpu_memory and device != "cuda":
            raise ValueError("--gpu-memory requires CUDA")
        self.torch = torch
        self.device = device
        torch.manual_seed(seed)
        options = {"revision": revision, "local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model, **options)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        memory = {0: gpu_memory, "cpu": "16GiB"} if gpu_memory else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto" if device == "cuda" else "cpu",
            max_memory=memory,
            dtype="auto",
            use_safetensors=True,
            **options,
        ).eval()
        self.metadata = {
            "model": model,
            "revision": getattr(self.model.config, "_commit_hash", None) or revision,
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "seed": seed,
            "torch": version("torch"),
            "transformers": version("transformers"),
        }

    def format_prompt(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def inputs(self, prompt):
        formatted = self.format_prompt(prompt)
        batch = self.tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
        return formatted, batch.to(self.device)

    def batch_inputs(self, prompts):
        if len(prompts) == 1:
            prompt, batch = self.inputs(prompts[0])
            return [prompt], batch
        formatted = [self.format_prompt(prompt) for prompt in prompts]
        batch = self.tokenizer(
            formatted, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        return formatted, batch.to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def generate_tokens(
        self, batch, max_new_tokens, *, sampled=False, tokens=False, **options
    ):
        with self.torch.inference_mode():
            return self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=sampled,
                temperature=0.7 if sampled else 1.0,
                top_p=0.9 if sampled else 1.0,
                top_k=0 if sampled else 50,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=tokens,
                **options,
            )

    def generate(self, prompt, max_new_tokens, *, sampled=False, tokens=False):
        return self.generate_batch(
            [prompt], max_new_tokens, sampled=sampled, tokens=tokens
        )[0]

    def generate_batch(self, prompts, max_new_tokens, *, sampled=False, tokens=False):
        _, batch = self.batch_inputs(prompts)
        generated = self.generate_tokens(
            batch, max_new_tokens, sampled=sampled, tokens=tokens
        )
        suffix = generated.sequences[:, batch["input_ids"].shape[1] :]
        token_ids = suffix.tolist()
        logprobs = repeat(())
        if tokens:
            selected = [
                logits.float()
                .log_softmax(dim=-1)
                .gather(1, ids.to(logits.device).unsqueeze(1))
                .squeeze(1)
                for ids, logits in zip(
                    suffix.unbind(dim=1), generated.logits, strict=True
                )
            ]
            logprobs = iter(self.torch.stack(selected, dim=1).tolist())
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
        special = set(self.tokenizer.all_special_ids)
        responses = []
        for ids, scores in zip(token_ids, logprobs):
            end = next(
                (i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids)
            )
            saved = []
            if tokens:
                saved = [
                    {"text": self.decode([token]), "logprob": logprob}
                    for token, logprob in zip(ids[:end], scores[:end], strict=True)
                    if token not in special
                ]
            responses.append((self.decode(ids[:end]), saved))
        return responses

    def p_true(self, prompt):
        return self.p_true_batch([prompt])[0]

    def choice_tokens(self, formatted, prefix, labels=("A", "B")):
        choices = []
        for choice in labels:
            continuation = self.tokenizer.encode(
                formatted + choice, add_special_tokens=False
            )
            if len(continuation) != len(prefix) + 1 or continuation[:-1] != prefix:
                raise ValueError("answer choices must be single continuation tokens")
            choices.append(continuation[-1])
        if len(set(choices)) != len(choices):
            raise ValueError("answer choices must use distinct tokens")
        return choices

    def p_true_batch(self, prompts):
        formatted, batch = self.batch_inputs(prompts)
        choices = [
            self.choice_tokens(
                text, [token for token, keep in zip(ids, mask, strict=True) if keep]
            )
            for text, ids, mask in zip(
                formatted,
                batch["input_ids"].tolist(),
                batch["attention_mask"].tolist(),
                strict=True,
            )
        ]
        with self.torch.inference_mode():
            if len(prompts) == 1:
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[
                    :, -1
                ]
            else:
                # Use the model's padding-aware generation setup for the next-token logits.
                logits = self.generate_tokens(
                    batch, 1, tokens=True, use_cache=False
                ).logits[0]
            choice_ids = self.torch.tensor(choices, device=logits.device)
            return logits.gather(1, choice_ids).float().softmax(dim=-1)[:, 0].tolist()


def write_responses(path, records):
    data = "\n".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data + "\n", encoding="utf-8", newline="\n")
