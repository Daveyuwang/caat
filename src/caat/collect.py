"""Collect replayable confidence signals with an optional local model."""

import json
from importlib.metadata import version
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


def collect_responses(
    prompts, backend, *, max_new_tokens=32, p_true=False, verbalized=False, samples=0
):
    if not prompts or max_new_tokens < 1 or samples < 0:
        raise ValueError(
            "collect needs prompts, a positive token budget, and a nonnegative sample count"
        )
    timings = {}

    def invoke(name, method, *args, **options):
        start = perf_counter()
        result = method(*args, **options)
        stats = timings.setdefault(name, {"calls": 0, "seconds": 0.0})
        stats["calls"] += 1
        stats["seconds"] += perf_counter() - start
        return result

    collected = []
    for item_id, prompt in prompts.items():
        text, tokens = invoke(
            "greedy", backend.generate, prompt, max_new_tokens, tokens=True
        )
        record = {"id": item_id, "text": text}
        if tokens:
            record["tokens"] = tokens
        review = f"Question or task:\n{prompt}\n\nProposed answer:\n{text}\n\n"
        if p_true:
            record["p_true"] = invoke(
                "p_true",
                backend.p_true,
                review
                + "Is the proposed answer correct? Reply with A for correct or B for incorrect.",
            )
        if verbalized:
            response, _ = invoke(
                "verbalized_confidence",
                backend.generate,
                review
                + "Give your probability that the proposed answer is correct. Reply only with a number between 0 and 1 or a percentage.",
                12,
            )
            record["verbalized_response"] = response
            try:
                record["verbalized_confidence"] = parse_confidence(response.strip())
            except ValueError:
                pass
        if samples:
            record["samples"] = [
                invoke(
                    "sample", backend.generate, prompt, max_new_tokens, sampled=True
                )[0]
                for _ in range(samples)
            ]
        collected.append(record)
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

    def inputs(self, prompt):
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = self.tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
        return formatted, batch.to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def generate(self, prompt, max_new_tokens, *, sampled=False, tokens=False):
        _, batch = self.inputs(prompt)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=sampled,
                temperature=0.7 if sampled else 1.0,
                top_p=0.9 if sampled else 1.0,
                top_k=0 if sampled else 50,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=tokens,
            )
        ids = generated.sequences[0, batch["input_ids"].shape[1] :].tolist()
        text = self.decode(ids)
        saved = []
        if tokens:
            special = set(self.tokenizer.all_special_ids)
            for token, logits in zip(ids, generated.logits, strict=True):
                if token not in special:
                    logprob = logits[0].float().log_softmax(dim=-1)[token].item()
                    saved.append({"text": self.decode([token]), "logprob": logprob})
        return text, saved

    def p_true(self, prompt):
        formatted, batch = self.inputs(prompt)
        prefix = batch["input_ids"][0].tolist()
        choices = []
        for choice in ("A", "B"):
            continuation = self.tokenizer.encode(
                formatted + choice, add_special_tokens=False
            )
            if len(continuation) != len(prefix) + 1 or continuation[:-1] != prefix:
                raise ValueError(
                    "P(True) needs A and B to be single continuation tokens"
                )
            choices.append(continuation[-1])
        if choices[0] == choices[1]:
            raise ValueError("P(True) needs distinct A and B tokens")
        with self.torch.inference_mode():
            logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[
                0, -1
            ]
            return logits[choices].float().softmax(dim=-1)[0].item()


def write_responses(path, records):
    data = "\n".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data + "\n", encoding="utf-8", newline="\n")
