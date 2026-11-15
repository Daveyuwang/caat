"""Score the saved Jigsaw labels with Qwen2.5-7B-Instruct."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import transformers
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_saved_predictions import threshold_points


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SEED = 42
SAMPLE_SIZE = 400
RISK_COVERAGE = 0.75
METHODS = {
    "answer_token": ("Answer-token probability", "#08745C", "-"),
    "self_report": ("Self-reported confidence", "#B4511E", ":"),
    "trust_scorer": ("Original trust scorer", "#24518C", "--"),
}


def load_sample():
    path = ROOT / "results" / "trust_scorer" / "jigsaw_lr_predictions.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return random.Random(SEED).sample(rows, SAMPLE_SIZE)


def make_prompt(tokenizer, text, labels):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": (
                "Task: Classify whether text is toxic or non-toxic. "
                f"Select A for {labels[0]} or B for {labels[1]}. Respond with A or B."
            )},
            {"role": "user", "content": f"Text: {text}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def score_saved_labels(sample, batch_size, gpu_memory):
    model_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    token_ids = [tokenizer.encode(code, add_special_tokens=False) for code in ("A", "B")]
    if any(len(ids) != 1 for ids in token_ids) or token_ids[0] == token_ids[1]:
        raise ValueError("A and B must map to distinct single tokens")
    choices = torch.tensor([ids[0] for ids in token_ids], device="cuda")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: gpu_memory, "cpu": "12GiB"}, attn_implementation="sdpa",
        use_safetensors=True,
    ).eval()
    load_seconds = perf_counter() - model_start
    print(f"Model loaded in {load_seconds:.1f}s", flush=True)

    # Keep padding small while preserving the original sampled row order in the output.
    ranked_indices = sorted(range(len(sample)), key=lambda i: len(sample[i]["text"]))
    order_scores = [[0.0, 0.0] for _ in sample]
    inference_start = perf_counter()
    for order, labels in enumerate((("non-toxic", "toxic"), ("toxic", "non-toxic"))):
        for start in range(0, len(sample), batch_size):
            indices = ranked_indices[start : start + batch_size]
            prompts = [make_prompt(tokenizer, sample[i]["text"], labels) for i in indices]
            if start == 0:
                prefix_ids = tokenizer.encode(prompts[0], add_special_tokens=False)
                for code, token in zip(("A", "B"), token_ids, strict=True):
                    answer_ids = tokenizer.encode(prompts[0] + code, add_special_tokens=False)
                    if answer_ids != prefix_ids + token:
                        raise ValueError(f"{code} does not tokenize as one continuation token")
            batch = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
            output = model(**batch.to("cuda"), use_cache=False, logits_to_keep=1)
            # Preserve differences between probabilities that are close to one.
            logits = output.logits[:, -1].index_select(1, choices).double()
            probabilities = logits.softmax(dim=1).tolist()
            for i, probability in zip(indices, probabilities, strict=True):
                original_label = sample[i]["llm_label_str"]
                order_scores[i][order] = probability[labels.index(original_label)]
            if start == 0 or start + batch_size >= len(sample) or (start // batch_size + 1) % 10 == 0:
                print(f"Option order {order + 1}/2: {min(start + batch_size, len(sample))}/{len(sample)}", flush=True)

    torch.cuda.synchronize()
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_budget": gpu_memory,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "load_seconds": round(load_seconds, 2),
        "inference_seconds": round(perf_counter() - inference_start, 2),
        "batch_size": batch_size,
        "seed": SEED,
        "sample_size": len(sample),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "offloaded_modules": [name for name, device in model.hf_device_map.items() if device == "cpu"],
    }
    return order_scores, metadata


def risk_at_count(points, target):
    """Expected error rate with a uniform draw from the boundary tie group."""
    previous_accepted = previous_errors = 0
    for _, counts in points:
        if counts.accepted >= target:
            share = (target - previous_accepted) / (counts.accepted - previous_accepted)
            return (previous_errors + share * (counts.errors - previous_errors)) / target
        previous_accepted, previous_errors = counts.accepted, counts.errors
    raise ValueError("Target exceeds the sample size")


def summarize(rows, output_dir):
    figure = Figure(figsize=(7.3, 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.axvline(RISK_COVERAGE, color="#999999", linewidth=0.9, linestyle=":")
    summary = {}
    for method, (label, color, style) in METHODS.items():
        values = [(row[method], row["correct"]) for row in rows]
        points = threshold_points((score, not correct) for score, correct in values)
        summary[method] = {
            "brier_score": mean((score - correct) ** 2 for score, correct in values),
            "risk_at_75pct_coverage": risk_at_count(points, round(RISK_COVERAGE * len(rows))),
        }
        ax.plot(
            [counts.coverage for _, counts in points],
            [counts.error_rate for _, counts in points],
            label=label, color=color, linestyle=style, linewidth=1.8,
            marker="o" if method == "self_report" else None, markersize=5,
        )
        ax.scatter(
            RISK_COVERAGE, summary[method]["risk_at_75pct_coverage"],
            facecolors="white", edgecolors=color, s=32, zorder=4,
        )
    ax.set(
        xlabel="Coverage (share of examples accepted)", ylabel="Error rate among accepted labels",
        xlim=(0, 1.02), ylim=(0, None),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"Jigsaw (n={len(rows)})", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.24)
    figure.legend(frameon=False, loc="outside lower center", ncol=3)
    figure.savefig(output_dir / "risk_coverage.png", dpi=180)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory", default="10GiB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "answer_token_pilot")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    sample = load_sample()
    scores, metadata = score_saved_labels(sample, args.batch_size, args.gpu_memory)
    rows = [
        {
            "example_id": row["example_id"],
            "gold_label": row["gold_label"],
            "original_qwen_label": row["llm_label"],
            "correct": int(row["llm_label"] == row["gold_label"]),
            "answer_token": mean(option_scores),
            "answer_token_a_non_toxic": option_scores[0],
            "answer_token_a_toxic": option_scores[1],
            "self_report": float(row["llm_confidence_numeric"]),
            "trust_scorer": float(row["trust_score_calibrated"]),
        }
        for row, option_scores in zip(sample, scores, strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "run": metadata,
        "saved_qwen_accuracy": mean(row["correct"] for row in rows),
        "methods": summarize(rows, args.output_dir),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
