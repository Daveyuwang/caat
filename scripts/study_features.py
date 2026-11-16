"""Compare candidate saved-response scores on the existing design splits."""

import argparse
from math import ceil, log
from operator import itemgetter
from pathlib import Path
from statistics import mean, pstdev

from caat.features import SavedResponse, token_logprobs
from caat.policy import split_rows
from caat.scores import ScoredRow, compare_scores

from common import (
    ARCHIVES,
    ROOT,
    SEED,
    read_csv,
    read_json,
    write_report,
)


def token_logprob_stddev(response):
    return -pstdev(response.logprobs)


def low_probability_token_share(response):
    return -sum(value < log(0.1) for value in response.logprobs) / len(
        response.logprobs
    )


def last_tenth_logprob(response):
    return mean(response.logprobs[-ceil(len(response.logprobs) / 10) :])


def distinct_bigram_ratio(response):
    words = response.text.split()
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


CANDIDATES = {
    "minimum_token_logprob": lambda response: min(response.logprobs),
    "negative_token_logprob_stddev": token_logprob_stddev,
    "negative_low_probability_token_share": low_probability_token_share,
    "last_tenth_logprob": last_tenth_logprob,
    "distinct_bigram_ratio": distinct_bigram_ratio,
}


def screen(replay):
    for archive, (folder, count, graders) in ARCHIVES.items():
        source = replay / folder
        rows = read_csv(source / "predictions_full.csv")
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise ValueError(f"{archive} needs {count} distinct prepared rows")
        design, _, _ = split_rows(rows, SEED, id_key=itemgetter("id"))
        model = archive.rsplit("_", 1)[-1]
        states = read_json(source / "sources" / f"{model}_scenario_state.json")[
            "request_states"
        ]
        completions = {
            state["instance"]["id"].removeprefix("id"): state["result"]["completions"][
                0
            ]
            for state in states
        }
        responses = []
        for row in design:
            completion = completions[row["id"]]
            responses.append(
                SavedResponse(
                    completion["text"],
                    completion["tokens"],
                    token_logprobs(completion, row["id"]),
                )
            )
        slots = ceil(len(design) * 0.75)
        scored = [
            ScoredRow(
                row["id"],
                (
                    float(row["mean_token_logprob"]),
                    *(score(response) for score in CANDIDATES.values()),
                ),
                (row[graders[0]] == "1",),
            )
            for row, response in zip(design, responses, strict=True)
        ]
        comparison = compare_scores(
            scored,
            names=("mean_token_logprob", *CANDIDATES),
            graders=graders[:1],
            split="all",
            bootstrap=0,
            coverages=(0.75,),
        )["graders"][graders[0]]["scores"]
        baseline = comparison["mean_token_logprob"]["budgets"][0]["error_rate"]
        for name in CANDIDATES:
            risk = comparison[name]["budgets"][0]["error_rate"]
            yield {
                "archive": archive,
                "score": name,
                "grader": graders[0],
                "design_rows": len(design),
                "accepted": slots,
                "baseline_risk": baseline,
                "risk": risk,
                "risk_delta": risk - baseline,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "replay" / "feature_screen.csv"
    )
    args = parser.parse_args()
    rows = list(screen(args.replay_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, rows)
    for name in CANDIDATES:
        wins = sum(row["score"] == name and row["risk_delta"] < -1e-12 for row in rows)
        print(f"{name}: {wins}/{len(ARCHIVES)} design archives improved")


if __name__ == "__main__":
    main()
