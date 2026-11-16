"""Compare candidate saved-response scores on the existing design splits."""

import argparse
from math import ceil, log
from operator import itemgetter
from pathlib import Path
from statistics import mean, pstdev

from caat.features import SavedResponse, token_logprobs
from caat.metrics import threshold_points
from caat.policy import split_rows

from common import ARCHIVES, ROOT, SEED, read_csv, read_json, risk_at_count, write_report


def token_logprob_stddev(response):
    return -pstdev(response.logprobs)


def low_probability_token_share(response):
    return -sum(value < log(0.1) for value in response.logprobs) / len(response.logprobs)


def last_tenth_logprob(response):
    return mean(response.logprobs[-ceil(len(response.logprobs) / 10):])


def distinct_bigram_ratio(response):
    words = response.text.split()
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


CANDIDATES = {
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
        states = read_json(source / "sources" / f"{model}_scenario_state.json")["request_states"]
        completions = {state["instance"]["id"].removeprefix("id"): state["result"]["completions"][0]
                       for state in states}
        responses = []
        for row in design:
            completion = completions[row["id"]]
            responses.append(SavedResponse(completion["text"], completion["tokens"],
                                           token_logprobs(completion, row["id"])))
        slots = ceil(len(design) * 0.75)
        errors = [row[graders[0]] == "0" for row in design]
        baseline = risk_at_count(threshold_points(zip((float(row["mean_token_logprob"]) for row in design),
                                                     errors, strict=True)), slots)
        for name, scorer in CANDIDATES.items():
            values = (scorer(response) for response in responses)
            risk = risk_at_count(threshold_points(zip(values, errors, strict=True)), slots)
            yield {"archive": archive, "score": name, "grader": graders[0], "design_rows": len(design),
                   "accepted": slots, "baseline_risk": baseline, "risk": risk, "risk_delta": risk - baseline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replay")
    parser.add_argument("--output", type=Path, default=ROOT / "replay" / "feature_screen.csv")
    args = parser.parse_args()
    rows = list(screen(args.replay_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, rows)
    for name in CANDIDATES:
        wins = sum(row["score"] == name and row["risk_delta"] < -1e-12 for row in rows)
        print(f"{name}: {wins}/{len(ARCHIVES)} design archives improved")


if __name__ == "__main__":
    main()
