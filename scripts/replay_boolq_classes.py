"""Report the BoolQ answer-aware scorer at one global 75% acceptance budget."""

import json

from caat.scorer import fit_scorer
from caat.scores import ScoredRow, compare_scores
from common import ROOT, read_csv


def replay():
    partitions = {"design": [], "holdout": []}
    for row in read_csv(ROOT / "results/boolq/scores.csv"):
        if row["role"] not in partitions:
            continue
        features = (
            float(row["winner_logprob"]),
            float(row["gap"]),
            float(row["answer"] == "Yes"),
        )
        partitions[row["role"]].append(
            ScoredRow(
                row["id"],
                features,
                (row["correct"] == "1",),
                predicted_class=row["answer"],
            )
        )
    model = fit_scorer(("winner_logprob", "gap", "answer_yes"), partitions["design"])
    target = [
        ScoredRow(
            r.id,
            (model.score(r.scores),),
            r.outcomes,
            predicted_class=r.predicted_class,
        )
        for r in partitions["holdout"]
    ]
    report = compare_scores(
        target,
        names=["answer_aware"],
        graders=["correct"],
        coverages=[0.75],
        split="all",
        bootstrap=0,
    )
    return report["graders"]["correct"]["scores"]["answer_aware"]["budgets"][0]


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
