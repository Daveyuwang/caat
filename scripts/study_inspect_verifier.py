"""Replay the archived Inspect features with the same score comparison as CAAT 0.4."""

import csv
import argparse

import numpy as np

from common import ROOT, read_csv, write_report
from study_routing_stability import score_vectors
from study_combined_scores import paired_results
from caat.scores import _ScoreRanking


def table(path):
    with (ROOT / "replay" / path).open(encoding="utf-8", newline="") as src:
        records = list(csv.DictReader(src))
    result = {r["id"]: r for r in records}
    assert len(result) == len(records)
    return result


def prepare(path):
    features = table("inspect-token-baseline/joined.csv")
    design = table("inspect-pilot/design.csv")
    holdout = table("inspect-pilot/holdout.csv")
    assert not (design.keys() & holdout.keys())
    for i, split in (design | holdout).items():
        assert (features[i]["output_sha256"], features[i]["triviaqa_alias"]) == (
            split["output_sha256"],
            split["correct"],
        )
    second = table("inspect-tie-confirmation/joined.csv")
    tokens = table("inspect-tie-confirmation/token/external-scores.csv")
    assert not (features.keys() & second.keys())
    assert tokens.keys() == second.keys()
    for i, row in second.items():
        assert all(
            row[k] == tokens[i][k] for k in ("eval_id", "epoch", "output_sha256")
        )
        row.update(tokens[i])

    rows = []
    for role, records in (
        ("design", [features[i] for i in sorted(design)]),
        ("original", [features[i] for i in sorted(holdout)]),
        ("second", list(second.values())),
    ):
        for r in records:
            rows.append(
                {
                    **{
                        k: r[k]
                        for k in (
                            "id",
                            "output_sha256",
                            "p_true",
                            "mean_token_logprob",
                            "sequence_logprob",
                            "triviaqa_alias",
                        )
                    },
                    "role": role,
                }
            )
    write_report(path, rows)


def study(path):
    data = read_csv(path)
    design = [r for r in data if r["role"] == "design"]
    for title in ("original", "second"):
        rows = design + sorted(
            (r for r in data if r["role"] == title), key=lambda r: r["id"]
        )
        names = ("sequence_logprob", "mean_token_logprob", "p_true")
        x = np.array([[float(r[col]) for col in names] for r in rows])
        y = np.array([r["triviaqa_alias"] == "1" for r in rows])
        d = list(range(len(design)))
        h = list(range(len(design), len(rows)))
        ranks = []
        for kind, selected, scores, _ in score_vectors(x, y, d, names):
            print(title, kind, selected)
            ranks.append(_ScoreRanking.build(scores[h], ~y[h]))
        token_only = _ScoreRanking.build(x[h, 0], ~y[h])
        print("Design-selected single vs fit", paired_results([ranks], len(h))[1])
        print(
            "Sequence log probability vs fit",
            paired_results([[token_only, ranks[1]]], len(h))[1],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Rebuild the table from the original local Inspect sidecars",
    )
    args = parser.parse_args()
    path = ROOT / "results/routing_stability/inspect_verifier.csv"
    if args.prepare:
        prepare(path)
    study(path)
