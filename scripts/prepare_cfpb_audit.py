"""Prepare a paired audit from the public CFPB routing encoder export."""

import argparse
import csv
import json
from math import isclose, isfinite
from pathlib import Path

from common import fetch


REVISION = "a8b059421bb7eb670f29ce93a77fcd3d14b5c29f"
BASE_URL = f"https://raw.githubusercontent.com/ryansloan/cfpb-routing/{REVISION}/results/followup"
TIER = "encoder"
GRADER_FIELDS = ("complaint_id", "output_text", "correct")
RISK_TARGET = 0.10


def unique_top_label(output_text):
    scores = json.loads(output_text)
    if (
        not isinstance(scores, dict)
        or not scores
        or any(
            type(score) not in (int, float)
            or not isfinite(score)
            or not 0 <= score <= 1
            for score in scores.values()
        )
    ):
        raise ValueError("encoder output must be a nonempty label-to-score object")
    best = max(scores.values())
    labels = [name for name, score in scores.items() if score == best]
    return labels[0] if len(labels) == 1 else None


def prepare(output):
    cache = output / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    archive = fetch(f"{BASE_URL}/encoder_eval.csv", cache / "encoder_eval.csv")
    stats = json.loads(
        fetch(f"{BASE_URL}/stats.json", cache / "stats.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = stats["encoder"]["hybrid"]["cheap"]["frozen"]
    threshold = frozen["threshold"]
    with archive.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["tier"] == TIER]
    if len(rows) != stats["n_complaints"] or len(
        {row["complaint_id"] for row in rows}
    ) != len(rows):
        raise ValueError("unexpected number of unique complaints in the selected tier")

    archived, regraded, discordant = [], [], []
    accepted = native_errors = upgraded_errors = 0
    for row in rows:
        item_id, output_text = row["complaint_id"], row["output_text"]
        before = row["prediction"] == row["gold"]
        if row["correct"] != str(int(before)):
            raise ValueError(
                f"published label differs from exact match for complaint {item_id}"
            )
        after = unique_top_label(output_text) == row["gold"]
        auto = float(row["confidence"]) >= threshold
        accepted += auto
        native_errors += auto and not before
        upgraded_errors += auto and not after
        if before != after:
            discordant.append(
                {"id": item_id, "accepted": auto, "archived": before, "regraded": after}
            )
        archived.append((item_id, output_text, int(before)))
        regraded.append((item_id, output_text, int(after)))

    if not isclose((len(rows) - accepted) / len(rows), frozen["pct_escalated"]):
        raise ValueError("routing mask differs from the published escalation rate")
    for name, outcomes in (("archived", archived), ("regraded", sorted(regraded))):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(GRADER_FIELDS)
            writer.writerows(outcomes)

    reference = {
        "source_revision": REVISION,
        "tier": TIER,
        "threshold": threshold,
        "risk_target": RISK_TARGET,
        "total": len(rows),
        "accepted": accepted,
        "archived_errors": native_errors,
        "regraded_errors": upgraded_errors,
        "archived_meets_target": accepted > 0
        and native_errors / accepted <= RISK_TARGET,
        "regraded_meets_target": accepted > 0
        and upgraded_errors / accepted <= RISK_TARGET,
        "discordant": sorted(discordant, key=lambda row: row["id"]),
    }
    (output / "reference.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} complaints in {output}. Frozen threshold {threshold}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("replay/cfpb"))
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
