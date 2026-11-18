"""Freeze the four-option ARC-Challenge test cohort and score before grading."""

import argparse

from collect_choice_logits import collect_choices
from common import ROOT, read_csv, write_report
from prepare_boolq import digest

OUT = ROOT / "results/arc_challenge"
REVISION = "210d026faf9955653af8916fad021475a3f00453"
CHOICES = "ABCD"


def dataset_file():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "allenai/ai2_arc",
        "ARC-Challenge/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )


def prompts():
    import pyarrow.parquet as pq

    source = pq.read_table(dataset_file(), columns=["id", "question", "choices"])
    rows = []
    for i, row in enumerate(source.to_pylist()):
        choices = row["choices"]["text"]
        if len(choices) != 4:
            continue
        meta = dict(id=row["id"], source_row=str(i), question_sha256=digest(row["question"]))
        lines = [row["question"]]
        for c, option in zip(CHOICES, choices, strict=True):
            meta[c + "_sha256"] = digest(option)
            lines.append(f"{c}: {option}")
        lines.append("Reply with exactly A, B, C or D.")
        rows.append((meta, "\n".join(lines)))
    if len(source) != 1172 or len({m["id"] for m, _ in rows}) != len(rows) or len(rows) != 1165:
        raise ValueError("pinned ARC cohort changed")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "collect"))
    args = parser.parse_args()
    rows = prompts()
    manifest = [m for m, _ in rows]
    path = OUT / "cohort.csv"
    if args.stage == "freeze":
        OUT.mkdir(parents=True, exist_ok=True)
        write_report(path, manifest)
    else:
        if manifest != read_csv(path):
            raise ValueError("ARC source differs from frozen cohort")
        collect_choices(rows, CHOICES, OUT, REVISION, checkpoint=ROOT / "replay/arc-collection.json")
