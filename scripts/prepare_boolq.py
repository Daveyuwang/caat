"""Freeze BoolQ passage groups and row IDs without reading the answer column."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path
from random import Random

REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def dataset_file(split):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "google/boolq",
        f"data/{split}-00000-of-00001.parquet",
        revision=REVISION,
        repo_type="dataset",
    )


def freeze(output):
    import pyarrow.parquet as pq

    seen = set()
    manifest = []
    for split in ("validation", "train"):
        path = dataset_file(split)
        rows = pq.read_table(path, columns=["passage", "question"]).to_pylist()
        groups = {}
        for i, row in enumerate(rows):
            passage = digest(normalized(row["passage"]))
            question = digest(
                normalized(row["passage"]) + "\n" + normalized(row["question"])
            )
            if passage not in seen:
                groups[passage] = min(groups.get(passage, (question, i)), (question, i))
        seen.update(groups)
        ids = sorted((q, p, i) for p, (q, i) in groups.items())
        Random(20261117).shuffle(ids)
        print(f"{split}: {len(rows)} source rows, {len(ids)} eligible passage groups")
        for index, (q, p, i) in enumerate(ids):
            if split == "validation":
                role = "holdout"
            elif index < 400:
                role = "design"
            elif index < 560:
                role = "reference"
            elif index < 1520:
                role = "calibration_pool"
            else:
                continue
            manifest.append((q, p, split, i, role))
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest, lineterminator="\n")
        writer.writerow(["id", "passage_sha256", "split", "row", "role"])
        writer.writerows(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    freeze(parser.parse_args().output)
