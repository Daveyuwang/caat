"""Freeze disjoint ANLI premise groups before using any grades."""

import argparse
from pathlib import Path
from random import Random

from common import write_report
from prepare_boolq import digest, normalized

REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
SEED = 20261118
SPLITS = ("test_r1", "test_r3", "train_r1")


def source(split, columns, *, ids=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "facebook/anli",
        f"plain_text/{split}-00000-of-00001.parquet",
        revision=REVISION,
        repo_type="dataset",
    )
    filters = [("uid", "in", sorted(ids))] if ids is not None else None
    return pq.read_table(path, columns=columns, filters=filters).to_pylist()


def freeze(output):
    seen = set()
    manifest = []
    for split in SPLITS:
        groups = {}
        for row in source(split, ["uid", "premise", "hypothesis"]):
            premise = digest(normalized(row["premise"]))
            if premise in seen:
                continue
            candidate = (digest(normalized(row["hypothesis"])), row["uid"])
            if premise not in groups or candidate < groups[premise]:
                groups[premise] = candidate
        seen.update(groups)
        rows = sorted(groups.items())
        Random(SEED).shuffle(rows)
        if split == "train_r1":
            roles = ["design"] * 400 + ["reference"] * 160 + ["calibration"] * 480
            rows = rows[: len(roles)]
        else:
            roles = ["primary" if split == "test_r1" else "stress"] * len(rows)
        for (premise, (hypothesis, uid)), role in zip(rows, roles, strict=True):
            manifest.append(
                dict(
                    id=uid,
                    split=split,
                    role=role,
                    premise_sha256=premise,
                    hypothesis_sha256=hypothesis,
                )
            )
    if len(set(r["id"] for r in manifest)) != len(manifest):
        raise ValueError("duplicate ANLI IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    freeze(parser.parse_args().output)
