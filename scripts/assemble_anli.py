"""Join frozen predictions to grades, optionally opening only the design split."""

import argparse
import json
from math import exp
from pathlib import Path

from collect_anli import LABELS
from common import ROOT, read_csv, write_report
from prepare_anli import source
from prepare_boolq import digest


def indexed(rows):
    lookup = {r["id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate row IDs")
    return lookup


def assemble(directory, output, design_only):
    manifest = indexed(read_csv(ROOT / "results/anli/manifest.csv"))
    gen = indexed(read_csv(directory / "generate.csv"))
    verify = indexed(read_csv(directory / "verify.csv"))
    rotated = indexed(read_csv(directory / "rotate.csv"))
    if not manifest.keys() == gen.keys() == verify.keys() or rotated.keys() != {
        i for i, r in manifest.items() if r["role"] == "design"
    }:
        raise ValueError("model outputs do not match the frozen groups")
    selected = [
        r for r in manifest.values() if not design_only or r["role"] == "design"
    ]
    gold = {}
    for split in {r["split"] for r in selected}:
        gold.update(
            {r["uid"]: int(r["label"]) for r in source(split, ["uid", "label"])}
        )
    rows = []
    for r in sorted(selected, key=lambda r: r["id"]):
        i = r["id"]
        g, v = gen[i], verify[i]
        options = [float(g[k]) for k in "ABC"]
        answer = LABELS[options.index(max(options))]
        if not (
            answer == g["answer"]
            and digest(answer) == g["output_sha256"] == v["output_sha256"]
        ):
            raise ValueError(f"answer or verifier hash mismatch for {i}")
        ordered = sorted(options, reverse=True)
        rotation = rotated.get(i)
        if rotation:
            vals = [float(rotation[k]) for k in "ABC"]
            if (
                LABELS[(vals.index(max(vals)) + 1) % 3] != rotation["answer"]
                or digest(rotation["answer"]) != rotation["output_sha256"]
            ):
                raise ValueError("invalid rotated response")
        rows.append(
            dict(
                id=i,
                role=r["role"],
                output_sha256=g["output_sha256"],
                answer=answer,
                correct=""
                if r["role"] == "reference"
                else int(answer == LABELS[gold[i]]),
                generator_prompt_sha256=g["prompt_sha256"],
                verifier_prompt_sha256=v["prompt_sha256"],
                rotated_answer=rotation["answer"] if rotation else "",
                rotated_prompt_sha256=rotation["prompt_sha256"] if rotation else "",
                **{k: g[k] for k in ("A", "B", "C", "log_z")},
                true_logit=v["True"],
                false_logit=v["False"],
                winner_logprob=ordered[0] - float(g["log_z"]),
                gap=ordered[0] - ordered[1],
                p_true=1 / (1 + exp(float(v["False"]) - float(v["True"]))),
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, rows)
    if not design_only:
        cost = {
            s: json.loads((directory / (s + ".json")).read_text())
            for s in ("generate", "rotate", "verify")
        }
        (output.parent / "collection.json").write_text(
            json.dumps(cost, indent=2) + "\n", encoding="utf8"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--design-only", action="store_true")
    args = p.parse_args()
    assemble(args.directory, args.output, args.design_only)
