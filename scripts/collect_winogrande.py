"""Freeze WinoGrande input families and score the two original choices on GPU."""

import argparse
import json

from collect_choice_logits import collect_choices
from common import ROOT, read_csv, write_report
from prepare_boolq import digest, normalized

REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
OUT = ROOT / "results/winogrande"
FIELDS = ["sentence", "option1", "option2"]


def dataset_file():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "allenai/winogrande",
        "winogrande_xl/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )


def prompts():
    import pyarrow.parquet as pq

    raw = pq.read_table(dataset_file(), columns=FIELDS).to_pylist()
    if len(raw) != 1267:
        raise ValueError("unexpected validation split size")
    families = {}
    for i, r in enumerate(raw):
        choices = [r["option1"], r["option2"]]
        norm = [normalized(s) for s in choices]
        family = digest(json.dumps(sorted(norm), ensure_ascii=False))
        uid = "wino-" + digest(
            json.dumps([normalized(r["sentence"]), *norm], ensure_ascii=False)
        )
        meta = dict(
            id=uid,
            source_row=i,
            family_sha256=family,
            A_sha256=digest(choices[0]),
            B_sha256=digest(choices[1]),
        )
        msg = f"Sentence: {r['sentence']}\nWhich choice fills the blank?\nA: {choices[0]}\nB: {choices[1]}\nReply with exactly A or B."
        if family not in families or uid < families[family][0]["id"]:
            families[family] = (meta, msg)
    return sorted(families.values(), key=lambda pair: pair[0]["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "collect"])
    args = parser.parse_args()
    rows = prompts()
    manifest = [m for m, _ in rows]
    if args.stage == "freeze":
        OUT.mkdir(parents=True, exist_ok=True)
        write_report(OUT / "manifest.csv", manifest)
        print("Frozen families:", len(manifest))
    else:
        if read_csv(OUT / "manifest.csv") != [
            {k: str(v) for k, v in m.items()} for m in manifest
        ]:
            raise ValueError("cohort changed after freeze")
        collect_choices(rows, "AB", OUT, REVISION)
