"""Select one prompt per HellaSwag source and score the four original endings."""

import argparse

from collect_choice_logits import collect_choices
from common import ROOT, read_csv, write_report
from prepare_boolq import digest

OUT = ROOT / "results/hellaswag"
REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
CHOICES = "ABCD"


def dataset_file():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        "Rowan/hellaswag",
        "data/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )


def prompts():
    import pyarrow.parquet as pq

    raw = pq.read_table(
        dataset_file(), columns=["ind", "source_id", "ctx", "endings"]
    ).to_pylist()
    families = {}
    for i, r in enumerate(raw):
        uid = digest(f"{r['source_id']}|{r['ind']}|{i}")
        family = digest(r["source_id"])
        endings = dict(zip(CHOICES, r["endings"], strict=True))
        meta = dict(
            id=uid,
            source_row=i,
            source_sha256=family,
            **{c + "_sha256": digest(v) for c, v in endings.items()},
        )
        prompt = (
            "Context: "
            + r["ctx"]
            + "\nWhich ending best continues the context?\n"
            + "\n".join(f"{c}: {v}" for c, v in endings.items())
            + "\nReply with exactly A, B, C or D."
        )
        if family not in families or uid < families[family][0]["id"]:
            families[family] = meta, prompt
    return [families[k] for k in sorted(families)[:1500]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "collect"))
    args = parser.parse_args()
    rows = prompts()
    manifest = [m for m, _ in rows]
    path = OUT / "manifest.csv"
    if args.stage == "freeze":
        OUT.mkdir(exist_ok=True, parents=True)
        write_report(path, manifest)
    else:
        if read_csv(path) != [{k: str(v) for k, v in m.items()} for m in manifest]:
            raise ValueError("HellaSwag cohort differs from the frozen manifest")
        collect_choices(rows, CHOICES, OUT, REVISION)
