"""Grade archived TriviaQA answers against the official Web validation aliases."""

import argparse
import csv
import re
from math import isfinite
from pathlib import Path
from string import punctuation

from common import fetch

DATASET_URL = (
    "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/"
    "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/rc.web.nocontext/validation-00000-of-00001.parquet"
)
RUN_URL = (
    "https://huggingface.co/datasets/yuxixia/triviaqa_llama8b-dpo_test/resolve/"
    "c73c578fd7302bec3c4ecc69d75691cced11aa4a/data/train-00000-of-00001.parquet"
)
SAMPLE_SIZE = 24
PUNCTUATION_TO_SPACE = str.maketrans(dict.fromkeys(punctuation + "‘’´", " "))


def normalize_answer(text: str) -> str:
    # TriviaQA's official v1.0 evaluator replaces punctuation with spaces.
    words = text.lower().translate(PUNCTUATION_TO_SPACE)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", words).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("replay/triviaqa"))
    output = parser.parse_args().output_dir

    from pyarrow.parquet import read_table

    cache = output / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    gold_path = cache / "rc.web.nocontext-validation.parquet"
    run_path = cache / "triviaqa_llama8b-dpo_test.parquet"
    for path, url in ((gold_path, DATASET_URL), (run_path, RUN_URL)):
        fetch(url, path)

    gold_rows = read_table(gold_path, columns=["question", "question_id", "answer"]).to_pylist()
    official = {row["question"]: row for row in gold_rows}
    if len(official) != len(gold_rows):
        raise ValueError("Official question text must identify a unique TriviaQA row")
    columns = ["question", "target", "response", "response_prob"]
    sample = read_table(run_path, columns=columns).slice(0, SAMPLE_SIZE).to_pylist()

    rows = []
    for index, row in enumerate(sample):
        gold = official.get(row["question"])
        if gold is None:
            raise ValueError(f"Archived row {index} has no exact question match in Web validation")
        question_id = gold["question_id"]
        aliases = {normalize_answer(alias) for alias in gold["answer"]["normalized_aliases"]}
        if normalize_answer(row["target"]) not in aliases:
            raise ValueError(f"Archived target differs from official aliases for TriviaQA {question_id}")
        score = row["response_prob"]
        if score is None or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid saved response score for TriviaQA {question_id}")
        correct = int(normalize_answer(row["response"]) in aliases)
        rows.append((question_id, score, correct))

    if len({row[0] for row in rows}) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} distinct TriviaQA questions")
    path = output / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("id", "response_prob", "correct"))
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
