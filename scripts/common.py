"""Shared I/O for the research replays."""

import csv
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SCENARIOS = {
    "wildbench": ("score", "wildbench_score_rescaled"),
    "omni_math": ("equivalence_judgement", "omni_math_accuracy"),
}


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def fetch(url, path):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"Accept-Encoding": "gzip"})
    with urlopen(request, timeout=90) as source:
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                if source.headers.get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=source) as uncompressed:
                        copyfileobj(uncompressed, output, length=1 << 20)
                else:
                    copyfileobj(source, output, length=1 << 20)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
    return path


def read_csv(path, *, encoding="utf-8"):
    with path.open(encoding=encoding, newline="") as source:
        return list(csv.DictReader(source))


def write_report(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=fields or list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
