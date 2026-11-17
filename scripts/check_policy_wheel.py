"""Run the README fitted-policy example from clean training and base wheel installs."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
from shutil import copyfile
from subprocess import run
from tempfile import TemporaryDirectory
from time import perf_counter
from venv import EnvBuilder


ROOT = Path(__file__).resolve().parents[1]


def install(directory, wheel, extras):
    EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(directory)
    binaries = directory / ("Scripts" if os.name == "nt" else "bin")
    python = binaries / ("python.exe" if os.name == "nt" else "python")
    cli = binaries / ("caat.exe" if os.name == "nt" else "caat")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel) + extras,
        ],
        cwd=directory,
        check=True,
    )
    return python, cli


def command(cli, cwd, *args, status=0):
    result = run([cli, *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != status:
        raise RuntimeError(
            f"{args} returned {result.returncode}: {result.stdout} {result.stderr}"
        )
    return result.stdout


def documented_commands(document, heading):
    text = (ROOT / document).read_text(encoding="utf-8")
    section = text.split("## " + heading + "\n", 1)[1].split("\n## ", 1)[0]
    commands = {}
    for line in section.splitlines():
        if line.startswith("caat "):
            _, name, *arguments = shlex.split(line)
            commands[name] = arguments
    return commands


def check_batch(cli, handoff, commands, expected):
    command(cli, handoff, "route", *commands["route"])
    output = commands["route"][commands["route"].index("--output") + 1]
    with (handoff / output).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    decisions = {r["id"]: r["decision"] for r in rows}
    if decisions != expected or len(rows) != len(expected):
        raise ValueError(f"unexpected routed IDs: {decisions}")
    for cmd, status in (("evaluate", 1), ("audit", 1), ("monitor", 0)):
        for item in (None, *expected):
            filt = ["--prediction-filter", f"image_id={item}"] if item else []
            result = json.loads(
                command(
                    cli, handoff, cmd, *commands[cmd], *filt, "--json", status=status
                )
            )
            counts = (
                result["graders"]["correct"] if cmd == "audit" else result["overall"]
            )
            accepted = (
                sum(v == "accept" for v in expected.values())
                if item is None
                else int(expected[item] == "accept")
            )
            if counts["accepted"] != accepted or counts["total"] != (
                len(expected) if item is None else 1
            ):
                raise ValueError(f"{cmd} disagrees on {item}")
    return decisions


def verify(wheel):
    numeric = documented_commands(
        "README.md", "Reuse one fitted policy from routing through audit"
    )
    categorical = documented_commands(
        "docs/guide/classifier.md", "Hand off a class-aware policy"
    )
    start = perf_counter()
    with TemporaryDirectory(prefix="caat-first-use-") as temp:
        root = Path(temp)
        examples = root / "examples"
        examples.mkdir()
        for name in (
            "digit_classifier.jsonl",
            "fitted_classifier.toml",
            "class_aware_digits.toml",
        ):
            copyfile(ROOT / "examples" / name, examples / name)
        _, train = install(root / "training", wheel, "[scorer,workflow]")
        for cmds in (numeric, categorical):
            command(train, root, "run", *cmds["run"])
        training_seconds = perf_counter() - start
        phase = perf_counter()
        python, base = install(root / "base", wheel, "")
        run(
            [
                python,
                "-c",
                "import importlib.util; assert importlib.util.find_spec('sklearn') is None",
            ],
            cwd=root,
            check=True,
        )
        handoff = root / "handoff"
        (handoff / "examples").mkdir(parents=True)
        for name in (
            "fitted_batch.csv",
            "fitted_batch_graded.csv",
            "class_batch.csv",
            "class_batch_graded.csv",
        ):
            copyfile(ROOT / "examples" / name, handoff / "examples" / name)
        for cmds in (numeric, categorical):
            report = Path(cmds["route"][cmds["route"].index("--threshold-report") + 1])
            (handoff / report).parent.mkdir(parents=True)
            copyfile(root / report, handoff / report)
        known = {"new-a": "accept", "new-b": "review"}
        decisions = check_batch(base, handoff, numeric, known)
        class_decisions = check_batch(
            base, handoff, categorical, {**known, "new-c": "review", "new-d": "review"}
        )
        return {
            "training_seconds": round(training_seconds, 2),
            "handoff_seconds": round(perf_counter() - phase, 2),
            "decisions": decisions,
            "categorical_decisions": class_decisions,
            "exit_codes": {
                "run": 0,
                "route": 0,
                "evaluate": 1,
                "audit": 1,
                "monitor": 0,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    print(json.dumps(verify(parser.parse_args().wheel.resolve())))
