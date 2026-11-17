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


def verify(wheel):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split(
        "## Reuse one fitted policy from routing through audit\n", 1
    )[1]
    commands = {}
    for line in section.split("\n## ", 1)[0].splitlines():
        if line.startswith("caat "):
            _, name, *arguments = shlex.split(line)
            commands[name] = arguments
    start = perf_counter()
    with TemporaryDirectory(prefix="caat-first-use-") as temp:
        root = Path(temp)
        examples = root / "examples"
        examples.mkdir()
        for name in ("digit_classifier.jsonl", "fitted_classifier.toml"):
            copyfile(ROOT / "examples" / name, examples / name)
        _, train = install(root / "training", wheel, "[scorer,workflow]")
        command(train, root, "run", *commands["run"])
        report = root / "replay/fitted-classifier/workflow.json"
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
        for name in ("fitted_batch.csv", "fitted_batch_graded.csv"):
            copyfile(ROOT / "examples" / name, handoff / "examples" / name)
        copied_report = handoff / report.relative_to(root)
        copied_report.parent.mkdir(parents=True)
        copyfile(report, copied_report)
        command(base, handoff, "route", *commands["route"])
        with (handoff / "replay/decisions.csv").open(
            encoding="utf8", newline=""
        ) as source:
            decisions = {r["id"]: r["decision"] for r in csv.DictReader(source)}
        if decisions != {"new-a": "accept", "new-b": "review"}:
            raise ValueError(f"unexpected routed IDs: {decisions}")
        for cmd, status in (("evaluate", 1), ("audit", 1), ("monitor", 0)):
            # Check the aggregate verdict and each ID so opposite mistakes cannot cancel.
            for item, expected in ((None, 1), ("new-a", 1), ("new-b", 0)):
                filt = ["--prediction-filter", f"image_id={item}"] if item else []
                result = json.loads(
                    command(
                        base,
                        handoff,
                        cmd,
                        *commands[cmd],
                        *filt,
                        "--json",
                        status=status,
                    )
                )
                counts = (
                    result["graders"]["correct"]
                    if cmd == "audit"
                    else result["overall"]
                )
                if counts["accepted"] != expected:
                    raise ValueError(f"{cmd} disagrees on {item}")
        return {
            "training_seconds": round(training_seconds, 2),
            "handoff_seconds": round(perf_counter() - phase, 2),
            "decisions": decisions,
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
