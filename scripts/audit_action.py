"""Run the referenced CAAT checkout for the composite grader audit action."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = (
    "threshold",
    "threshold-report",
    "risk-target",
    "id-column",
    "score-column",
    "raw-score-column",
    "output-key-column",
    "grader-id-column",
    "grader-output-key-column",
    "group-column",
    "prediction-filter",
    "holdout-confidence",
    "accepted-population",
    "batch-decisions",
    "plan-report",
    "sample-seed",
    "bootstrap",
    "seed",
)
REPEATED = {"grader-files": "grader-file", "correct-columns": "correct-column"}
BOOLEAN_OPTIONS = ("allow-extra-grades", "review-missing-scores")


def input_error(inputs):
    if not inputs.get("predictions"):
        return "Provide a predictions path."
    threshold = inputs.get("threshold")
    if bool(threshold) == bool(inputs.get("threshold-report")):
        return "Provide exactly one of threshold and threshold-report."
    if threshold and not (inputs.get("risk-target") or inputs.get("plan-report")):
        return "Provide risk-target or plan-report with a numeric threshold."
    for option in BOOLEAN_OPTIONS:
        if inputs.get(option, "") not in ("", "true", "false"):
            return f"Set {option} to true or false."
    return None


def audit_args(inputs):
    flags = [f"--{option}={inputs[option]}" for option in OPTIONS if inputs.get(option)]
    flags.extend(
        f"--{option}" for option in BOOLEAN_OPTIONS if inputs.get(option) == "true"
    )
    if path := inputs.get("disagreements-path"):
        flags.append(f"--disagreements-output={path}")
    for key, option in REPEATED.items():
        flags.extend(
            f"--{option}={entry}"
            for entry in inputs.get(key, "").splitlines()
            if entry.strip()
        )
    return [
        "audit",
        *flags,
        f"--output={inputs['report-path']}",
        "--",
        inputs["predictions"],
    ]


def run_audit(args):
    with TemporaryDirectory(
        prefix="caat-audit-", dir=os.environ.get("RUNNER_TEMP")
    ) as directory:
        subprocess.run([sys.executable, "-I", "-m", "venv", directory], check=True)
        python = Path(directory) / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(
            [
                python,
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(ROOT),
            ],
            check=True,
        )
        result = subprocess.run(
            [
                python,
                "-I",
                "-X",
                "utf8",
                "-c",
                "from caat.cli import main; raise SystemExit(main())",
                *args,
            ],
            check=False,
        )
        return result.returncode


def main():
    inputs = json.loads(os.environ["CAAT_ACTION_INPUTS"])
    if error := input_error(inputs):
        print(error, file=sys.stderr)
        status = 2
    else:
        status = run_audit(audit_args(inputs))
    with Path(os.environ["GITHUB_OUTPUT"]).open(
        "a", encoding="utf-8", newline="\n"
    ) as output:
        output.write(f"status={status}\n")
        if status in (0, 1):
            for name in ("report-path", "disagreements-path"):
                if path := inputs.get(name):
                    delimiter = uuid4().hex
                    output.write(f"{name}<<{delimiter}\n{path}\n{delimiter}\n")


if __name__ == "__main__":
    main()
