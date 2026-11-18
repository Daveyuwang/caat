"""Persist a fixed draw's requests and keyed grade receipts in a local SQLite journal."""

import json
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from .partial import progress_report
from .plan import candidate_draw
from .queue import write_queue
from .report import emit, make_report, read_label_plan, read_threshold_report
from .table import read_grader_file, read_table_rows


def block_options(parser, named_file):
    parser.add_argument(
        "--handoff",
        required=True,
        type=Path,
        help="Directory for the audit journal and issued CSVs.",
    )
    parser.add_argument("--plan-report", required=True, type=Path)
    parser.add_argument("--threshold-report", required=True, type=Path)
    parser.add_argument("--batch-predictions", required=True, type=Path)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument(
        "--grader",
        required=True,
        action="append",
        help="Name of a requested grader. Repeat for multiple graders.",
    )
    parser.add_argument("--request", help="Request ID that issued these rows.")
    parser.add_argument(
        "--grader-file",
        action="append",
        type=named_file,
        metavar="NAME=FILE",
        help="Return rows from this request, keyed by id and output_key.",
    )
    parser.add_argument("--json", action="store_true")


def receive(state, token, files, draw):
    if not files:
        if token:
            raise ValueError("--request requires a --grader-file")
        return
    request = next((r for r in state["requests"] if r["id"] == token), None)
    if request is None:
        raise ValueError("return grades with the issuing --request ID")
    keys = {row.id: row.output_key for row in draw[request["start"] : request["end"]]}
    for name, path in files:
        if name not in request["graders"]:
            raise ValueError(f"{name}: grader was not issued this request")
        incoming = read_grader_file(
            path,
            name,
            keys,
            id_column="id",
            output_key_column="output_key",
            require_all=False,
        )
        known = state["grades"][name]
        for uid, value in incoming.items():
            if uid in known and known[uid] != value:
                raise ValueError(f"{name}: conflicting grade for {uid!r}")
            known[uid] = value


def advance(state, draw, plan, block_size, run_id):
    requests = state["requests"]
    current = requests[-1] if requests else None
    end = current["end"] if current else 0
    progress = {}
    for name, known in state["grades"].items():
        prefix = {}
        for row in draw:
            if row.id not in known:
                break
            prefix[row.id] = known[row.id]
        progress[name] = prefix
    result = progress_report(plan, progress)
    pending = [
        name for name, r in result["graders"].items() if r["batch_verdict"] is None
    ]
    outstanding = {}
    if current:
        for name in current["graders"]:
            outstanding[name] = [
                row.id
                for row in draw[current["start"] : end]
                if row.id not in state["grades"][name]
            ]
    if not any(outstanding.values()):
        current = None
        outstanding = {}
        if pending:
            current = {
                "id": f"{run_id}-{len(requests) + 1}",
                "start": end,
                "end": min(end + block_size, len(draw)),
                "graders": pending,
            }
            requests.append(current)
            outstanding = {
                name: [r.id for r in draw[end : current["end"]]] for name in pending
            }
    return {
        **result,
        "request": current,
        "outstanding": outstanding,
        "requests": len(requests),
        "unique_requested": requests[-1]["end"] if requests else 0,
        "remaining_capacity": len(draw) - (requests[-1]["end"] if requests else 0),
        "requested_labels": sum(
            (r["end"] - r["start"]) * len(r["graders"]) for r in requests
        ),
        "returned_labels": sum(len(g) for g in state["grades"].values()),
    }


def run_block(args):
    names = sorted(args.grader)
    if args.block_size < 1 or not all(names) or len(names) != len(set(names)):
        raise ValueError("use a positive block size and distinct grader names")
    plan = read_label_plan(args.plan_report, allow_empty=True)
    if "threshold_report_sha256" not in plan:
        raise ValueError(
            "audit-block requires a fixed-budget plan for raw batch predictions"
        )
    saved = read_threshold_report(args.threshold_report, fingerprint=True)
    draw = candidate_draw(args.batch_predictions, saved, plan)
    if draw and draw[0].output_key is None:
        raise ValueError("a block handoff requires keyed batch predictions")
    contract = {k: v for k, v in plan.items() if k != "sample_path"}
    contract.update(block_size=args.block_size, graders=names)
    binding = json.dumps(contract, sort_keys=True)
    run_id = sha256(binding.encode()).hexdigest()[:20]
    args.handoff.mkdir(parents=True, exist_ok=True)
    journal = args.handoff / "audit.sqlite3"
    files = args.grader_file or []
    inputs = [
        args.plan_report,
        args.threshold_report,
        args.batch_predictions,
        *(p for _, p in files),
    ]
    for path in inputs:
        resolved = path.resolve()
        if resolved.parent == args.handoff.resolve() and (
            resolved.name.startswith("audit.sqlite3")
            or resolved.name.startswith("request-")
        ):
            raise ValueError(
                "handoff journal and issued requests must be separate from inputs"
            )
    try:
        with closing(sqlite3.connect(journal, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT, state TEXT)"
            )
            entry = db.execute("SELECT binding, state FROM audit WHERE id=1").fetchone()
            if entry and entry[0] != binding:
                raise ValueError(
                    "handoff does not match the saved plan, block size or graders"
                )
            state = (
                json.loads(entry[1])
                if entry
                else {"requests": [], "grades": {n: {} for n in names}}
            )
            receive(state, args.request, files, draw)
            result = advance(state, draw, plan, args.block_size, run_id)
            db.execute(
                "INSERT OR REPLACE INTO audit VALUES(1,?,?)",
                (binding, json.dumps(state)),
            )
            # Commit the issued identities before handing them to an external labeler.
            db.commit()
            if request := result["request"]:
                db.execute("BEGIN IMMEDIATE")
                path = args.handoff / f"request-{request['id']}.csv"
                issued = draw[request["start"] : request["end"]]
                if path.exists():
                    rows, _ = read_table_rows(
                        path, output_key_column="output_key", require_labels=False
                    )
                    if [(r.id, r.score, r.group, r.output_key) for r in rows] != [
                        (r.id, r.score, r.group, r.output_key) for r in issued
                    ]:
                        raise ValueError(
                            f"issued CSV differs from the frozen request: {path}"
                        )
                else:
                    write_queue(path, issued, include_output_key=True)
                result["request"] = dict(request, path=str(path))
    except sqlite3.Error as exc:
        raise ValueError(f"{journal}: {exc}") from exc
    emit(make_report("audit-block", contract, result), json_output=args.json)
    return 0
