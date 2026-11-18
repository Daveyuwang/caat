"""Fit and replay a linear confidence ranking on named feature columns."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

from .metrics import ranking_metrics, threshold_points
from .scores import read_score_rows
from .table import atomic_output, open_prediction_table, parse_score, write_rows


FITTED_SCORE = "fitted_score"


def positive_number(value):
    return type(value) in (int, float) and isfinite(value) and value > 0


@dataclass(frozen=True)
class LinearScorer:
    columns: tuple
    signs: tuple
    means: tuple
    scales: tuple
    weights: tuple
    intercept: float
    regularization: float
    fit_rows: int
    fit_ids_sha256: str
    class_column: str | None = None
    categories: tuple = ()

    @property
    def input_columns(self):
        return (*self.columns, self.class_column) if self.class_column else self.columns

    def __post_init__(self):
        n = len(self.columns) + len(self.categories)
        if (
            not self.columns
            or bool(self.categories) != (self.class_column is not None)
            or (
                self.class_column is not None
                and (
                    not isinstance(self.class_column, str)
                    or not self.class_column.strip()
                    or self.class_column in self.columns
                )
            )
            or any(not isinstance(v, str) or not v for v in self.categories)
            or len(set(self.categories)) != len(self.categories)
            or any(not isinstance(v, str) or not v.strip() for v in self.columns)
            or len(set(self.columns)) != len(self.columns)
            or any(
                len(v) != n for v in (self.signs, self.means, self.scales, self.weights)
            )
            or any(type(v) is not int or v not in (-1, 1) for v in self.signs)
            or any(
                type(v) not in (int, float) or not isfinite(v)
                for v in (*self.means, *self.weights, self.intercept)
            )
            or not all(map(positive_number, self.scales))
            or not positive_number(self.regularization)
            or type(self.fit_rows) is not int
            or self.fit_rows < 1
            or not isinstance(self.fit_ids_sha256, str)
            or len(self.fit_ids_sha256) != 64
        ):
            raise ValueError("invalid linear scorer parameters")

    def score(self, values, *, predicted_class=None, review_missing_scores=False):
        values = tuple(values)
        if len(values) != len(self.columns) or any(
            v is not None and not isfinite(v) for v in values
        ):
            raise ValueError("expected one finite or missing value per scorer feature")
        if self.class_column is not None:
            if predicted_class not in self.categories:
                if review_missing_scores:
                    return None
                raise ValueError(
                    f"missing or unseen {self.class_column} value {predicted_class!r}. Use --review-missing-scores to send this row to review"
                )
            values += tuple(
                float(predicted_class == level) for level in self.categories
            )
        if any(v is None for v in values):
            return None
        result = self.intercept + sum(
            w * ((s * value - m) / scale)
            for value, s, m, scale, w in zip(
                values, self.signs, self.means, self.scales, self.weights, strict=True
            )
        )
        if not isfinite(result):
            raise ValueError("nonfinite fitted score")
        return result

    def score_row(self, row, line, review_missing_scores):
        values = [
            parse_score(row[c], line, review_missing_scores=review_missing_scores)
            for c in self.columns
        ]
        if any(v is not None and not isfinite(v) for v in values):
            raise ValueError(f"nonfinite feature at input line {line}")
        return self.score(
            values,
            predicted_class=row.get(self.class_column),
            review_missing_scores=review_missing_scores,
        )

    def to_dict(self):
        data = asdict(self)
        if self.class_column is None:
            del data["class_column"], data["categories"]
        return {
            "version": 2 if self.class_column else 1,
            "method": "standardized-logistic",
            **data,
        }

    @classmethod
    def from_dict(cls, payload):
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload.get("version") not in (1, 2)
            or payload.get("method") != "standardized-logistic"
        ):
            raise ValueError("unsupported saved confidence scorer")
        fields = dict(payload)
        version = fields.pop("version")
        del fields["method"]
        extras = {"class_column", "categories"} & fields.keys()
        if (
            version == 1
            and extras
            or version == 2
            and (
                len(extras) != 2
                or not fields["class_column"]
                or not fields["categories"]
            )
        ):
            raise ValueError("invalid saved scorer category metadata")
        try:
            arrays = ("columns", "signs", "means", "scales", "weights")
            for key in (*arrays, "categories") if version == 2 else arrays:
                if not isinstance(fields[key], (tuple, list)):
                    raise ValueError("expected arrays in saved confidence scorer")
                fields[key] = tuple(fields[key])
            return cls(**fields)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid saved confidence scorer") from exc


def _choose_regularization(features, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = min(5, int(labels.sum()), int(len(labels) - labels.sum()))
    if folds < 2:
        return 0.1
    splits = list(
        StratifiedKFold(folds, shuffle=True, random_state=42).split(features, labels)
    )

    def gain(strength):
        values = []
        for train, validation in splits:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=strength, max_iter=1000)
            )
            model.fit(features[train], labels[train])
            score = model.decision_function(features[validation])
            points = threshold_points(zip(score, labels[validation] == 0, strict=True))
            values.append(ranking_metrics(points)["normalized_aurc_gain"])
        return sum(values) / len(values)

    return max((0.01, 0.1, 1.0), key=gain)


def fit_scorer(
    columns,
    rows,
    *,
    lower_is_better=(),
    regularization=0.1,
    class_column=None,
    review_missing_scores=False,
):
    """Fit on design ScoredRows with raw features and binary correctness in outcome 0.

    Incomplete rows are omitted. A single-class design yields a constant ranking.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ValueError(
            "install caat-annotation[scorer] to fit a confidence scorer"
        ) from exc
    columns = tuple(columns)
    if (regularization != "auto" and not positive_number(regularization)) or set(
        lower_is_better
    ).difference(columns):
        raise ValueError(
            "provide positive regularization C and known feature directions"
        )
    signs = tuple(-1 if c in lower_is_better else 1 for c in columns)
    rows = list(rows)
    if any(not r.id or not r.outcomes for r in rows) or len(
        {r.id for r in rows}
    ) != len(rows):
        raise ValueError("design needs unique nonempty IDs and binary grades")
    if (
        class_column is not None
        and not review_missing_scores
        and any(not r.predicted_class for r in rows)
    ):
        raise ValueError(
            f"missing {class_column} in design data. Use --review-missing-scores to skip incomplete design rows"
        )
    complete = sorted(
        (
            r
            for r in rows
            if None not in r.scores and (class_column is None or r.predicted_class)
        ),
        key=lambda r: r.id,
    )
    if not complete:
        raise ValueError("no complete design rows to fit a scorer")
    x = np.asarray([r.scores for r in complete], dtype=float)
    if x.shape != (len(complete), len(columns)) or not np.isfinite(x).all():
        raise ValueError("design needs one finite value per feature")
    categories = (
        tuple(sorted({r.predicted_class for r in complete}))
        if class_column is not None
        else ()
    )
    if categories:
        indicators = [
            [r.predicted_class == category for category in categories] for r in complete
        ]
        x = np.column_stack((x, indicators))
        signs += (1,) * len(categories)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(x * signs)
    labels = [r.outcomes[0] for r in complete]
    if any(v not in (True, False) for v in labels):
        raise ValueError("expected binary design grades")
    if regularization == "auto":
        regularization = _choose_regularization(x * signs, np.asarray(labels))
    weights, intercept = (0.0,) * len(signs), 0.0
    if len(set(labels)) > 1:
        model = LogisticRegression(C=regularization, max_iter=1000).fit(
            standardized, labels
        )
        weights, intercept = (
            tuple(map(float, model.coef_[0])),
            float(model.intercept_[0]),
        )
    digest = sha256(
        json.dumps([r.id for r in complete], separators=(",", ":")).encode()
    ).hexdigest()
    return LinearScorer(
        columns,
        signs,
        tuple(map(float, scaler.mean_)),
        tuple(map(float, scaler.scale_)),
        weights,
        intercept,
        regularization,
        len(complete),
        digest,
        class_column,
        categories,
    )


def read_scorer(path):
    with Path(path).open(encoding="utf-8") as source:
        artifact = json.load(source)
    return LinearScorer.from_dict(
        artifact.get("scorer", artifact) if isinstance(artifact, dict) else artifact
    )


def apply_scorer(
    path, output, scorer, *, column=FITTED_SCORE, review_missing_scores=False
):
    if output.resolve() == path.resolve():
        raise ValueError("output would overwrite the feature input")
    with open_prediction_table(path, set(scorer.input_columns), None) as (reader, _):
        if not column or column in reader.fieldnames:
            raise ValueError(f"score column already exists: {column}")
        fields = tuple(reader.fieldnames)
        return write_rows(
            output,
            (*fields, column),
            (
                (
                    *(row[f] for f in fields),
                    scorer.score_row(row, reader.line_num, review_missing_scores),
                )
                for row in reader
            ),
        )


def configure_scorer(parser):
    commands = parser.add_subparsers(dest="operation", required=True)
    fit = commands.add_parser(
        "fit", help="Fit a standardized logistic score on design rows."
    )
    fit.add_argument("predictions", type=Path)
    fit.add_argument("--id-column", default="id")
    fit.add_argument("--score-column", action="append", required=True)
    fit.add_argument(
        "--class-column", help="Optional categorical prediction available at inference."
    )
    fit.add_argument("--lower-is-better", action="append", default=[])
    fit.add_argument(
        "--regularization",
        default="0.1",
        help="Positive inverse L2 penalty C, or auto for design-fold tuning.",
    )
    fit.add_argument("--correct-column")
    fit.add_argument("--prediction-column")
    fit.add_argument("--reference-column")
    apply = commands.add_parser(
        "apply", help="Add a frozen score to an unlabeled table."
    )
    apply.add_argument("predictions", type=Path)
    apply.add_argument(
        "--scorer",
        type=Path,
        required=True,
        help="Scorer JSON or fitted workflow report.",
    )
    apply.add_argument("--output-column", default=FITTED_SCORE)
    for sub in (fit, apply):
        sub.add_argument("--review-missing-scores", action="store_true")
        sub.add_argument("--output", type=Path, required=True)


def run_scorer(args):
    if args.output.resolve() == args.predictions.resolve():
        raise ValueError("output would overwrite the feature input")
    if args.operation == "apply":
        if args.scorer.resolve() == args.output.resolve():
            raise ValueError("output would overwrite the saved scorer")
        count = apply_scorer(
            args.predictions,
            args.output,
            read_scorer(args.scorer),
            column=args.output_column,
            review_missing_scores=args.review_missing_scores,
        )
        print(f"Wrote {count} scored rows to {args.output}.")
    else:
        if args.correct_column and (args.prediction_column or args.reference_column):
            raise ValueError("choose binary grades or predicted/reference labels")
        labels = (
            None
            if args.correct_column
            else (
                args.prediction_column or "prediction",
                args.reference_column or "reference",
            )
        )
        grades = [args.correct_column] if args.correct_column else []
        if set(args.score_column) & {args.id_column, *grades, *(labels or ())}:
            raise ValueError("feature names must differ from ID and label columns")
        if args.class_column is not None and args.class_column in {
            args.id_column,
            *grades,
            *args.score_column,
            *(labels[1:] if labels else []),
        }:
            raise ValueError(
                "class column must differ from numeric features, ID and reference grades"
            )
        rows = read_score_rows(
            args.predictions,
            args.score_column,
            grades,
            id_column=args.id_column,
            label_columns=labels,
            class_column=args.class_column,
            review_missing_scores=args.review_missing_scores,
        )
        scorer = fit_scorer(
            args.score_column,
            rows,
            lower_is_better=args.lower_is_better,
            class_column=args.class_column,
            review_missing_scores=args.review_missing_scores,
            regularization=(
                "auto" if args.regularization == "auto" else float(args.regularization)
            ),
        )
        with atomic_output(args.output, newline="") as handle:
            json.dump(scorer.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            f"Fitted {len(scorer.input_columns)} features on {scorer.fit_rows} design rows and wrote {args.output}."
        )
    return 0
