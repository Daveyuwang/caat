"""Compatibility imports for the original evaluator and prediction type."""

from dataclasses import dataclass

from .evaluate import evaluate as evaluate_graded
from .metrics import Counts, grouped_counts, ranking_metrics, threshold_points
from .table import GradedRow, exact_match, read_label_records


@dataclass(frozen=True)
class Prediction:
    id: str
    prediction: str | None
    reference: str | None
    score: float
    group: str | None = None
    correct: bool | None = None


def read_predictions(path, columns, group_column):
    for values, score, correct, group in read_label_records(path, columns, group_column):
        yield Prediction(values["id"], values.get("prediction"), values.get("reference"), score, group, correct)


def evaluate(predictions, *, threshold, probability=False):
    def graded():
        for row in predictions:
            correct = row.correct
            if correct is None:
                correct = exact_match(row.id, row.prediction, row.reference)
            yield GradedRow(row.id, row.score, (bool(correct),), row.group, reference=row.reference)

    return evaluate_graded(graded(), threshold=threshold, probability=probability)
