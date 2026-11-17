"""Run a sequential risk test for one frozen acceptance policy."""

from fractions import Fraction
from itertools import islice
from math import ceil, isfinite

from .metrics import Counts
from .table import accepted_at, validated_rows


ALTERNATIVE_FRACTIONS = (0.1, 0.3, 0.6)


class _MixtureTest:
    """Track bounds on the mixture and resolve ambiguous comparisons exactly."""

    def __init__(self, risk_target, failure_rate, row_count):
        risk = Fraction(str(risk_target))
        fractions = tuple(Fraction(str(value)) for value in ALTERNATIVE_FRACTIONS)
        self.factors = (
            tuple(1 - value for value in fractions),
            tuple(1 + value * (1 - risk) / risk for value in fractions),
        )
        self.ratios = tuple(
            tuple(value.as_integer_ratio() for value in factors)
            for factors in self.factors
        )
        self.alternatives = len(fractions)
        # Leave room for deposits of order 1 / row_count**2 and their rounding.
        self.scale = 1 << (80 + 3 * row_count.bit_length())
        self.level = 1 / Fraction(str(failure_rate))
        self.cutoff = ceil(self.scale * self.level)
        # The reciprocal of this evidence rounds to zero in binary64.
        self.underflow_level = self.scale << 1075
        self.accounts = [(0, 0)] * self.alternatives
        self.peak_lower = self.peak_upper = self.scale
        self.alerted = False
        self.outcomes = bytearray()
        self.exact_accounts = [Fraction(0)] * self.alternatives
        self.exact_rows = 0
        self.exact_peak = Fraction(1)

    def _exact_maximum(self):
        for position in range(self.exact_rows, len(self.outcomes)):
            count = position + 1
            evidence = Fraction(1, count + 1)
            deposit = evidence / (count * self.alternatives)
            for index, factor in enumerate(self.factors[self.outcomes[position]]):
                amount = (self.exact_accounts[index] + deposit) * factor
                self.exact_accounts[index] = amount
                evidence += amount
            self.exact_peak = max(self.exact_peak, evidence)
        self.exact_rows = len(self.outcomes)
        return self.exact_peak

    def update(self, error):
        if self.alerted and self.peak_lower >= self.underflow_level:
            return True
        self.outcomes.append(error)
        count = len(self.outcomes)
        # Keep the weight of all future starts at one until they enter.
        evidence_lower, remainder = divmod(self.scale, count + 1)
        evidence_upper = evidence_lower + bool(remainder)
        deposit_lower, remainder = divmod(
            self.scale, count * (count + 1) * self.alternatives
        )
        deposit_upper = deposit_lower + bool(remainder)
        for index, (numerator, denominator) in enumerate(self.ratios[error]):
            lower, upper = self.accounts[index]
            lower = (lower + deposit_lower) * numerator // denominator
            upper = (
                (upper + deposit_upper) * numerator + denominator - 1
            ) // denominator
            self.accounts[index] = (lower, upper)
            evidence_lower += lower
            evidence_upper += upper
        self.peak_lower = max(self.peak_lower, evidence_lower)
        self.peak_upper = max(self.peak_upper, evidence_upper)
        if not self.alerted and evidence_upper >= self.cutoff:
            self.alerted = (
                evidence_lower >= self.cutoff or self._exact_maximum() >= self.level
            )
        return self.alerted

    def pvalue(self):
        lower = float(Fraction(self.scale, self.peak_upper))
        upper = float(Fraction(self.scale, self.peak_lower))
        return lower if lower == upper else float(1 / self._exact_maximum())


def _check_coverage(rows, *, threshold, routed, window_size, min_coverage):
    total = len(rows)
    check = {
        "window_size": window_size,
        "min_coverage": min_coverage,
        "windows_checked": max(0, total - window_size + 1),
        "first_alert": None,
        "latest_window": None,
    }
    if not check["windows_checked"]:
        return check

    minimum = ceil(Fraction(str(min_coverage)) * window_size)

    def accepts(row):
        return row.accepted if routed else accepted_at(row, threshold)

    def window_metrics(item_id, end, accepted, missing):
        return {
            "id": item_id,
            "start_row": end - window_size + 1,
            "end_row": end,
            "accepted": accepted,
            "coverage": accepted / window_size,
            "missing_scores": missing,
        }

    accepted = missing = 0
    outgoing = iter(rows)
    for end, row in enumerate(rows, 1):
        accepted += accepts(row)
        missing += row.score is None
        if end > window_size:
            oldest = next(outgoing)
            accepted -= accepts(oldest)
            missing -= oldest.score is None
        if end >= window_size and accepted < minimum:
            first = window_metrics(row.id, end, accepted, missing)
            check["windows_checked"] = first["start_row"]
            check["first_alert"] = first
            break
    if end < total:
        accepted = missing = 0
        for row in islice(rows, total - window_size, None):
            accepted += accepts(row)
            missing += row.score is None
    check["latest_window"] = {
        **window_metrics(row.id, total, accepted, missing),
        "meets_minimum": accepted >= minimum,
    }
    return check


def monitor_risk(
    rows,
    *,
    grader,
    threshold,
    risk_target,
    failure_rate=0.05,
    labels=None,
    routed=False,
    review_missing_scores=False,
    coverage_window=None,
    min_coverage=None,
):
    """Monitor risk over the labeled prefix and coverage over the full stream."""
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    if grader is not None:
        if risk_target is None or not 0 <= risk_target <= 1:
            raise ValueError("risk target must be between 0 and 1")
        if not 0 < failure_rate < 1:
            raise ValueError("failure rate must be strictly between 0 and 1")
    elif coverage_window is None:
        raise ValueError("monitoring without a grader requires a coverage check")
    if (coverage_window is None) != (min_coverage is None):
        raise ValueError("--coverage-window and --min-coverage must be given together")
    if coverage_window is not None and (
        type(coverage_window) is not int or coverage_window < 1
    ):
        raise ValueError("coverage window must be a positive integer")
    if min_coverage is not None and not 0 <= min_coverage <= 1:
        raise ValueError("minimum coverage must be between 0 and 1")
    if routed:
        # The decision reader validates scores, IDs, decisions, and output keys.
        if not rows:
            raise ValueError("at least one decision is required")
        if grader is not None and labels is None:
            raise ValueError(
                "monitoring risk from decisions requires a separate label file"
            )
    elif grader is not None and labels is None:
        rows = validated_rows(
            rows, (grader,), review_missing_scores=review_missing_scores
        )
    else:
        rows = validated_rows(
            rows, (), minimum_graders=0, review_missing_scores=review_missing_scores
        )
    if labels is not None and (
        not grader or any(type(value) is not bool for value in labels.values())
    ):
        raise ValueError("provide a grader name and boolean labels")
    coverage = (
        _check_coverage(
            rows,
            threshold=threshold,
            routed=routed,
            window_size=coverage_window,
            min_coverage=min_coverage,
        )
        if coverage_window is not None
        else None
    )
    if grader is None:
        total = len(rows)
        accepted = sum(
            row.accepted if routed else accepted_at(row, threshold) for row in rows
        )
        return {
            "threshold": threshold,
            "overall": {
                "total": total,
                "accepted": accepted,
                "coverage": accepted / total,
                "missing_scores": sum(row.score is None for row in rows),
            },
            "coverage_check": coverage,
        }
    test = (
        _MixtureTest(risk_target, failure_rate, len(rows))
        if 0 < risk_target < 1
        else None
    )
    accepted = errors = 0
    stream_accepted = unlabeled_accepted = missing_scores = 0
    first_alert = None
    first_unlabeled = None
    checked_rows = len(rows)
    for index, row in enumerate(rows, 1):
        if routed or review_missing_scores:
            missing_scores += row.score is None
        is_accepted = row.accepted if routed else accepted_at(row, threshold)
        if not is_accepted:
            continue
        if labels is None:
            correct = row.outcomes[0]
        else:
            stream_accepted += 1
            correct = labels.get(row.id)
            if correct is None:
                unlabeled_accepted += 1
                if first_unlabeled is None:
                    checked_rows = index - 1
                    first_unlabeled = {"id": row.id, "row": index}
            if first_unlabeled is not None:
                continue
        accepted += 1
        error = not correct
        errors += error
        alert = test.update(error) if test is not None else risk_target == 0 and error
        if first_alert is None and alert:
            first_alert = {
                "id": row.id,
                "row": index,
                "accepted": accepted,
                "errors": errors,
            }
    overall = (
        Counts(checked_rows, accepted, errors).metrics()
        if checked_rows
        else {
            "total": 0,
            "accepted": 0,
            "errors": 0,
            "coverage": None,
            "error_rate": None,
        }
    )
    report = {
        "threshold": threshold,
        "risk_target": risk_target,
        "failure_rate": failure_rate,
        "grader": grader,
        "overall": overall,
        "anytime_pvalue": test.pvalue()
        if test is not None
        else float(first_alert is None),
        "first_alert": first_alert,
    }
    if labels is not None:
        report["stream"] = {
            "total": len(rows),
            "accepted": stream_accepted,
            "coverage": stream_accepted / len(rows),
            "unlabeled_accepted": unlabeled_accepted,
            "first_unlabeled": first_unlabeled,
        }
        if routed or review_missing_scores:
            report["stream"]["missing_scores"] = missing_scores
    elif review_missing_scores:
        overall["missing_scores"] = missing_scores
    if coverage is not None:
        report["coverage_check"] = coverage
    return report
