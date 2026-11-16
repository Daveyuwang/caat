"""Compatibility imports for the original selection module."""

from .rules import RULES as _REGISTRY
from .rules import binomial_pvalue, log_lower_tail
from .select import design_thresholds, select_threshold

RULES = tuple(_REGISTRY)
