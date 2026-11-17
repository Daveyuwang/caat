"""Input adapters produce graded rows for CAAT's analysis commands."""

from dataclasses import dataclass
from typing import Callable

from .registry import Registry
from .table import read_table_rows


@dataclass(frozen=True)
class Adapter:
    read: Callable
    source_format: str
    source_version: str
    fixture: Callable | None = None


def csv_rows(path, options):
    rows, _ = read_table_rows(path, **options)
    return rows


ADAPTERS = Registry("adapters", Adapter, {"csv": Adapter(csv_rows, "CSV", "1")})
