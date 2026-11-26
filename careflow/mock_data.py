"""Read-only access to the mock hospital tables (CSV files).

This is the "mock system" the tools talk to. Each table is read once
and cached. Rows are returned as plain dictionaries of text values.
"""

import csv
from functools import lru_cache

from careflow import config


@lru_cache(maxsize=None)
def load_table(name: str) -> tuple[dict, ...]:
    """Return all rows of mock_api/<name>.csv as a tuple of dicts."""
    path = config.MOCK_API_DIR / f"{name}.csv"
    with open(path, "r", encoding="utf-8", newline="") as file:
        return tuple(csv.DictReader(file))


def find_row(table: str, **conditions) -> dict | None:
    """Return the first row where every column equals the given value."""
    for row in load_table(table):
        if all(
            row.get(column, "").strip() == str(value).strip()
            for column, value in conditions.items()
        ):
            return row
    return None