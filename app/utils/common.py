from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utc_filename_timestamp() -> str:
    """Return a collision-resistant UTC timestamp suitable for file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def chunk_list(items: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
