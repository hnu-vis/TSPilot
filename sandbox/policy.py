"""Policy constants for the lightweight Python analysis sandbox."""
from __future__ import annotations

DEFAULT_TIMEOUT_SECONDS = 5
MAX_TIMEOUT_SECONDS = 30
MAX_STDIO_CHARS = 12000
MAX_RESULT_BYTES = 1_000_000


def clamp_timeout(value: int | float | None) -> int:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, min(timeout, MAX_TIMEOUT_SECONDS))
