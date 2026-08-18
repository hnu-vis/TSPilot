"""Policy constants for the lightweight Python analysis sandbox."""
from __future__ import annotations

MAX_STDIO_CHARS = 12000
MAX_RESULT_BYTES = 1_000_000


def validate_timeout(value: int | float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sandbox timeout must be a positive number of seconds") from exc
    if timeout <= 0:
        raise ValueError("sandbox timeout must be a positive number of seconds")
    return timeout
