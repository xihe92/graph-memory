"""The wall clock every time-dependent decision reads: item decay, dormancy, idle
graduation, exchange and event timestamps.

It is real time plus an offset. Normally the offset is zero; the simulator advances it to
replay weeks of use in seconds (`advance(days=30)`). Durations measured for latency and
the daemon's own idle shutdown use time.monotonic() and are not affected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

_offset = timedelta(0)


def now() -> datetime:
    return datetime.now(UTC) + _offset


def advance(delta: timedelta | None = None, **kw: float) -> datetime:
    """Move the clock forward (simulation only). Returns the new time."""
    global _offset
    _offset += delta if delta is not None else timedelta(**kw)
    return now()


def reset() -> None:
    global _offset
    _offset = timedelta(0)


def offset() -> timedelta:
    return _offset
