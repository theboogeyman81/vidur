"""Latency percentiles. Report p50 and p95 — never just the mean."""

from __future__ import annotations


def percentile(values: list[float], p: int) -> float:
    """Linear-interpolated percentile, matching numpy.percentile's default method."""
    if not values:
        raise ValueError("percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)
