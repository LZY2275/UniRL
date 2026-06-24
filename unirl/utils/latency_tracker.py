"""Per-step phase latency tracker — P50/P95/P99 for training-loop analysis.

Accumulates wall-clock samples (seconds) per phase name across multiple
train steps, then computes percentile statistics so driver/engine/train
latency can be monitored for tail-latency analysis (e.g. rollout rollout
long-tail issues).

Usage::

    tracker = LatencyTracker(max_samples=5000)
    tracker.add("generate", 2.3)   # seconds
    tracker.add("generate", 1.8)
    stats = tracker.get_percentile_stats(clear=True)
    # → {"generate_mean_ms": 2050.0, "generate_p50_ms": 1800.0, ...}
"""

from __future__ import annotations

from typing import Dict, List


class LatencyTracker:
    """Accumulate phase-latency samples over multiple train steps.

    ``add(name, seconds)`` after each step; ``get_percentile_stats()``
    returns P50/P95/P99 plus mean in **milliseconds**, then clears the
    buffer so the next window starts fresh.
    """

    def __init__(self, max_samples: int = 10000) -> None:
        self._max_samples = max(1, max_samples)
        self._samples: Dict[str, List[float]] = {}

    def add(self, name: str, value_s: float) -> None:
        """Record one latency sample (in seconds) under *name*."""
        samples = self._samples.setdefault(name, [])
        samples.append(value_s)
        # Bounded FIFO: oldest sample is dropped when over capacity so
        # a long-running run doesn't accumulate unbounded history.
        if len(samples) > self._max_samples:
            samples.pop(0)

    def get_percentile_stats(self, *, clear: bool = True) -> Dict[str, float]:
        """Compute percentile stats for every tracked phase.

        Returns a flat dict keyed like ``{phase}_mean_ms``,
        ``{phase}_p50_ms``, ``{phase}_p95_ms``, ``{phase}_p99_ms``,
        ``{phase}_count`` (number of samples window).

        When *clear* is True (default) the sample buffer is emptied after
        computing, so each window covers fresh steps.
        """
        stats: Dict[str, float] = {}
        for name, samples in self._samples.items():
            if not samples:
                continue
            n = len(samples)
            sorted_s = sorted(samples)
            stats[f"{name}_mean_ms"] = (sum(sorted_s) / n) * 1000.0
            p50_idx = min(int(n * 0.50), n - 1)
            stats[f"{name}_p50_ms"] = sorted_s[p50_idx] * 1000.0
            p95_idx = min(int(n * 0.95), n - 1)
            stats[f"{name}_p95_ms"] = sorted_s[p95_idx] * 1000.0
            p99_idx = min(int(n * 0.99), n - 1)
            stats[f"{name}_p99_ms"] = sorted_s[p99_idx] * 1000.0
            stats[f"{name}_count"] = float(n)
        if clear:
            self._samples.clear()
        return stats
