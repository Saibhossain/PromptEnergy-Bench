"""High-precision latency measurement and phase tracking."""

import time
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class LatencyProfile:
    model_load_time_ms: Optional[float] = None
    warmup_time_ms: Optional[float] = None
    retrieval_latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    total_latency_ms: float = 0.0


class LatencyTracker:
    """Utility for measuring discrete execution phases using time.perf_counter()."""

    def __init__(self):
        self.timers: Dict[str, float] = {}

    def start(self, phase_name: str) -> None:
        self.timers[phase_name] = time.perf_counter()

    def stop(self, phase_name: str) -> float:
        if phase_name not in self.timers:
            raise KeyError(f"Timer for phase '{phase_name}' was not started.")
        elapsed = (time.perf_counter() - self.timers.pop(phase_name)) * 1000.0
        return round(elapsed, 2)
