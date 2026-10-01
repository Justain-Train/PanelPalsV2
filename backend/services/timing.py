"""
Per-request stage timing.

Lap-style stopwatch for the chapter pipeline: call lap("ocr") when the OCR
stage finishes and the time since the previous lap is recorded under "ocr".
Stages run sequentially, so the laps add up to the request total.

Output:
- summary():              "read=0.3s ocr=11.8s tts=24.1s total=39.0s" for logs
- server_timing_header(): "read;dur=312, ocr;dur=11812, ..." for the standard
                          Server-Timing response header (milliseconds), which
                          browser dev tools display per request
"""

import time
from typing import Dict


class StageTimer:
    """Records how long each pipeline stage takes."""

    def __init__(self):
        self._start = time.perf_counter()
        self._last = self._start
        self.stages: Dict[str, float] = {}  # stage name → seconds, in order recorded

    def lap(self, stage: str) -> float:
        """Close the current stage, record its duration, and start the next one."""
        now = time.perf_counter()
        elapsed = now - self._last
        # A stage lapped twice (e.g. a retry) accumulates rather than overwrites
        self.stages[stage] = self.stages.get(stage, 0.0) + elapsed
        self._last = now
        return elapsed

    @property
    def total(self) -> float:
        """Seconds since the timer was created."""
        return time.perf_counter() - self._start

    def summary(self) -> str:
        parts = [f"{name}={secs:.2f}s" for name, secs in self.stages.items()]
        parts.append(f"total={self.total:.2f}s")
        return " ".join(parts)

    def server_timing_header(self) -> str:
        parts = [f"{name};dur={secs * 1000:.0f}" for name, secs in self.stages.items()]
        parts.append(f"total;dur={self.total * 1000:.0f}")
        return ", ".join(parts)
