from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    pose_ms: float = 0.0
    gesture_ms: float = 0.0
    total_ms: float = 0.0
    cpu_percent: float | None = None
    memory_mb: float | None = None


class PerformanceTracker:
    def __init__(self, window: int = 30):
        self.window = window
        self._frame_seconds: list[float] = []
        self._pose_ms: list[float] = []
        self._gesture_ms: list[float] = []
        self._total_ms: list[float] = []
        self._last_frame = perf_counter()

    def update(self, pose_ms: float, gesture_ms: float, total_ms: float) -> PerformanceMetrics:
        now = perf_counter()
        self._frame_seconds.append(now - self._last_frame)
        self._last_frame = now
        self._pose_ms.append(pose_ms)
        self._gesture_ms.append(gesture_ms)
        self._total_ms.append(total_ms)
        for values in (self._frame_seconds, self._pose_ms, self._gesture_ms, self._total_ms):
            del values[:-self.window]
        return PerformanceMetrics(
            fps=1.0 / (sum(self._frame_seconds) / len(self._frame_seconds)),
            pose_ms=sum(self._pose_ms) / len(self._pose_ms),
            gesture_ms=sum(self._gesture_ms) / len(self._gesture_ms),
            total_ms=sum(self._total_ms) / len(self._total_ms),
        )


def process_resource_metrics() -> tuple[float | None, float | None]:
    try:
        import os
        import psutil

        process = psutil.Process(os.getpid())
        return process.cpu_percent(None), process.memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        return None, None
