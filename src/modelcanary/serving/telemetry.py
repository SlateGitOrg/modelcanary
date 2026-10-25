"""Minimal Prometheus text-format counters for the gateway (no client library needed)."""

from __future__ import annotations

import threading
from collections import defaultdict

_HELP = {
    "modelcanary_predictions_total": "Predictions served, by model and arm.",
    "modelcanary_inference_failures_total": "Model inference failures, by model.",
    "modelcanary_shadow_scored_total": "Requests scored by the shadow candidate.",
    "modelcanary_feedback_total": "Feedback items received, by result.",
    "modelcanary_rollout_decisions_total": "Terminal rollout decisions, by action.",
}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Counters:
    """Thread-safe labelled counters rendered in the Prometheus exposition format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._values[name]
            series[key] = series.get(key, 0.0) + amount

    def value(self, name: str, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(name, {}).get(key, 0.0)

    def render(self, gauges: dict[str, float] | None = None) -> str:
        lines: list[str] = []
        with self._lock:
            snapshot = {name: dict(series) for name, series in self._values.items()}
        for name in sorted(snapshot):
            lines.append(f"# HELP {name} {_HELP.get(name, name)}")
            lines.append(f"# TYPE {name} counter")
            for key, val in sorted(snapshot[name].items()):
                label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in key)
                suffix = f"{{{label_str}}}" if label_str else ""
                lines.append(f"{name}{suffix} {val:g}")
        for name, val in sorted((gauges or {}).items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {val:g}")
        return "\n".join(lines) + "\n"
