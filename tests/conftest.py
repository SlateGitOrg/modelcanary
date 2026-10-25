"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modelcanary.engine.policy import RolloutPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "default.toml"


@pytest.fixture
def default_policy() -> RolloutPolicy:
    return RolloutPolicy.from_toml(DEFAULT_POLICY)


def small_policy(**overrides: Any) -> RolloutPolicy:
    """A compact two-stage policy for fast controller tests."""
    data: dict[str, Any] = {
        "false_rollback_budget": 0.05,
        "false_promotion_budget": 0.05,
        "check_every": 1,
        "guardrails": {"max_requests": 50_000, "min_failure_attempts": 20},
        "stages": [
            {"name": "shadow", "weight": 0.0, "shadow": True, "min_requests": 100},
            {"name": "canary", "weight": 0.5},
        ],
        "metrics": [
            {
                "name": "error_rate",
                "source": "error",
                "direction": "lower_is_better",
                "method": "sprt",
                "odds_ratio": 2.0,
                "tolerance": 0.05,
                "planning_sd": 0.4,
            },
            {
                "name": "latency_ms",
                "source": "latency",
                "direction": "lower_is_better",
                "method": "asymptotic_cs",
                "tolerance": 2.0,
                "min_pairs": 20,
                "planning_sd": 5.0,
            },
        ],
    }
    data.update(overrides)
    return RolloutPolicy.from_mapping(data)
