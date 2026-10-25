"""Policy validation and FIFO pairing."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modelcanary.engine.pairing import Arm, PairBuffer
from modelcanary.engine.policy import Method, MetricSource, RolloutPolicy
from modelcanary.errors import ConfigError
from tests.conftest import small_policy


def test_default_policy_loads(default_policy: RolloutPolicy) -> None:
    assert default_policy.stages[0].name == "shadow"
    assert {m.source for m in default_policy.metrics} == set(MetricSource)
    assert default_policy.metric_alpha_rollback == pytest.approx(0.05 / 3)
    assert default_policy.metric_alpha_promotion == pytest.approx(0.05 / 3)


def _metric(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "m",
        "source": "latency",
        "direction": "lower_is_better",
        "method": "asymptotic_cs",
        "tolerance": 1.0,
    }
    base.update(kw)
    return base


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (_metric(method="sprt"), "only defined for the binary"),
        (_metric(source="error", method="sprt"), "requires odds_ratio"),
        (
            _metric(source="error", method="sprt", odds_ratio=1.5, direction="higher_is_better"),
            "lower_is_better",
        ),
        (_metric(method="betting_cs"), "requires bounds"),
        (_metric(method="betting_cs", bounds=[1.0, 0.0]), "hi > lo"),
        (_metric(name="Bad Name"), "pattern"),
        (_metric(tolerance=0), "greater than 0"),
    ],
)
def test_invalid_metric_specs(metric: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        small_policy(metrics=[metric])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stages": [{"name": "s", "weight": 0.1, "shadow": True}]}, "serves no candidate"),
        ({"stages": [{"name": "s", "weight": 0.0}]}, "needs weight > 0"),
        ({"promote_min_stage": 5}, "promote_min_stage"),
        ({"false_rollback_budget": 0.6, "false_promotion_budget": 0.5}, "sum to < 1"),
        ({"metrics": [_metric(), _metric()]}, "unique"),
        ({"metrics": [_metric(), _metric(name="other")]}, "one metric per source"),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_invalid_policies(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        small_policy(**overrides)


def test_with_metric_updates_validates() -> None:
    policy = small_policy()
    updated = policy.with_metric_updates({"latency_ms": {"planning_sd": 9.0}})
    assert updated.metrics[1].planning_sd == 9.0
    assert policy.metrics[1].planning_sd == 5.0  # original untouched (frozen)
    with pytest.raises(ConfigError, match="unknown metrics"):
        policy.with_metric_updates({"nope": {}})


def test_from_toml_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        RolloutPolicy.from_toml(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read"):
        RolloutPolicy.from_toml(bad)


def test_method_enum_values_are_stable() -> None:
    assert {m.value for m in Method} == {"sprt", "asymptotic_cs", "betting_cs"}


def test_pair_buffer_matches_fifo() -> None:
    buf = PairBuffer(capacity=10)
    b, c = buf.push(Arm.BASELINE, [1, 2, 3])
    assert b.size == 0
    assert c.size == 0
    b, c = buf.push(Arm.CANDIDATE, [10, 20])
    np.testing.assert_array_equal(b, [1, 2])
    np.testing.assert_array_equal(c, [10, 20])
    assert buf.backlog(Arm.BASELINE) == 1
    b, c = buf.push(Arm.CANDIDATE, [30, 40])
    np.testing.assert_array_equal(b, [3])
    np.testing.assert_array_equal(c, [30])
    assert buf.backlog(Arm.CANDIDATE) == 1
    assert buf.paired == 3


def test_pair_buffer_drops_oldest_surplus() -> None:
    buf = PairBuffer(capacity=3)
    buf.push(Arm.BASELINE, np.arange(10))
    assert buf.backlog(Arm.BASELINE) == 3
    assert buf.dropped == 7
    b, _ = buf.push(Arm.CANDIDATE, [0.0])
    np.testing.assert_array_equal(b, [7.0])  # newest surplus kept
    with pytest.raises(ConfigError):
        PairBuffer(capacity=0)


def test_rollback_margin_validation() -> None:
    with pytest.raises(ConfigError, match="must not exceed"):
        small_policy(metrics=[_metric(rollback_margin=2.0)])
    sprt = _metric(source="error", method="sprt", odds_ratio=1.5, rollback_margin=0.01)
    with pytest.raises(ConfigError, match="odds_ratio"):
        small_policy(metrics=[sprt])
