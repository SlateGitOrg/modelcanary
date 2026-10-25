"""Ground-truth simulator."""

import math

import numpy as np
import pytest

from modelcanary.audit import AuditLog
from modelcanary.engine.controller import Action
from modelcanary.engine.monitors import Strategy
from modelcanary.engine.policy import MetricSource, RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.sim.runner import TrafficGenerator, simulate_rollout
from modelcanary.sim.scenarios import (
    BASELINE,
    ArmProfile,
    Scenario,
    Truth,
    derive_truth,
    standard_scenarios,
)

EXPECTED_TRUTH = {
    "better": Truth.BETTER,
    "better-error-only": Truth.BETTER,
    "equal": Truth.EQUAL,
    "worse-error": Truth.WORSE,
    "worse-latency": Truth.WORSE,
    "worse-conversion": Truth.WORSE,
    "broken": Truth.WORSE,
}


def test_standard_scenarios_have_expected_truth(default_policy: RolloutPolicy) -> None:
    scenarios = standard_scenarios()
    assert set(scenarios) == set(EXPECTED_TRUTH)
    for name, sc in scenarios.items():
        assert derive_truth(sc, default_policy) is EXPECTED_TRUTH[name], name


def test_indifferent_zone(default_policy: RolloutPolicy) -> None:
    slightly_slower = Scenario(
        "slow-1ms", BASELINE, ArmProfile(0.10, 41.0, 0.6, 0.12), description="inside tolerance"
    )
    assert derive_truth(slightly_slower, default_policy) is Truth.INDIFFERENT


def test_error_worse_needs_both_margin_and_odds_ratio(default_policy: RolloutPolicy) -> None:
    # +2.1 points but at a high base rate the odds ratio is only ~1.1 < 1.25.
    high = ArmProfile(0.50, 40.0, 0.6, 0.12)
    sc = Scenario("high-base", high, ArmProfile(0.521, 40.0, 0.6, 0.12))
    assert sc.error_odds_ratio() < 1.25
    assert derive_truth(sc, default_policy) is Truth.INDIFFERENT


def test_arm_profile_validation_and_moments() -> None:
    with pytest.raises(ConfigError):
        ArmProfile(1.2, 40.0, 0.6, 0.1)
    with pytest.raises(ConfigError):
        ArmProfile(0.1, -1.0, 0.6, 0.1)
    p = ArmProfile(0.1, 40.0, 0.6, 0.1)
    assert p.latency_mean_ms == pytest.approx(40.0 * math.exp(0.18))
    rng = np.random.default_rng(0)
    gen = TrafficGenerator(Scenario("x", p, p), rng)
    lat = gen.latencies(p, 400_000)
    assert lat.mean() == pytest.approx(p.latency_mean_ms, rel=0.01)
    assert lat.std() == pytest.approx(p.latency_sd_ms, rel=0.03)


def test_paired_draws_keep_marginals_and_add_correlation() -> None:
    sc = standard_scenarios()["worse-error"]
    gen = TrafficGenerator(sc, np.random.default_rng(1))
    b, c = gen.paired_errors(300_000)
    assert b.mean() == pytest.approx(0.10, abs=0.003)
    assert c.mean() == pytest.approx(0.13, abs=0.003)
    assert np.corrcoef(b, c)[0, 1] > 0.3
    lb, lc = gen.paired_latencies(300_000)
    assert lb.mean() == pytest.approx(sc.baseline.latency_mean_ms, rel=0.01)
    assert np.corrcoef(np.log(lb), np.log(lc))[0, 1] == pytest.approx(0.5, abs=0.02)


def test_true_degradation_signs() -> None:
    sc = standard_scenarios()["better"]
    assert sc.true_degradation(MetricSource.ERROR) < 0
    assert sc.true_degradation(MetricSource.LATENCY) < 0
    assert (
        sc.true_degradation(MetricSource.BUSINESS) < 0
    )  # higher conversion = negative degradation


def test_simulation_is_deterministic_and_audited(default_policy: RolloutPolicy) -> None:
    sc = standard_scenarios()["worse-error"]
    audit = AuditLog()
    a = simulate_rollout(sc, default_policy, 1_000_123, audit=audit)
    b = simulate_rollout(sc, default_policy, 1_000_123)
    assert a == b
    assert a.action is Action.ROLLBACK
    assert a.truth is Truth.WORSE
    assert audit.verify().ok
    assert a.to_dict()["strategy"] == "sequential"


@pytest.mark.parametrize("strategy", list(Strategy))
def test_broken_candidate_hits_guardrail(default_policy: RolloutPolicy, strategy: Strategy) -> None:
    r = simulate_rollout(standard_scenarios()["broken"], default_policy, 1_000_001, strategy)
    assert r.action is Action.ROLLBACK
    assert r.reason.startswith("guardrail")
    assert r.candidate_requests == 0  # caught in shadow


def test_sequential_promotes_better_candidate(default_policy: RolloutPolicy) -> None:
    r = simulate_rollout(standard_scenarios()["better"], default_policy, 1_000_002)
    assert r.action is Action.PROMOTE
    assert r.stage != "shadow"  # promote_min_stage = 1
