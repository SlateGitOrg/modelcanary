"""Evaluation harness and pilot tuning."""

import pytest

from modelcanary.engine.controller import Action
from modelcanary.engine.monitors import Strategy
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.evaluation.harness import (
    EVAL_SEED_START,
    Rate,
    eval_seeds,
    render_markdown,
    run_grid,
    summarize_scenarios,
    summarize_strategies,
    wilson_interval,
)
from modelcanary.evaluation.tuning import pilot_planning_sd, tune_policy
from modelcanary.sim.scenarios import BASELINE, standard_scenarios


def test_wilson_interval_reference_values() -> None:
    # Reference: Wilson score interval for 0/10 and 5/10 at z = 1.96.
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_rate_formatting() -> None:
    assert Rate(0, 0).fmt() == "n/a"
    assert Rate(1, 4).fmt().startswith("0.250 [")
    assert Rate(1, 4).to_dict()["count"] == 1


def test_eval_seeds_are_disjoint_from_tuning() -> None:
    seeds = eval_seeds(3)
    assert seeds == [EVAL_SEED_START, EVAL_SEED_START + 1, EVAL_SEED_START + 2]
    with pytest.raises(ConfigError, match="overlap"):
        eval_seeds(3, start=10)


def test_run_grid_serial_equals_parallel(default_policy: RolloutPolicy) -> None:
    sc = standard_scenarios()
    scenarios = [sc["worse-latency"], sc["broken"]]
    seeds = eval_seeds(4)
    serial = run_grid(default_policy, scenarios, list(Strategy), seeds, workers=1, chunk=2)
    parallel = run_grid(default_policy, scenarios, list(Strategy), seeds, workers=2, chunk=2)
    key = lambda r: (r.scenario, r.strategy.value, r.seed)  # noqa: E731
    assert sorted(serial, key=key) == sorted(parallel, key=key)
    assert len(serial) == 2 * 3 * 4
    assert all(r.action is Action.ROLLBACK for r in serial)

    strategies = summarize_strategies(serial)
    assert {s.strategy for s in strategies} == set(Strategy)
    for s in strategies:
        assert s.false_promotion.count == 0
        assert s.false_rollback.n == 0  # no good scenarios in this grid
    per_scenario = summarize_scenarios(serial)
    assert len(per_scenario) == 6
    md = render_markdown(strategies, per_scenario, "## test")
    assert md.startswith("## test")
    assert "| broken | worse | sequential | 4 |" in md
    assert per_scenario[0].to_dict()["runs"] == 4
    assert strategies[0].to_dict()["false_rollback"]["n"] == 0


def test_pilot_sd_matches_analytic_values(default_policy: RolloutPolicy) -> None:
    sds = pilot_planning_sd(default_policy, BASELINE, seeds=[1, 2], pairs_per_seed=50_000)
    # Difference of two independent Bernoulli(p): sd = sqrt(2 p (1 - p)).
    assert sds["error_rate"] == pytest.approx((2 * 0.10 * 0.90) ** 0.5, rel=0.02)
    assert sds["conversion"] == pytest.approx((2 * 0.12 * 0.88) ** 0.5, rel=0.02)
    assert sds["latency_ms"] == pytest.approx(BASELINE.latency_sd_ms * 2**0.5, rel=0.03)


def test_tune_policy_rejects_eval_seeds(default_policy: RolloutPolicy) -> None:
    with pytest.raises(ConfigError, match="tuning range"):
        tune_policy(default_policy, BASELINE, [EVAL_SEED_START])
    with pytest.raises(ConfigError, match="at least one"):
        tune_policy(default_policy, BASELINE, [])
    tuned, sds = tune_policy(default_policy, BASELINE, [3], pairs_per_seed=1000)
    assert tuned.metrics[0].planning_sd == pytest.approx(sds["error_rate"], abs=1e-6)
