"""Pilot-based planning for the fixed-horizon comparator.

A fixed-horizon test needs the standard deviation of the per-pair degradation
*before* the experiment starts.  In practice it comes from a pilot on the
baseline alone (an A/A comparison).  :func:`pilot_planning_sd` reproduces that:
it simulates baseline-vs-baseline canary pairs on seeds from the reserved
tuning range only, so nothing estimated here is ever evaluated on the same
randomness the evaluation uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from modelcanary.engine.policy import Direction, MetricSource, RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.evaluation.harness import TUNING_SEED_RANGE
from modelcanary.sim.runner import TrafficGenerator
from modelcanary.sim.scenarios import ArmProfile, Scenario


def _draw(gen: TrafficGenerator, source: MetricSource, profile: ArmProfile, n: int) -> np.ndarray:
    if source is MetricSource.ERROR:
        return gen.errors(profile, n)
    if source is MetricSource.LATENCY:
        return gen.latencies(profile, n)
    return gen.conversions(profile, n)


def pilot_planning_sd(
    policy: RolloutPolicy,
    baseline: ArmProfile,
    seeds: Sequence[int],
    pairs_per_seed: int = 20_000,
) -> dict[str, float]:
    """Standard deviation of A/A canary degradations per metric, pooled over ``seeds``."""
    if not seeds:
        raise ConfigError("need at least one pilot seed")
    bad = [s for s in seeds if s not in TUNING_SEED_RANGE]
    if bad:
        raise ConfigError(f"pilot seeds must come from the tuning range, got {bad[:3]}")
    scenario = Scenario("pilot-aa", baseline, baseline)
    diffs: dict[str, list[np.ndarray]] = {m.name: [] for m in policy.metrics}
    for seed in seeds:
        gen = TrafficGenerator(scenario, np.random.default_rng(seed))
        for spec in policy.metrics:
            a = _draw(gen, spec.source, baseline, pairs_per_seed)
            b = _draw(gen, spec.source, baseline, pairs_per_seed)
            sign = 1.0 if spec.direction is Direction.LOWER_IS_BETTER else -1.0
            diffs[spec.name].append(sign * (b - a))
    return {name: float(np.concatenate(parts).std(ddof=1)) for name, parts in diffs.items()}


def tune_policy(
    policy: RolloutPolicy, baseline: ArmProfile, seeds: Sequence[int], pairs_per_seed: int = 20_000
) -> tuple[RolloutPolicy, dict[str, float]]:
    """Return ``policy`` with every metric's ``planning_sd`` set from a pilot."""
    sds = pilot_planning_sd(policy, baseline, seeds, pairs_per_seed)
    updates = {name: {"planning_sd": round(sd, 6)} for name, sd in sds.items()}
    return policy.with_metric_updates(updates), sds
