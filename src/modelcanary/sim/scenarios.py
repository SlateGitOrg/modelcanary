"""Ground-truth scenarios for the rollout simulator.

A scenario fixes the *true* per-request behaviour of both arms, so whether the
candidate is better, equal or worse is known exactly and every rollout decision
can be scored as correct or not.  Truth is derived from the scenario and the
policy's tolerances rather than hand-labelled (:func:`derive_truth`), so a
scenario cannot silently be mislabelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from modelcanary.engine.policy import Method, MetricSource, RolloutPolicy
from modelcanary.errors import ConfigError


class Truth(StrEnum):
    BETTER = "better"  # no metric degraded, at least one improved
    EQUAL = "equal"  # identical behaviour
    WORSE = "worse"  # at least one metric degraded by >= its tolerance (or failures)
    INDIFFERENT = "indifferent"  # some degradation inside (0, tolerance): no right answer


@dataclass(frozen=True, slots=True)
class ArmProfile:
    """True behaviour of one model version."""

    error_rate: float
    latency_median_ms: float
    latency_sigma: float  # lognormal shape parameter
    conversion_rate: float
    failure_rate: float = 0.0  # probability an inference call raises

    def __post_init__(self) -> None:
        for name in ("error_rate", "conversion_rate", "failure_rate"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ConfigError(f"{name} must be in [0, 1), got {v}")
        if self.latency_median_ms <= 0 or self.latency_sigma <= 0:
            raise ConfigError("latency parameters must be positive")

    @property
    def latency_mean_ms(self) -> float:
        return self.latency_median_ms * math.exp(self.latency_sigma**2 / 2.0)

    @property
    def latency_sd_ms(self) -> float:
        return self.latency_mean_ms * math.sqrt(math.exp(self.latency_sigma**2) - 1.0)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    baseline: ArmProfile
    candidate: ArmProfile
    error_correlation: float = 0.5  # shadow mode: P(both models see the same "difficulty")
    latency_correlation: float = 0.5  # shadow mode: shared per-request latency factor
    description: str = ""

    def true_degradation(self, source: MetricSource) -> float:
        """Mean per-pair degradation (positive = candidate worse) in metric units."""
        b, c = self.baseline, self.candidate
        if source is MetricSource.ERROR:
            return c.error_rate - b.error_rate
        if source is MetricSource.LATENCY:
            return c.latency_mean_ms - b.latency_mean_ms
        return b.conversion_rate - c.conversion_rate

    def error_odds_ratio(self) -> float:
        b, c = self.baseline.error_rate, self.candidate.error_rate
        return (c / (1 - c)) / (b / (1 - b))


def derive_truth(scenario: Scenario, policy: RolloutPolicy) -> Truth:
    """Classify a scenario against the policy's tolerances."""
    if scenario.candidate.failure_rate > policy.guardrails.max_candidate_failure_rate:
        return Truth.WORSE
    any_better = False
    indifferent = False
    for spec in policy.metrics:
        delta = scenario.true_degradation(spec.source)
        worse_by_margin = delta >= spec.tolerance - 1e-12
        if spec.method is Method.SPRT and spec.odds_ratio is not None:
            worse_by_margin = worse_by_margin and scenario.error_odds_ratio() >= spec.odds_ratio
        if worse_by_margin:
            return Truth.WORSE
        if delta > 1e-12:
            indifferent = True
        elif delta < -1e-12:
            any_better = True
    if indifferent:
        return Truth.INDIFFERENT
    return Truth.BETTER if any_better else Truth.EQUAL


BASELINE = ArmProfile(
    error_rate=0.10, latency_median_ms=40.0, latency_sigma=0.6, conversion_rate=0.12
)


def standard_scenarios() -> dict[str, Scenario]:
    """The scenario suite used by the evaluation (better / equal / worse variants)."""
    b = BASELINE

    def with_(**kw: float) -> ArmProfile:
        fields = {
            "error_rate": b.error_rate,
            "latency_median_ms": b.latency_median_ms,
            "latency_sigma": b.latency_sigma,
            "conversion_rate": b.conversion_rate,
            "failure_rate": b.failure_rate,
            **kw,
        }
        return ArmProfile(**fields)

    scenarios = [
        Scenario(
            "better",
            b,
            with_(error_rate=0.08, latency_median_ms=38.0, conversion_rate=0.13),
            description="lower error, faster, higher conversion",
        ),
        Scenario(
            "better-error-only",
            b,
            with_(error_rate=0.085),
            description="lower error; latency and conversion unchanged",
        ),
        Scenario("equal", b, b, description="candidate identical to baseline"),
        Scenario(
            "worse-error",
            b,
            with_(error_rate=0.13),
            description="error 10% -> 13% (odds ratio 1.34)",
        ),
        Scenario(
            "worse-latency",
            b,
            with_(latency_median_ms=44.0),
            description="mean latency +4.8 ms (tolerance 3 ms)",
        ),
        Scenario(
            "worse-conversion",
            b,
            with_(conversion_rate=0.095),
            description="conversion 12% -> 9.5% (tolerance 2 points)",
        ),
        Scenario(
            "broken",
            b,
            with_(failure_rate=0.05),
            description="5% of candidate inference calls raise",
        ),
    ]
    return {s.name: s for s in scenarios}
