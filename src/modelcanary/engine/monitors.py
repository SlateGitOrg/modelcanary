"""Per-metric monitors: map paired observations to a three-way verdict.

Every monitor consumes aligned pairs ``(baseline_i, candidate_i)`` and reports

* ``WORSE``      -- the candidate is confidently degraded (``delta > rollback_margin``,
  0 by default): roll back;
* ``NOT_WORSE``  -- the degradation is confidently below the metric's tolerance;
* ``CONTINUE``   -- not enough evidence yet.

``delta`` is the mean per-pair degradation (``candidate - baseline`` for
lower-is-better metrics).  Three strategies share this interface so that the
evaluation compares them on identical traffic:

* :data:`Strategy.SEQUENTIAL` -- SPRT / confidence sequences (always valid);
* :data:`Strategy.FIXED_HORIZON` -- one z-test at a pre-registered sample size;
* :data:`Strategy.PEEKING` -- the same z-tests repeated at every check (invalid,
  included to quantify why sequential methods are needed).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modelcanary.engine.policy import Direction, Method, MetricSpec
from modelcanary.errors import ConfigError
from modelcanary.stats.confseq import BettingCS, Interval, MixtureCS
from modelcanary.stats.fixed import non_inferiority_sample_size, paired_z_test_from_moments
from modelcanary.stats.running import RunningMoments
from modelcanary.stats.sprt import DiscordantPairSPRT, SPRTDecision


class Verdict(StrEnum):
    CONTINUE = "continue"
    NOT_WORSE = "not_worse"
    WORSE = "worse"


class Strategy(StrEnum):
    SEQUENTIAL = "sequential"
    FIXED_HORIZON = "fixed_horizon"
    PEEKING = "peeking"


def _finite(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


@dataclass(frozen=True, slots=True)
class MonitorSnapshot:
    """JSON-friendly state of a monitor, recorded in the audit trail."""

    metric: str
    method: str
    pairs: int
    baseline_mean: float | None
    candidate_mean: float | None
    degradation: float | None
    lower: float | None
    upper: float | None
    statistic: float | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Monitor(ABC):
    """Base class: tracks arm means and the degradation sign convention."""

    method_name: str = "abstract"

    def __init__(self, spec: MetricSpec, alpha_rollback: float, alpha_promotion: float) -> None:
        self.spec = spec
        self.alpha_rollback = alpha_rollback
        self.alpha_promotion = alpha_promotion
        self._sign = 1.0 if spec.direction is Direction.LOWER_IS_BETTER else -1.0
        self._baseline = RunningMoments()
        self._candidate = RunningMoments()
        self._verdict = Verdict.CONTINUE

    @property
    def verdict(self) -> Verdict:
        return self._verdict

    @property
    def pairs(self) -> int:
        return self._baseline.n

    def observe(self, baseline: ArrayLike, candidate: ArrayLike) -> None:
        base = np.asarray(baseline, dtype=np.float64).ravel()
        cand = np.asarray(candidate, dtype=np.float64).ravel()
        if base.shape != cand.shape:
            raise ValueError("baseline and candidate batches must have equal length")
        if base.size == 0 or self._verdict is Verdict.WORSE:
            return
        used = self._consume(base, cand)
        self._baseline.update_many(base[:used])
        self._candidate.update_many(cand[:used])

    def check(self) -> Verdict:
        """Re-evaluate at a controller check point (only peeking needs this hook)."""
        return self._verdict

    @abstractmethod
    def _consume(self, base: NDArray[np.float64], cand: NDArray[np.float64]) -> int:
        """Ingest a batch; return how many leading pairs were actually used."""

    def _degradation(
        self, base: NDArray[np.float64], cand: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._sign * (cand - base)

    def _interval(self) -> Interval | None:
        return None

    def _statistic(self) -> float | None:
        return None

    def snapshot(self) -> MonitorSnapshot:
        b, c = self._baseline.mean, self._candidate.mean
        degradation = self._sign * (c - b) if self.pairs else None
        iv = self._interval()
        return MonitorSnapshot(
            metric=self.spec.name,
            method=self.method_name,
            pairs=self.pairs,
            baseline_mean=_finite(b),
            candidate_mean=_finite(c),
            degradation=_finite(degradation),
            lower=_finite(iv.lower) if iv else None,
            upper=_finite(iv.upper) if iv else None,
            statistic=_finite(self._statistic()),
            verdict=self._verdict.value,
        )


class SPRTMonitor(Monitor):
    """Error-rate monitor: discordant-pair SPRT plus a discordance-rate shortcut.

    The SPRT only learns from discordant pairs, so a candidate that agrees with
    the baseline on (almost) every request -- a re-export, a refactor -- would
    never reach a decision.  Because the error-rate difference is bounded by the
    discordance rate (``p_c - p_b = P(c wrong, b right) - P(b wrong, c right)
    <= P(discordant)``), an always-valid upper bound on ``P(discordant)`` below
    the metric's ``tolerance`` proves non-inferiority directly.  The
    false-promotion budget is split equally between the SPRT's ``beta`` and that
    bound (union bound), so the combined guarantee is unchanged.

    ``WORSE`` and SPRT-``NOT_WORSE`` latch; shortcut-``NOT_WORSE`` keeps the SPRT
    running and is revised if the SPRT later accepts H1.
    """

    method_name = "sprt"

    def __init__(self, spec: MetricSpec, alpha_rollback: float, alpha_promotion: float) -> None:
        super().__init__(spec, alpha_rollback, alpha_promotion)
        if spec.odds_ratio is None:
            raise ConfigError("SPRT monitor needs odds_ratio")
        half = alpha_promotion / 2.0
        # alpha: P(declare worse | equal) ; beta: P(declare not-worse | OR = odds_ratio)
        self.test = DiscordantPairSPRT(spec.odds_ratio, alpha_rollback, half)
        self.discordance = BettingCS(half, lo=0.0, hi=1.0, grid_size=1000)

    def _consume(self, base: NDArray[np.float64], cand: NDArray[np.float64]) -> int:
        if self.test.decision is not SPRTDecision.CONTINUE:
            return 0
        before = self.test.pairs
        decision = self.test.update_pairs(base > 0.5, cand > 0.5)
        used = self.test.pairs - before
        self.discordance.update_many((base[:used] > 0.5) != (cand[:used] > 0.5))
        if decision is SPRTDecision.ACCEPT_H1:
            self._verdict = Verdict.WORSE
        elif decision is SPRTDecision.ACCEPT_H0 or (
            self.discordance.n >= self.spec.min_pairs
            and self.discordance.interval().upper < self.spec.tolerance
        ):
            self._verdict = Verdict.NOT_WORSE
        else:
            self._verdict = Verdict.CONTINUE
        return used

    def _statistic(self) -> float:
        return self.test.inner.llr


class ConfidenceSequenceMonitor(Monitor):
    """Monitor driven by an always-valid CS on the mean per-pair degradation.

    ``WORSE`` when the lower bound exceeds the metric's ``rollback_margin``
    (default 0); ``NOT_WORSE`` when the upper bound is below the tolerance.
    ``WORSE`` is absorbing, ``NOT_WORSE`` is not: the CS keeps running, and
    because its coverage is simultaneous over time the continued monitoring
    does not spend extra error budget.
    """

    def __init__(self, spec: MetricSpec, alpha_rollback: float, alpha_promotion: float) -> None:
        super().__init__(spec, alpha_rollback, alpha_promotion)
        alpha = min(alpha_rollback, alpha_promotion)
        self.cs: MixtureCS | BettingCS
        if spec.method is Method.BETTING_CS:
            if spec.bounds is None:
                raise ConfigError("betting CS needs bounds")
            width = spec.bounds[1] - spec.bounds[0]
            self.cs = BettingCS(alpha, lo=-width, hi=width, grid_size=spec.grid_size)
            self.method_name = "betting_cs"
        else:
            self.cs = MixtureCS(alpha, t_star=spec.t_star)
            self.method_name = "asymptotic_cs"

    def _consume(self, base: NDArray[np.float64], cand: NDArray[np.float64]) -> int:
        iv = self.cs.update_many(self._degradation(base, cand))
        if self.cs.n >= self.spec.min_pairs:
            if iv.lower > self.spec.rollback_margin:
                self._verdict = Verdict.WORSE
            elif iv.upper < self.spec.tolerance:
                self._verdict = Verdict.NOT_WORSE
            else:
                self._verdict = Verdict.CONTINUE
        return base.size

    def _interval(self) -> Interval:
        return self.cs.interval()


class _ZTestMonitor(Monitor):
    """Shared state for the z-test based comparators."""

    def __init__(self, spec: MetricSpec, alpha_rollback: float, alpha_promotion: float) -> None:
        super().__init__(spec, alpha_rollback, alpha_promotion)
        self.diffs = RunningMoments()
        self._last_z: float | None = None

    def _ztest_verdict(self) -> Verdict:
        test = paired_z_test_from_moments(
            self.diffs.n,
            self.diffs.mean,
            self.diffs.variance,
            self.spec.tolerance,
            self.spec.rollback_margin,
        )
        self._last_z = test.z_superiority
        if test.candidate_worse(self.alpha_rollback):
            return Verdict.WORSE
        if test.non_inferior(self.alpha_promotion):
            return Verdict.NOT_WORSE
        return Verdict.CONTINUE

    def _interval(self) -> Interval | None:
        if self.diffs.n < 2:
            return None
        se = math.sqrt(self.diffs.variance / self.diffs.n)
        return Interval(self.diffs.mean - 1.96 * se, self.diffs.mean + 1.96 * se, self.diffs.n)

    def _statistic(self) -> float | None:
        return self._last_z


class FixedHorizonMonitor(_ZTestMonitor):
    """One non-inferiority z-test at a pre-registered number of pairs.

    The horizon is the power-analysis sample size for detecting non-inferiority
    with power ``1 - alpha_rollback`` when the true degradation is zero, using
    the metric's ``planning_sd``.  At the horizon the verdict is forced to
    ``NOT_WORSE`` (non-inferiority shown) or ``WORSE`` (it was not).
    """

    method_name = "fixed_horizon"

    def __init__(self, spec: MetricSpec, alpha_rollback: float, alpha_promotion: float) -> None:
        super().__init__(spec, alpha_rollback, alpha_promotion)
        if spec.planning_sd is None:
            raise ConfigError(f"metric {spec.name!r} needs planning_sd for a fixed-horizon test")
        self.horizon = non_inferiority_sample_size(
            spec.planning_sd, spec.tolerance, alpha_promotion, 1.0 - alpha_rollback
        )

    def _consume(self, base: NDArray[np.float64], cand: NDArray[np.float64]) -> int:
        if self._verdict is not Verdict.CONTINUE:
            return 0
        take = min(base.size, self.horizon - self.diffs.n)
        self.diffs.update_many(self._degradation(base[:take], cand[:take]))
        if self.diffs.n >= self.horizon:
            test = paired_z_test_from_moments(
                self.diffs.n, self.diffs.mean, self.diffs.variance, self.spec.tolerance
            )
            self._last_z = test.z_non_inferiority
            self._verdict = (
                Verdict.NOT_WORSE if test.non_inferior(self.alpha_promotion) else Verdict.WORSE
            )
        return take


class PeekingMonitor(_ZTestMonitor):
    """Fixed-sample z-tests re-run at every check point (the "peeking" anti-pattern)."""

    method_name = "peeking"

    def _consume(self, base: NDArray[np.float64], cand: NDArray[np.float64]) -> int:
        self.diffs.update_many(self._degradation(base, cand))
        return base.size

    def check(self) -> Verdict:
        if self._verdict is not Verdict.WORSE and self.diffs.n >= self.spec.min_pairs:
            self._verdict = self._ztest_verdict()
        return self._verdict


def build_monitor(
    spec: MetricSpec, strategy: Strategy, alpha_rollback: float, alpha_promotion: float
) -> Monitor:
    """Instantiate the monitor for ``spec`` under the chosen strategy."""
    if strategy is Strategy.FIXED_HORIZON:
        return FixedHorizonMonitor(spec, alpha_rollback, alpha_promotion)
    if strategy is Strategy.PEEKING:
        return PeekingMonitor(spec, alpha_rollback, alpha_promotion)
    if spec.method is Method.SPRT:
        return SPRTMonitor(spec, alpha_rollback, alpha_promotion)
    return ConfidenceSequenceMonitor(spec, alpha_rollback, alpha_promotion)
