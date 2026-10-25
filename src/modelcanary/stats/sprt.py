"""Wald's sequential probability ratio test (SPRT) for Bernoulli data.

Two layers:

* :class:`BernoulliSPRT` tests ``H0: p = p0`` against ``H1: p = p1`` one
  observation at a time with Wald's boundaries ``A = (1 - beta) / alpha`` and
  ``B = beta / (1 - alpha)``.
* :class:`DiscordantPairSPRT` turns a *two-arm* error-rate comparison into a
  one-sample Bernoulli SPRT by looking only at discordant pairs (one model wrong,
  the other right).  Under independent arms
  ``P(candidate is the one wrong | discordant) = OR / (1 + OR)`` where ``OR`` is
  the error odds ratio, so the test is an exact sequential test of the odds ratio
  (the sequential analogue of McNemar's test).

The analytic helpers (:func:`wald_oc`, :func:`wald_asn` and
:func:`exact_operating_characteristics`) implement Wald's approximations and an
exact dynamic programme over the lattice of (n, successes); the test-suite checks
the online test against both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from modelcanary.errors import ConfigError


class SPRTDecision(StrEnum):
    CONTINUE = "continue"
    ACCEPT_H0 = "accept_h0"
    ACCEPT_H1 = "accept_h1"


@dataclass(frozen=True, slots=True)
class WaldBoundaries:
    """Log-likelihood-ratio stopping boundaries for nominal error rates.

    ``alpha`` bounds P(accept H1 | H0) and ``beta`` bounds P(accept H0 | H1),
    up to Wald's overshoot approximation; the exact guarantees are
    ``alpha' <= alpha / (1 - beta)`` and ``beta' <= beta / (1 - alpha)``.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ConfigError(f"alpha and beta must be in (0, 1), got {self.alpha}, {self.beta}")
        if self.alpha + self.beta >= 1.0:
            raise ConfigError("alpha + beta must be < 1")

    @property
    def log_upper(self) -> float:
        """log A: cross upwards -> accept H1."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def log_lower(self) -> float:
        """log B: cross downwards -> accept H0."""
        return math.log(self.beta / (1.0 - self.alpha))


def _validate_probability(name: str, p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ConfigError(f"{name} must be in (0, 1), got {p}")


class BernoulliSPRT:
    """Online Wald SPRT for ``H0: p = p0`` vs ``H1: p = p1``.

    The test latches: after the first boundary crossing further observations are
    ignored and :attr:`decision` never changes.
    """

    def __init__(self, p0: float, p1: float, alpha: float, beta: float) -> None:
        _validate_probability("p0", p0)
        _validate_probability("p1", p1)
        if p0 == p1:
            raise ConfigError("p0 and p1 must differ")
        self.p0 = p0
        self.p1 = p1
        self.boundaries = WaldBoundaries(alpha, beta)
        self.step_success = math.log(p1 / p0)
        self.step_failure = math.log((1.0 - p1) / (1.0 - p0))
        self._n = 0
        self._successes = 0
        self._llr = 0.0
        self._decision = SPRTDecision.CONTINUE

    @property
    def n(self) -> int:
        """Observations consumed (stops counting once the test has decided)."""
        return self._n

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def llr(self) -> float:
        """Current log-likelihood ratio log(L1 / L0)."""
        return self._llr

    @property
    def decision(self) -> SPRTDecision:
        return self._decision

    @property
    def decided(self) -> bool:
        return self._decision is not SPRTDecision.CONTINUE

    def update(self, x: bool | int) -> SPRTDecision:
        return self.update_many(np.asarray([bool(x)]))

    def update_many(self, xs: ArrayLike) -> SPRTDecision:
        """Consume observations in order, stopping at the first boundary crossing."""
        if self.decided:
            return self._decision
        arr = np.asarray(xs, dtype=bool).ravel()
        if arr.size == 0:
            return self._decision
        path = self._llr + np.cumsum(np.where(arr, self.step_success, self.step_failure))
        upper = path >= self.boundaries.log_upper
        lower = path <= self.boundaries.log_lower
        crossed = upper | lower
        if crossed.any():
            idx = int(np.argmax(crossed))
            consumed = arr[: idx + 1]
            self._decision = SPRTDecision.ACCEPT_H1 if upper[idx] else SPRTDecision.ACCEPT_H0
            self._llr = float(path[idx])
        else:
            consumed = arr
            self._llr = float(path[-1])
        self._n += int(consumed.size)
        self._successes += int(consumed.sum())
        return self._decision


def odds_ratio_to_discordant_p(odds_ratio: float) -> float:
    """P(candidate wrong | exactly one arm wrong) for a given error odds ratio."""
    if odds_ratio <= 0:
        raise ConfigError("odds ratio must be positive")
    return odds_ratio / (1.0 + odds_ratio)


def discordant_p_independent(p_baseline: float, p_candidate: float) -> float:
    """P(candidate wrong | discordant) when the two arms err independently."""
    num = p_candidate * (1.0 - p_baseline)
    den = num + p_baseline * (1.0 - p_candidate)
    return num / den


class DiscordantPairSPRT:
    """Two-arm error-rate SPRT on discordant pairs (sequential McNemar test).

    ``H0``: error odds ratio candidate/baseline equals 1 (no degradation).
    ``H1``: the odds ratio equals ``odds_ratio`` (> 1, a degradation worth a rollback).
    """

    def __init__(self, odds_ratio: float, alpha: float, beta: float) -> None:
        if odds_ratio <= 1.0:
            raise ConfigError("the H1 odds ratio must be > 1 (candidate worse)")
        self.odds_ratio = odds_ratio
        self.inner = BernoulliSPRT(0.5, odds_ratio_to_discordant_p(odds_ratio), alpha, beta)
        self._pairs = 0
        self._baseline_errors = 0
        self._candidate_errors = 0

    @property
    def pairs(self) -> int:
        return self._pairs

    @property
    def discordant(self) -> int:
        return self.inner.n

    @property
    def decision(self) -> SPRTDecision:
        return self.inner.decision

    @property
    def baseline_error_rate(self) -> float:
        return self._baseline_errors / self._pairs if self._pairs else math.nan

    @property
    def candidate_error_rate(self) -> float:
        return self._candidate_errors / self._pairs if self._pairs else math.nan

    def update_pairs(self, baseline_errors: ArrayLike, candidate_errors: ArrayLike) -> SPRTDecision:
        """Feed aligned pairs of error indicators (1 = the model was wrong)."""
        if self.inner.decided:
            return self.inner.decision
        base = np.asarray(baseline_errors, dtype=bool).ravel()
        cand = np.asarray(candidate_errors, dtype=bool).ravel()
        if base.shape != cand.shape:
            raise ValueError("baseline and candidate arrays must have the same length")
        discordant = base != cand
        if discordant.any():
            # Only count pairs up to (and including) the deciding discordant pair.
            positions = np.flatnonzero(discordant)
            before = self.inner.n
            self.inner.update_many(cand[discordant])
            used = self.inner.n - before
            stop = int(positions[used - 1]) + 1 if self.inner.decided else base.size
        else:
            stop = base.size
        self._pairs += stop
        self._baseline_errors += int(base[:stop].sum())
        self._candidate_errors += int(cand[:stop].sum())
        return self.inner.decision


# --------------------------------------------------------------------------- analytics


def _drift(p: float, step_success: float, step_failure: float) -> float:
    return p * step_success + (1.0 - p) * step_failure


def solve_wald_h(p: float, p0: float, p1: float) -> float:
    """Non-zero root ``h`` of ``p (p1/p0)^h + (1-p) ((1-p1)/(1-p0))^h = 1``.

    Returns 0.0 at the single ``p`` where the drift of the log-likelihood ratio
    is zero (the root is then degenerate).  By construction ``h(p0) = 1`` and
    ``h(p1) = -1``.
    """
    _validate_probability("p", p)
    a = math.log(p1 / p0)
    b = math.log((1.0 - p1) / (1.0 - p0))
    drift = _drift(p, a, b)
    if abs(drift) < 1e-14:
        return 0.0

    def f(h: float) -> float:
        return p * math.exp(h * a) + (1.0 - p) * math.exp(h * b) - 1.0

    # The non-trivial root has the opposite sign to the drift.
    sign = -1.0 if drift > 0 else 1.0
    lo, hi = 1e-9, 1.0
    while f(sign * hi) < 0:
        hi *= 2.0
        if hi > 1e6:  # pragma: no cover - unreachable for valid inputs
            raise ArithmeticError("failed to bracket Wald's h")
    root = brentq(lambda x: f(sign * x), lo, hi, xtol=1e-12)
    return sign * float(root)


def wald_oc(p: float, p0: float, p1: float, alpha: float, beta: float) -> float:
    """Wald's approximation of the operating characteristic P(accept H0 | p)."""
    bounds = WaldBoundaries(alpha, beta)
    log_a, log_b = bounds.log_upper, bounds.log_lower
    h = solve_wald_h(p, p0, p1)
    if h == 0.0:
        return log_a / (log_a - log_b)
    a_h = math.exp(h * log_a)
    b_h = math.exp(h * log_b)
    return (a_h - 1.0) / (a_h - b_h)


def wald_asn(p: float, p0: float, p1: float, alpha: float, beta: float) -> float:
    """Wald's approximation of the expected sample number E[N | p]."""
    bounds = WaldBoundaries(alpha, beta)
    log_a, log_b = bounds.log_upper, bounds.log_lower
    a = math.log(p1 / p0)
    b = math.log((1.0 - p1) / (1.0 - p0))
    drift = _drift(p, a, b)
    if abs(drift) < 1e-14:
        second_moment = p * a * a + (1.0 - p) * b * b
        return -log_a * log_b / second_moment
    oc = wald_oc(p, p0, p1, alpha, beta)
    return (oc * log_b + (1.0 - oc) * log_a) / drift


@dataclass(frozen=True, slots=True)
class ExactOC:
    """Exact operating characteristics of a truncated Bernoulli SPRT."""

    p_accept_h0: float
    p_accept_h1: float
    p_undecided: float
    expected_n: float


def exact_operating_characteristics(
    p: float, p0: float, p1: float, alpha: float, beta: float, max_n: int
) -> ExactOC:
    """Exact stopping probabilities and E[min(N, max_n)] by dynamic programming.

    The log-likelihood ratio after ``n`` observations with ``k`` successes is
    ``k*a + (n-k)*b``, so the continuation region is a lattice and the
    probability mass over ``k`` can be propagated exactly.  Cost: O(max_n^2).
    """
    _validate_probability("p", p)
    if max_n < 1:
        raise ConfigError("max_n must be >= 1")
    bounds = WaldBoundaries(alpha, beta)
    a = math.log(p1 / p0)
    b = math.log((1.0 - p1) / (1.0 - p0))
    mass = np.array([1.0])  # mass[k]: P(still running, k successes) after n steps
    accept_h0 = accept_h1 = 0.0
    expected_n = 0.0
    for n in range(1, max_n + 1):
        expected_n += float(mass.sum())  # P(N >= n)
        new = np.zeros(n + 1)
        new[:-1] += mass * (1.0 - p)
        new[1:] += mass * p
        k = np.arange(n + 1)
        llr = k * a + (n - k) * b
        up = llr >= bounds.log_upper
        down = llr <= bounds.log_lower
        accept_h1 += float(new[up].sum())
        accept_h0 += float(new[down].sum())
        new[up | down] = 0.0
        mass = new
        if mass.sum() < 1e-15:
            break
    return ExactOC(accept_h0, accept_h1, float(mass.sum()), expected_n)
