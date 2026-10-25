"""Always-valid confidence sequences (CS) for a stream mean.

A ``(1 - alpha)`` confidence sequence is a sequence of intervals ``C_t`` with
``P(for all t: mu in C_t) >= 1 - alpha`` -- coverage holds *simultaneously* over
time, so the intervals may be checked after every observation and the
experiment stopped whenever they exclude a value ("anytime-valid").

Implemented:

* :func:`gaussian_mixture_radius` -- Robbins' normal-mixture boundary.  Mixing
  the exponential martingale ``exp(l*S_t - l^2 t / 2)`` over ``l ~ N(0, rho^2)``
  gives the closed form :func:`normal_mixture_martingale`; Ville's inequality
  turns it into the radius
  ``sigma * sqrt(2 (t rho^2 + 1) / (t^2 rho^2) * log(sqrt(t rho^2 + 1) / alpha))``.
  With known ``sigma`` and Gaussian data this CS is exact-valid.
* :class:`MixtureCS` -- the same boundary with ``sigma`` either known or replaced
  by a running estimate: the *asymptotic* CS of Waudby-Smith, Arnold, Wu, Wang
  and Ramdas (2021, "Time-uniform central limit theory and asymptotic confidence
  sequences").  Valid for any finite-variance data in the asymptotic sense.
* :class:`BettingCS` -- the hedged-capital betting CS of Waudby-Smith and Ramdas
  (2024, "Estimating means of bounded random variables by betting", JRSS-B).
  Non-asymptotic: valid at every ``t`` for data in a known bounded range.

Both classes support ``update_many`` so the online gateway and the vectorised
simulator share one implementation, and both optionally keep the running
intersection of past intervals (still a valid CS, never wider).  If a new
interval is disjoint from the running intersection -- an event of probability
at most ``alpha`` for stationary data -- the intersection restarts from the
new interval and ``resets`` is incremented, so a drifting stream cannot freeze
a stale interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modelcanary.errors import ConfigError
from modelcanary.stats.running import RunningMoments


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval for the mean at sample size ``n``."""

    lower: float
    upper: float
    n: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ConfigError(f"alpha must be in (0, 1), got {alpha}")


def optimal_rho(t_star: float, alpha: float) -> float:
    """Mixture scale that (approximately) minimises the radius at ``t = t_star``."""
    _check_alpha(alpha)
    if t_star <= 0:
        raise ConfigError("t_star must be positive")
    two_log = -2.0 * math.log(alpha)
    return math.sqrt((two_log + math.log(two_log + 1.0)) / t_star)


def gaussian_mixture_radius(
    t: float | NDArray[np.float64], sigma: float, rho: float, alpha: float
) -> float | NDArray[np.float64]:
    """Half-width of the two-sided normal-mixture CS for the mean after ``t`` samples."""
    t_arr = np.asarray(t, dtype=np.float64)
    tr2 = t_arr * rho * rho
    radius = sigma * np.sqrt(
        2.0 * (tr2 + 1.0) / (t_arr * t_arr * rho * rho) * np.log(np.sqrt(tr2 + 1.0) / alpha)
    )
    return float(radius) if radius.ndim == 0 else radius


def normal_mixture_martingale(s: float, t: float, rho: float) -> float:
    """Closed form of ``E_{l~N(0, rho^2)} exp(l*s - l^2 t / 2)`` for a standardised sum ``s``."""
    v = 1.0 + t * rho * rho
    return math.exp(rho * rho * s * s / (2.0 * v)) / math.sqrt(v)


class MixtureCS:
    """Normal-mixture CS with known or plug-in variance (asymptotic CS).

    Parameters
    ----------
    alpha:
        Miscoverage level of the whole sequence.
    t_star:
        Sample size at which the boundary is tightest (tunes ``rho``; any value
        keeps validity, it only trades early vs late width).
    sigma:
        Known standard deviation.  ``None`` uses the running sample standard
        deviation, floored at ``min_sigma``.
    intersect:
        Keep the running intersection of all intervals produced so far.
    """

    def __init__(
        self,
        alpha: float,
        t_star: float = 1000.0,
        sigma: float | None = None,
        min_sigma: float = 1e-9,
        intersect: bool = True,
    ) -> None:
        _check_alpha(alpha)
        if sigma is not None and sigma <= 0:
            raise ConfigError("sigma must be positive")
        self.alpha = alpha
        self.t_star = t_star
        self.rho = optimal_rho(t_star, alpha)
        self.sigma = sigma
        self.min_sigma = min_sigma
        self.intersect = intersect
        self.moments = RunningMoments()
        self._lower = -math.inf
        self._upper = math.inf
        self.resets = 0  # times the running intersection became empty and was restarted

    @property
    def n(self) -> int:
        return self.moments.n

    @property
    def mean(self) -> float:
        return self.moments.mean

    def _sigma_hat(self) -> float:
        if self.sigma is not None:
            return self.sigma
        var = self.moments.variance  # unbiased; nan below two observations
        if math.isnan(var):
            return math.inf
        return max(math.sqrt(var), self.min_sigma)

    def current(self) -> Interval:
        """The (non-intersected) interval at the current sample size."""
        n = self.moments.n
        if n == 0:
            return Interval(-math.inf, math.inf, 0)
        sigma = self._sigma_hat()
        if not math.isfinite(sigma):
            return Interval(-math.inf, math.inf, n)
        r = float(gaussian_mixture_radius(float(n), sigma, self.rho, self.alpha))
        return Interval(self.moments.mean - r, self.moments.mean + r, n)

    def update_many(self, xs: ArrayLike) -> Interval:
        arr = np.asarray(xs, dtype=np.float64).ravel()
        if arr.size:
            self.moments.update_many(arr)
        return self._refresh()

    def _refresh(self) -> Interval:
        cur = self.current()
        if not self.intersect:
            return cur
        lower = max(self._lower, cur.lower)
        upper = min(self._upper, cur.upper)
        if lower <= upper:
            self._lower, self._upper = lower, upper
        else:
            # Disjoint from the running intersection: under stationarity this has
            # probability <= alpha, so treat it as evidence of drift and restart the
            # intersection from the current interval instead of freezing a stale one.
            self._lower, self._upper = cur.lower, cur.upper
            self.resets += 1
        return self.interval()

    def interval(self) -> Interval:
        if not self.intersect:
            return self.current()
        return Interval(self._lower, self._upper, self.moments.n)


class BettingCS:
    """Hedged-capital betting CS for the mean of data bounded in ``[lo, hi]``.

    For every candidate mean ``m`` on a grid we run two capital processes
    ``K+_t(m) = prod(1 + l+_i(m) (x_i - m))`` and
    ``K-_t(m) = prod(1 - l-_i(m) (x_i - m))`` with predictable bets
    ``l_i = sqrt(2 log(2/alpha) / (s2_{i-1} * i * log(1 + i)))`` truncated to
    ``c/m`` and ``c/(1-m)``.  The average ``(K+ + K-)/2`` is a non-negative
    martingale under ``mu = m``, so by Ville's inequality
    ``{m : (K+ + K-)/2 < 1/alpha}`` is a ``(1 - alpha)`` CS.

    Cost: O(grid_size) per observation and O(grid_size) memory.
    """

    def __init__(
        self,
        alpha: float,
        lo: float = 0.0,
        hi: float = 1.0,
        grid_size: int = 1000,
        c: float = 0.5,
        intersect: bool = True,
    ) -> None:
        _check_alpha(alpha)
        if not hi > lo:
            raise ConfigError("hi must be greater than lo")
        if grid_size < 10:
            raise ConfigError("grid_size must be >= 10")
        if not 0.0 < c < 1.0:
            raise ConfigError("c must be in (0, 1)")
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.c = c
        self.intersect = intersect
        self.grid = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
        self._half_cell = 0.5 / grid_size
        self._log_k_plus = np.zeros(grid_size)
        self._log_k_minus = np.zeros(grid_size)
        self._n = 0
        self._sum = 0.0  # sum of scaled observations
        self._sq_dev = 0.0  # sum of (x_i - mu_hat_i)^2
        self._raw_sum = 0.0
        self._log_threshold = math.log(1.0 / alpha)
        self._bet_scale = 2.0 * math.log(2.0 / alpha)
        self._lower = 0.0
        self._upper = 1.0
        self.resets = 0

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._raw_sum / self._n if self._n else math.nan

    def update_many(self, xs: ArrayLike) -> Interval:
        raw = np.asarray(xs, dtype=np.float64).ravel()
        if raw.size == 0:
            return self.interval()
        if raw.min() < self.lo or raw.max() > self.hi:
            raise ValueError(f"observations must lie in [{self.lo}, {self.hi}]")
        x = (raw - self.lo) / (self.hi - self.lo)
        k = x.size
        idx = self._n + np.arange(1, k + 1, dtype=np.float64)  # 1-based time index i
        # Predictable variance estimate s2_{i-1} built from observations before i.
        cum_x = self._sum + np.cumsum(x)
        mu_hat = (0.5 + cum_x) / (idx + 1.0)  # mu_hat_i includes x_i
        sq = (x - mu_hat) ** 2
        cum_sq = self._sq_dev + np.cumsum(sq)
        prev_sq = np.concatenate(([self._sq_dev], cum_sq[:-1]))
        s2_prev = (0.25 + prev_sq) / idx  # (1/4 + sum_{j<i}) / ((i-1) + 1)
        lam = np.sqrt(self._bet_scale / (s2_prev * idx * np.log1p(idx)))
        m = self.grid[None, :]
        dev = x[:, None] - m
        lam_plus = np.minimum(lam[:, None], self.c / m)
        lam_minus = np.minimum(lam[:, None], self.c / (1.0 - m))
        self._log_k_plus += np.log1p(lam_plus * dev).sum(axis=0)
        self._log_k_minus += np.log1p(-lam_minus * dev).sum(axis=0)
        self._n += k
        self._sum = float(cum_x[-1])
        self._sq_dev = float(cum_sq[-1])
        self._raw_sum += float(raw.sum())
        return self._refresh()

    def log_capital(self) -> NDArray[np.float64]:
        """log of the hedged capital (K+ + K-)/2 on the grid."""
        return np.logaddexp(self._log_k_plus, self._log_k_minus) - math.log(2.0)

    def _current_scaled(self) -> tuple[float, float]:
        keep = self.log_capital() < self._log_threshold
        if not keep.any():  # every grid point rejected: collapse to the empirical mean
            mean = self._sum / self._n
            return mean, mean
        inside = self.grid[keep]
        return (
            max(0.0, float(inside[0]) - self._half_cell),
            min(1.0, float(inside[-1]) + self._half_cell),
        )

    def _refresh(self) -> Interval:
        lower, upper = self._current_scaled()
        if self.intersect:
            new_lower = max(self._lower, lower)
            new_upper = min(self._upper, upper)
            if new_lower <= new_upper:
                self._lower, self._upper = new_lower, new_upper
            else:  # disjoint: drift evidence, restart (see MixtureCS._refresh)
                self._lower, self._upper = lower, upper
                self.resets += 1
        else:
            self._lower, self._upper = lower, upper
        return self.interval()

    def interval(self) -> Interval:
        scale = self.hi - self.lo
        return Interval(self.lo + self._lower * scale, self.lo + self._upper * scale, self._n)
