"""Confidence sequences checked against analytic identities and Monte-Carlo coverage."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from modelcanary.errors import ConfigError
from modelcanary.stats.confseq import (
    BettingCS,
    Interval,
    MixtureCS,
    gaussian_mixture_radius,
    normal_mixture_martingale,
    optimal_rho,
)


@pytest.mark.parametrize(
    ("s", "t", "rho"), [(0.0, 10.0, 0.3), (5.0, 50.0, 0.2), (-12.0, 400.0, 0.05)]
)
def test_mixture_martingale_closed_form_matches_quadrature(s: float, t: float, rho: float) -> None:
    def integrand(lam: float) -> float:
        prior = math.exp(-lam * lam / (2 * rho * rho)) / (rho * math.sqrt(2 * math.pi))
        return math.exp(lam * s - lam * lam * t / 2) * prior

    numeric, _ = integrate.quad(integrand, -40 * rho, 40 * rho, limit=200)
    assert normal_mixture_martingale(s, t, rho) == pytest.approx(numeric, rel=1e-7)


def test_radius_is_where_mixture_martingale_reaches_one_over_alpha() -> None:
    t, rho, alpha = 300.0, 0.1, 0.05
    r = float(gaussian_mixture_radius(t, 1.0, rho, alpha))
    s = r * t  # standardised sum at the boundary (sigma = 1)
    assert normal_mixture_martingale(s, t, rho) == pytest.approx(1 / alpha, rel=1e-9)


def test_optimal_rho_minimises_radius_at_t_star() -> None:
    t_star, alpha = 2000.0, 0.05
    rhos = np.geomspace(1e-3, 1.0, 4000)
    radii = gaussian_mixture_radius(t_star, 1.0, rhos, alpha)  # type: ignore[arg-type]
    best = float(np.min(radii))
    assert (
        float(gaussian_mixture_radius(t_star, 1.0, optimal_rho(t_star, alpha), alpha))
        <= best * 1.01
    )


def test_radius_is_vectorised_and_shrinks_like_sqrt_log_t_over_t() -> None:
    ts = np.array([100.0, 1000.0, 10000.0, 100000.0])
    r = gaussian_mixture_radius(ts, 1.0, 0.05, 0.05)
    assert isinstance(r, np.ndarray)
    assert np.all(np.diff(r) < 0)
    # Law-of-iterated-logarithm rate: r * sqrt(t / log t) stays bounded.
    scaled = r * np.sqrt(ts / np.log(ts))
    assert scaled.max() / scaled.min() < 3


def _gaussian_paths(rng: np.random.Generator, reps: int, steps: int) -> np.ndarray:
    return rng.normal(0.3, 2.0, size=(reps, steps))


def test_known_sigma_mixture_cs_time_uniform_coverage() -> None:
    rng = np.random.default_rng(0)
    alpha, reps, steps = 0.1, 2000, 2000
    x = _gaussian_paths(rng, reps, steps)
    t = np.arange(1, steps + 1)
    means = np.cumsum(x, axis=1) / t
    radius = gaussian_mixture_radius(t.astype(float), 2.0, optimal_rho(500, alpha), alpha)
    miss = (np.abs(means - 0.3) > radius).any(axis=1).mean()
    assert miss <= alpha  # Ville: P(ever miss) <= alpha
    # ... and a fixed-n 1-alpha CI checked at every step misses far more often.
    naive = 1.645 * 2.0 / np.sqrt(t)
    assert (np.abs(means - 0.3) > naive).any(axis=1).mean() > 3 * alpha


def test_asymptotic_cs_coverage_with_skewed_data() -> None:
    rng = np.random.default_rng(1)
    alpha, reps, steps = 0.1, 400, 3000
    misses = 0
    true_mean = math.exp(0.5)  # lognormal(0, 1)
    for _ in range(reps):
        cs = MixtureCS(alpha, t_star=500)
        missed = False
        for chunk in np.array_split(rng.lognormal(0, 1, steps), 30):
            if not cs.update_many(chunk).contains(true_mean):
                missed = True
                break
        misses += missed
    assert misses / reps <= alpha + 0.03


def test_mixture_cs_intersection_and_edge_cases() -> None:
    cs = MixtureCS(0.05, t_star=100)
    assert cs.interval() == Interval(-math.inf, math.inf, 0)
    cs.update_many([1.0])  # one point: variance undefined -> unbounded
    assert math.isinf(cs.current().lower)
    rng = np.random.default_rng(2)
    widths = []
    for _ in range(20):
        widths.append(cs.update_many(rng.normal(size=50)).width)
    assert all(a >= b for a, b in itertools.pairwise(widths))
    assert cs.n == 1001
    assert abs(cs.mean) < 0.2
    plain = MixtureCS(0.05, sigma=1.0, intersect=False)
    plain.update_many(rng.normal(size=10))
    assert plain.interval() == plain.current()
    with pytest.raises(ConfigError):
        MixtureCS(0.05, sigma=0.0)
    with pytest.raises(ConfigError):
        optimal_rho(0, 0.05)
    with pytest.raises(ConfigError):
        MixtureCS(1.5)


@pytest.mark.parametrize("p", [0.05, 0.5, 0.9])
def test_betting_cs_coverage_bernoulli(p: float) -> None:
    rng = np.random.default_rng(int(p * 100))
    alpha, reps = 0.1, 150
    misses = 0
    for _ in range(reps):
        cs = BettingCS(alpha, grid_size=400)
        missed = False
        for chunk in np.array_split((rng.random(1500) < p).astype(float), 15):
            if not cs.update_many(chunk).contains(p):
                missed = True
                break
        misses += missed
    assert misses / reps <= alpha + 0.04


def test_betting_capital_is_a_martingale_at_the_true_mean() -> None:
    # E[(K+ + K-)/2] = 1 at the true mean; check by Monte Carlo at a grid point.
    rng = np.random.default_rng(9)
    grid_size = 100
    true_m = (40 + 0.5) / grid_size
    values = []
    for _ in range(3000):
        cs = BettingCS(0.05, grid_size=grid_size, intersect=False)
        cs.update_many(rng.beta(true_m * 4, (1 - true_m) * 4, size=20))
        values.append(math.exp(cs.log_capital()[40]))
    arr = np.asarray(values)
    assert arr.mean() == pytest.approx(1.0, abs=4 * arr.std() / math.sqrt(arr.size))


def test_betting_cs_respects_range_and_shrinks() -> None:
    rng = np.random.default_rng(4)
    cs = BettingCS(0.05, lo=-1.0, hi=1.0, grid_size=500)
    assert cs.interval().lower == -1.0
    assert cs.interval().upper == 1.0
    assert math.isnan(cs.mean)
    first = cs.update_many(rng.uniform(-0.5, 0.7, 100))
    later = cs.update_many(rng.uniform(-0.5, 0.7, 4000))
    assert later.width < first.width < 2.0
    assert later.contains(0.1)
    assert cs.update_many([]) == later
    with pytest.raises(ValueError, match="must lie in"):
        cs.update_many([2.0])
    for kwargs in ({"lo": 1.0, "hi": 0.0}, {"grid_size": 5}, {"c": 1.0}):
        with pytest.raises(ConfigError):
            BettingCS(0.05, **kwargs)


def test_betting_cs_degenerate_data_collapses_without_crashing() -> None:
    cs = BettingCS(0.2, grid_size=50, intersect=False)
    for _ in range(40):
        iv = cs.update_many(np.ones(200))
    assert iv.upper == pytest.approx(1.0)
    assert iv.lower > 0.9


@pytest.mark.parametrize("kind", ["mixture", "betting"])
def test_disjoint_interval_restarts_intersection(kind: str) -> None:
    rng = np.random.default_rng(12)
    cs: MixtureCS | BettingCS = (
        MixtureCS(0.05, t_star=200) if kind == "mixture" else BettingCS(0.05, grid_size=200)
    )
    cs.update_many(np.clip(rng.normal(0.2, 0.05, 2000), 0, 1))
    assert cs.interval().upper < 0.5
    for _ in range(40):  # the stream's mean jumps to 0.8
        cs.update_many(np.clip(rng.normal(0.8, 0.05, 500), 0, 1))
    assert cs.resets >= 1
    assert cs.interval().lower > 0.3
