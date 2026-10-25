"""SPRT checked against analytic results (Wald identities, exact lattice DP)."""

import itertools
import math

import numpy as np
import pytest

from modelcanary.errors import ConfigError
from modelcanary.stats.sprt import (
    BernoulliSPRT,
    DiscordantPairSPRT,
    SPRTDecision,
    WaldBoundaries,
    discordant_p_independent,
    exact_operating_characteristics,
    odds_ratio_to_discordant_p,
    solve_wald_h,
    wald_asn,
    wald_oc,
)

P0, P1, ALPHA, BETA = 0.5, 0.6, 0.05, 0.1


def test_wald_boundaries_closed_form() -> None:
    b = WaldBoundaries(0.05, 0.1)
    assert b.log_upper == pytest.approx(math.log(0.9 / 0.05))
    assert b.log_lower == pytest.approx(math.log(0.1 / 0.95))


@pytest.mark.parametrize(("alpha", "beta"), [(0.0, 0.1), (0.1, 1.0), (0.6, 0.5)])
def test_invalid_boundaries(alpha: float, beta: float) -> None:
    with pytest.raises(ConfigError):
        WaldBoundaries(alpha, beta)


def test_wald_h_is_plus_one_at_p0_and_minus_one_at_p1() -> None:
    assert solve_wald_h(P0, P0, P1) == pytest.approx(1.0, abs=1e-8)
    assert solve_wald_h(P1, P0, P1) == pytest.approx(-1.0, abs=1e-8)


def test_wald_oc_hits_nominal_rates_at_hypotheses() -> None:
    # Classical identities: L(p0) = 1 - alpha and L(p1) = beta under Wald's approximation.
    assert wald_oc(P0, P0, P1, ALPHA, BETA) == pytest.approx(1 - ALPHA, abs=1e-9)
    assert wald_oc(P1, P0, P1, ALPHA, BETA) == pytest.approx(BETA, abs=1e-9)


def test_wald_oc_is_monotone_decreasing_in_p() -> None:
    ps = np.linspace(0.3, 0.8, 26)
    oc = [wald_oc(float(p), P0, P1, ALPHA, BETA) for p in ps]
    assert all(a > b for a, b in itertools.pairwise(oc))


def test_wald_asn_at_zero_drift_uses_second_moment_formula() -> None:
    a, b = math.log(P1 / P0), math.log((1 - P1) / (1 - P0))
    p_star = -b / (a - b)  # drift p*a + (1-p)*b = 0
    bounds = WaldBoundaries(ALPHA, BETA)
    expected = -bounds.log_upper * bounds.log_lower / (p_star * a * a + (1 - p_star) * b * b)
    assert wald_asn(p_star, P0, P1, ALPHA, BETA) == pytest.approx(expected)
    # Continuity of the ASN curve through the zero-drift point.
    near = wald_asn(p_star + 1e-5, P0, P1, ALPHA, BETA)
    assert near == pytest.approx(expected, rel=1e-3)


def test_exact_error_rates_satisfy_wald_inequalities() -> None:
    h0 = exact_operating_characteristics(P0, P0, P1, ALPHA, BETA, max_n=3000)
    h1 = exact_operating_characteristics(P1, P0, P1, ALPHA, BETA, max_n=3000)
    alpha_true, beta_true = h0.p_accept_h1, h1.p_accept_h0
    assert h0.p_undecided < 1e-9
    assert h1.p_undecided < 1e-9
    # Exact (not approximate) consequences of Wald's argument.
    assert alpha_true <= ALPHA / (1 - BETA)
    assert beta_true <= BETA / (1 - ALPHA)
    assert alpha_true + beta_true <= ALPHA + BETA + 1e-12


@pytest.mark.parametrize("p", [0.45, 0.5, 0.55, 0.6, 0.65])
def test_wald_approximations_close_to_exact_for_small_steps(p: float) -> None:
    exact = exact_operating_characteristics(p, P0, P1, ALPHA, BETA, max_n=4000)
    assert exact.p_accept_h0 == pytest.approx(wald_oc(p, P0, P1, ALPHA, BETA), abs=0.03)
    # Wald's ASN ignores overshoot, so it slightly under-estimates the exact ASN.
    asn = wald_asn(p, P0, P1, ALPHA, BETA)
    assert asn <= exact.expected_n * 1.02
    assert exact.expected_n == pytest.approx(asn, rel=0.15)


def test_online_sprt_matches_exact_dp_by_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    p, reps = 0.55, 4000
    exact = exact_operating_characteristics(p, P0, P1, ALPHA, BETA, max_n=5000)
    accepted_h0 = 0
    total_n = 0
    for _ in range(reps):
        test = BernoulliSPRT(P0, P1, ALPHA, BETA)
        while not test.decided:
            test.update_many(rng.random(64) < p)
        accepted_h0 += test.decision is SPRTDecision.ACCEPT_H0
        total_n += test.n
    se = math.sqrt(exact.p_accept_h0 * (1 - exact.p_accept_h0) / reps)
    assert accepted_h0 / reps == pytest.approx(exact.p_accept_h0, abs=4 * se)
    assert total_n / reps == pytest.approx(exact.expected_n, rel=0.05)


def test_sprt_batch_equals_scalar_and_latches() -> None:
    rng = np.random.default_rng(3)
    xs = rng.random(2000) < 0.62
    scalar = BernoulliSPRT(P0, P1, ALPHA, BETA)
    for x in xs:
        scalar.update(bool(x))
    batch = BernoulliSPRT(P0, P1, ALPHA, BETA)
    for chunk in np.array_split(xs, 37):
        batch.update_many(chunk)
    assert scalar.decision is batch.decision is SPRTDecision.ACCEPT_H1
    assert scalar.n == batch.n
    assert scalar.successes == batch.successes
    assert scalar.llr == pytest.approx(batch.llr)
    frozen = batch.n
    batch.update_many(np.zeros(100, dtype=bool))
    assert batch.n == frozen
    batch.update_many(np.array([], dtype=bool))
    assert batch.decision is SPRTDecision.ACCEPT_H1


def test_invalid_sprt_parameters() -> None:
    with pytest.raises(ConfigError):
        BernoulliSPRT(0.5, 0.5, 0.05, 0.05)
    with pytest.raises(ConfigError):
        BernoulliSPRT(0.0, 0.5, 0.05, 0.05)
    with pytest.raises(ConfigError):
        DiscordantPairSPRT(0.9, 0.05, 0.05)
    with pytest.raises(ConfigError):
        odds_ratio_to_discordant_p(0.0)
    with pytest.raises(ConfigError):
        exact_operating_characteristics(0.5, P0, P1, ALPHA, BETA, max_n=0)


def test_discordant_probability_equals_odds_ratio_identity() -> None:
    pb, pc = 0.10, 0.14
    odds = (pc / (1 - pc)) / (pb / (1 - pb))
    assert discordant_p_independent(pb, pc) == pytest.approx(odds_ratio_to_discordant_p(odds))
    rng = np.random.default_rng(11)
    b = rng.random(400_000) < pb
    c = rng.random(400_000) < pc
    disc = b != c
    assert c[disc].mean() == pytest.approx(discordant_p_independent(pb, pc), abs=0.006)


def test_discordant_sprt_detects_worse_and_accepts_equal() -> None:
    rng = np.random.default_rng(5)
    worse = DiscordantPairSPRT(1.5, 0.05, 0.05)
    while worse.decision is SPRTDecision.CONTINUE:
        worse.update_pairs(rng.random(200) < 0.10, rng.random(200) < 0.16)
    assert worse.decision is SPRTDecision.ACCEPT_H1
    assert worse.pairs > 0
    assert worse.discordant <= worse.pairs
    equal = DiscordantPairSPRT(1.5, 0.05, 0.05)
    while equal.decision is SPRTDecision.CONTINUE:
        equal.update_pairs(rng.random(200) < 0.10, rng.random(200) < 0.10)
    assert equal.decision is SPRTDecision.ACCEPT_H0
    assert 0.0 < equal.baseline_error_rate < 0.2
    assert 0.0 < equal.candidate_error_rate < 0.2


def test_discordant_sprt_stops_counting_at_decision_pair() -> None:
    test = DiscordantPairSPRT(3.0, 0.2, 0.2)
    base = np.zeros(50, dtype=bool)
    cand = np.ones(50, dtype=bool)  # every pair discordant against the candidate
    test.update_pairs(base, cand)
    assert test.decision is SPRTDecision.ACCEPT_H1
    assert test.pairs == test.discordant < 50
    with pytest.raises(ValueError, match="same length"):
        DiscordantPairSPRT(2.0, 0.1, 0.1).update_pairs([1, 0], [1])
    fresh = DiscordantPairSPRT(2.0, 0.1, 0.1)
    fresh.update_pairs([0, 1], [0, 1])  # concordant only
    assert fresh.pairs == 2
    assert fresh.discordant == 0
    assert math.isnan(DiscordantPairSPRT(2.0, 0.1, 0.1).candidate_error_rate)
