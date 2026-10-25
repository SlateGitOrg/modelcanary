"""Metric monitors under every strategy."""

import math

import numpy as np
import pytest

from modelcanary.engine.monitors import (
    ConfidenceSequenceMonitor,
    FixedHorizonMonitor,
    PeekingMonitor,
    SPRTMonitor,
    Strategy,
    Verdict,
    build_monitor,
)
from modelcanary.engine.policy import MetricSpec
from modelcanary.errors import ConfigError

ERR = MetricSpec(
    name="error_rate",
    source="error",
    direction="lower_is_better",
    method="sprt",
    odds_ratio=1.5,
    tolerance=0.03,
    planning_sd=0.42,
)
LAT = MetricSpec(
    name="latency_ms",
    source="latency",
    direction="lower_is_better",
    method="asymptotic_cs",
    tolerance=2.0,
    min_pairs=50,
    planning_sd=10.0,
)
CONV = MetricSpec(
    name="conversion",
    source="business",
    direction="higher_is_better",
    method="betting_cs",
    bounds=(0.0, 1.0),
    tolerance=0.05,
    min_pairs=50,
    grid_size=200,
    planning_sd=0.5,
)


def test_build_monitor_dispatch() -> None:
    assert isinstance(build_monitor(ERR, Strategy.SEQUENTIAL, 0.05, 0.05), SPRTMonitor)
    lat = build_monitor(LAT, Strategy.SEQUENTIAL, 0.05, 0.05)
    assert isinstance(lat, ConfidenceSequenceMonitor)
    conv = build_monitor(CONV, Strategy.SEQUENTIAL, 0.05, 0.05)
    assert conv.method_name == "betting_cs"
    assert isinstance(build_monitor(LAT, Strategy.FIXED_HORIZON, 0.05, 0.05), FixedHorizonMonitor)
    assert isinstance(build_monitor(LAT, Strategy.PEEKING, 0.05, 0.05), PeekingMonitor)


def test_sprt_monitor_flags_worse_and_freezes() -> None:
    rng = np.random.default_rng(0)
    mon = build_monitor(ERR, Strategy.SEQUENTIAL, 0.05, 0.05)
    while mon.verdict is Verdict.CONTINUE:
        mon.observe(rng.random(100) < 0.10, rng.random(100) < 0.25)
    assert mon.verdict is Verdict.WORSE
    pairs = mon.pairs
    mon.observe(np.zeros(10), np.zeros(10))
    assert mon.pairs == pairs  # absorbing
    snap = mon.snapshot()
    assert snap.verdict == "worse"
    assert snap.statistic is not None
    assert snap.statistic > 0
    assert snap.degradation is not None
    assert snap.degradation > 0


def test_sprt_monitor_accepts_equal() -> None:
    rng = np.random.default_rng(1)
    mon = build_monitor(ERR, Strategy.SEQUENTIAL, 0.05, 0.05)
    while mon.verdict is Verdict.CONTINUE:
        mon.observe(rng.random(100) < 0.10, rng.random(100) < 0.10)
    assert mon.verdict is Verdict.NOT_WORSE
    before = mon.pairs
    mon.observe(np.ones(5), np.ones(5))
    assert mon.pairs == before  # SPRT latched: later data not consumed


def test_cs_monitor_direction_and_min_pairs() -> None:
    rng = np.random.default_rng(2)
    mon = build_monitor(CONV, Strategy.SEQUENTIAL, 0.05, 0.05)
    # Candidate converts much less: degradation = baseline - candidate > 0.
    mon.observe(np.ones(10), np.zeros(10))
    assert mon.verdict is Verdict.CONTINUE  # below min_pairs
    while mon.verdict is Verdict.CONTINUE:
        mon.observe(rng.random(100) < 0.5, rng.random(100) < 0.2)
    assert mon.verdict is Verdict.WORSE
    snap = mon.snapshot()
    assert snap.lower is not None
    assert snap.lower > 0
    assert snap.degradation == pytest.approx(0.3, abs=0.15)


def test_cs_monitor_not_worse_is_revisable() -> None:
    mon = build_monitor(LAT, Strategy.SEQUENTIAL, 0.05, 0.05)
    rng = np.random.default_rng(3)
    for _ in range(100):
        base = rng.normal(40, 3, 100)
        mon.observe(base, base + rng.normal(0, 0.5, 100))
        if mon.verdict is Verdict.NOT_WORSE:
            break
    assert mon.verdict is Verdict.NOT_WORSE
    # A sustained regression later still triggers WORSE.
    for _ in range(200):
        base = rng.normal(40, 3, 100)
        mon.observe(base, base + 1.5 + rng.normal(0, 0.5, 100))
        if mon.check() is Verdict.WORSE:
            break
    assert mon.check() is Verdict.WORSE


def test_fixed_horizon_decides_exactly_at_horizon() -> None:
    mon = build_monitor(LAT, Strategy.FIXED_HORIZON, 0.05, 0.05)
    assert isinstance(mon, FixedHorizonMonitor)
    horizon = mon.horizon
    rng = np.random.default_rng(4)
    mon.observe(np.zeros(horizon - 1), rng.normal(0, 10, horizon - 1))
    assert mon.verdict is Verdict.CONTINUE
    assert mon.snapshot().lower is not None
    mon.observe(np.zeros(50), rng.normal(0, 10, 50))
    assert mon.verdict in (Verdict.NOT_WORSE, Verdict.WORSE)
    assert mon.pairs == horizon
    mon.observe(np.zeros(5), np.zeros(5))
    assert mon.pairs == horizon


def test_fixed_horizon_requires_planning_sd() -> None:
    spec = LAT.model_copy(update={"planning_sd": None})
    with pytest.raises(ConfigError, match="planning_sd"):
        build_monitor(spec, Strategy.FIXED_HORIZON, 0.05, 0.05)


def test_peeking_only_decides_on_check() -> None:
    mon = build_monitor(LAT, Strategy.PEEKING, 0.05, 0.05)
    mon.observe(np.zeros(500), np.full(500, 5.0) + np.random.default_rng(5).normal(0, 1, 500))
    assert mon.verdict is Verdict.CONTINUE
    assert mon.check() is Verdict.WORSE
    assert mon.snapshot().statistic is not None


def test_snapshot_before_data_is_json_safe() -> None:
    snap = build_monitor(LAT, Strategy.SEQUENTIAL, 0.05, 0.05).snapshot().to_dict()
    assert snap["pairs"] == 0
    assert snap["degradation"] is None
    assert snap["lower"] is None
    assert all(not isinstance(v, float) or math.isfinite(v) for v in snap.values())


def test_monitor_rejects_misaligned_batches() -> None:
    mon = build_monitor(LAT, Strategy.SEQUENTIAL, 0.05, 0.05)
    with pytest.raises(ValueError, match="equal length"):
        mon.observe([1.0, 2.0], [1.0])


def test_identical_candidate_is_cleared_by_discordance_bound() -> None:
    mon = build_monitor(ERR, Strategy.SEQUENTIAL, 0.05, 0.05)
    assert isinstance(mon, SPRTMonitor)
    errors = (np.random.default_rng(6).random(5000) < 0.1).astype(float)
    for chunk in np.array_split(errors, 50):
        mon.observe(chunk, chunk)  # same model: every pair concordant
        if mon.verdict is Verdict.NOT_WORSE:
            break
    assert mon.verdict is Verdict.NOT_WORSE
    assert mon.test.discordant == 0  # the SPRT itself never moved
    assert mon.discordance.interval().upper < ERR.tolerance


def test_discordance_shortcut_is_revised_by_sprt() -> None:
    spec = ERR.model_copy(update={"tolerance": 0.5, "min_pairs": 10})
    mon = build_monitor(spec, Strategy.SEQUENTIAL, 0.05, 0.05)
    rng = np.random.default_rng(8)
    seen_not_worse = False
    for _ in range(400):
        base = rng.random(50) < 0.05
        mon.observe(base, base | (rng.random(50) < 0.08))  # candidate adds errors
        seen_not_worse |= mon.verdict is Verdict.NOT_WORSE
        if mon.verdict is Verdict.WORSE:
            break
    assert seen_not_worse
    assert mon.check() is Verdict.WORSE


def test_rollback_margin_tolerates_small_regressions() -> None:
    lenient = LAT.model_copy(update={"rollback_margin": 1.0})
    rng = np.random.default_rng(9)
    for strategy in (Strategy.SEQUENTIAL, Strategy.PEEKING):
        strict_mon = build_monitor(LAT, strategy, 0.05, 0.05)
        lenient_mon = build_monitor(lenient, strategy, 0.05, 0.05)
        for _ in range(100):
            base = rng.normal(40, 1, 200)
            cand = base + 0.5 + rng.normal(0, 0.3, 200)  # +0.5 ms: inside the margin
            strict_mon.observe(base, cand)
            lenient_mon.observe(base, cand)
        assert strict_mon.check() is Verdict.WORSE
        assert lenient_mon.check() is Verdict.NOT_WORSE
