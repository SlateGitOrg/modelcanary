"""Rollout controller state machine."""

import numpy as np
import pytest

from modelcanary.audit import AuditLog
from modelcanary.engine.controller import Action, RolloutController, RolloutStatus
from modelcanary.engine.pairing import Arm
from modelcanary.engine.policy import MetricSource
from modelcanary.errors import RolloutStateError
from tests.conftest import small_policy


def _ctrl(**overrides: object) -> tuple[RolloutController, AuditLog]:
    audit = AuditLog(clock=lambda: 0.0)
    ctrl = RolloutController(
        small_policy(**overrides), "v1", "v2", audit=audit, rollout_id="ro-test", clock=lambda: 0.0
    )
    return ctrl, audit


def _shadow_batch(
    ctrl: RolloutController, rng: np.random.Generator, n: int, err_c: float, lat_c: float
) -> None:
    ctrl.observe_paired(MetricSource.ERROR, rng.random(n) < 0.1, rng.random(n) < err_c)
    base = rng.normal(20, 2, n)
    ctrl.observe_paired(MetricSource.LATENCY, base, base + lat_c + rng.normal(0, 0.5, n))
    ctrl.record_candidate_attempts(n, 0)
    ctrl.record_traffic(n, 0, shadow_scored=n)


def _canary_batch(
    ctrl: RolloutController, rng: np.random.Generator, n: int, err_c: float, lat_c: float
) -> None:
    for arm, err, shift in ((Arm.BASELINE, 0.1, 0.0), (Arm.CANDIDATE, err_c, lat_c)):
        ctrl.observe_arm(MetricSource.ERROR, arm, rng.random(n) < err)
        ctrl.observe_arm(MetricSource.LATENCY, arm, rng.normal(20 + shift, 2, n))
    ctrl.record_candidate_attempts(n, 0)
    ctrl.record_traffic(n, n)


def _drive(ctrl: RolloutController, err_c: float, lat_c: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    while ctrl.running:
        if ctrl.shadow_active:
            _shadow_batch(ctrl, rng, 50, err_c, lat_c)
        else:
            _canary_batch(ctrl, rng, 50, err_c, lat_c)
        ctrl.evaluate()


def test_equal_candidate_is_promoted_after_shadow() -> None:
    ctrl, audit = _ctrl(promote_min_stage=1)
    assert ctrl.shadow_active
    assert ctrl.candidate_weight == 0.0
    _drive(ctrl, 0.1, 0.0)
    assert ctrl.status is RolloutStatus.PROMOTED
    assert ctrl.decision is not None
    assert ctrl.decision.action is Action.PROMOTE
    assert ctrl.decision.stage == "canary"
    events = [r.event for r in audit.recent(1000)]
    assert events[0] == "rollout_started"
    assert "stage_advanced" in events
    assert events[-1] == "promoted"
    assert audit.verify().ok
    assert ctrl.candidate_weight == 0.0  # no longer running


def test_worse_error_rate_rolls_back_in_shadow() -> None:
    ctrl, audit = _ctrl(
        stages=[
            {"name": "shadow", "weight": 0.0, "shadow": True, "min_requests": 5000},
            {"name": "canary", "weight": 0.5},
        ]
    )
    _drive(ctrl, 0.35, 0.0)
    assert ctrl.status is RolloutStatus.ROLLED_BACK
    assert ctrl.decision is not None
    assert "error_rate" in ctrl.decision.reason
    assert ctrl.decision.stage == "shadow"
    assert ctrl.candidate_requests == 0  # never exposed users
    payload = audit.recent(1)[0].payload
    assert payload["action"] == "rollback"
    assert {m["metric"] for m in payload["metrics"]} == {"error_rate", "latency_ms"}


def test_worse_latency_rolls_back_in_canary() -> None:
    ctrl, _ = _ctrl(stages=[{"name": "canary", "weight": 0.5}])
    _drive(ctrl, 0.1, 3.0)
    assert ctrl.decision is not None
    assert "latency_ms" in ctrl.decision.reason
    assert ctrl.snapshot()["pairs_dropped"]["latency"] == 0


def test_failure_guardrail_preempts_statistics() -> None:
    ctrl, _ = _ctrl()
    ctrl.record_candidate_attempts(19, 19)
    assert ctrl.evaluate() is None  # below min_failure_attempts
    ctrl.record_candidate_attempts(1, 1)
    decision = ctrl.evaluate()
    assert decision is not None
    assert decision.action is Action.ROLLBACK
    assert decision.reason.startswith("guardrail")
    with pytest.raises(ValueError, match="failures"):
        ctrl.record_candidate_attempts(1, 2)


def test_budget_exhaustion_uses_on_inconclusive() -> None:
    ctrl, _ = _ctrl(
        guardrails={"max_requests": 100, "on_inconclusive": "promote"},
        stages=[{"name": "canary", "weight": 0.5}],
    )
    ctrl.record_traffic(60, 40)
    decision = ctrl.evaluate()
    assert decision is not None
    assert decision.action is Action.PROMOTE
    assert decision.inconclusive


def test_maybe_evaluate_respects_check_every() -> None:
    ctrl, _ = _ctrl(check_every=100, guardrails={"max_requests": 10})
    ctrl.record_traffic(5, 0)
    assert ctrl.maybe_evaluate() is None
    ctrl.record_traffic(95, 0)
    decision = ctrl.maybe_evaluate()
    assert decision is not None
    assert decision.inconclusive


def test_abort_and_closed_rollout_ignores_input() -> None:
    ctrl, audit = _ctrl()
    decision = ctrl.abort("operator request")
    assert ctrl.status is RolloutStatus.ABORTED
    assert decision.reason == "aborted: operator request"
    with pytest.raises(RolloutStateError):
        ctrl.abort("again")
    ctrl.record_traffic(10, 10)
    ctrl.observe_arm(MetricSource.ERROR, Arm.CANDIDATE, [1.0])
    ctrl.observe_paired(MetricSource.ERROR, [1.0], [1.0])
    ctrl.record_candidate_attempts(1, 1)
    assert ctrl.requests == 0
    assert ctrl.candidate_attempts == 0
    assert ctrl.evaluate() is None
    assert audit.recent(1)[0].event == "aborted"


def test_unknown_metric_source_is_ignored() -> None:
    ctrl, _ = _ctrl()
    assert not ctrl.has_metric(MetricSource.BUSINESS)
    ctrl.observe_arm(MetricSource.BUSINESS, Arm.BASELINE, [1.0])
    ctrl.observe_paired(MetricSource.BUSINESS, [1.0], [0.0])
    assert set(ctrl.verdicts()) == {"error_rate", "latency_ms"}


def test_snapshot_shape() -> None:
    ctrl, _ = _ctrl()
    snap = ctrl.snapshot()
    assert snap["rollout_id"] == "ro-test"
    assert snap["status"] == "running"
    assert snap["shadow"] is True
    assert snap["decision"] is None
    assert len(snap["metrics"]) == 2
    assert len(ctrl.snapshots()) == 2
    assert ctrl.stage_index == 0


def test_promotion_waits_for_candidate_exposure() -> None:
    ctrl, _ = _ctrl(
        promote_min_stage=1,
        guardrails={"max_requests": 50_000, "min_candidate_requests_before_promote": 3000},
    )
    _drive(ctrl, 0.1, 0.0, seed=3)
    assert ctrl.decision is not None
    assert ctrl.decision.action is Action.PROMOTE
    assert ctrl.decision.candidate_requests >= 3000
