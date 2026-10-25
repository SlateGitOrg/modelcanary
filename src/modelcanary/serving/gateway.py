"""Serving core: routes predictions between the production model and a candidate.

:class:`Gateway` holds all serving logic and is independent of HTTP so it can be
unit-tested directly; :mod:`modelcanary.serving.app` exposes it with FastAPI.

Request flow while a rollout is running:

* **shadow stage** -- the production model answers every request; the
  candidate scores the same rows afterwards (a background task, so it adds no
  user-visible latency).  Latencies are recorded as pairs immediately; error
  pairs complete when feedback arrives.
* **canary stage** -- each row is assigned to an arm by a sticky hash of its
  routing key (:mod:`modelcanary.serving.router`).  If the candidate raises,
  its rows are re-served by the production model and counted as failures for
  the guardrail.

After every predict / shadow / feedback event the controller gets a chance to
decide; a promotion swaps the production pointer atomically.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from modelcanary.audit import AuditLog
from modelcanary.engine.controller import Decision, RolloutController, RolloutStatus
from modelcanary.engine.pairing import Arm
from modelcanary.engine.policy import MetricSource, RolloutPolicy
from modelcanary.errors import ConfigError, InferenceError, RolloutStateError
from modelcanary.logging_setup import get_logger
from modelcanary.serving.feedback import FeedbackStore, PendingPrediction, apply_feedback
from modelcanary.serving.registry import ModelRegistry, OnnxModel
from modelcanary.serving.router import assign_arm
from modelcanary.serving.telemetry import Counters

log = get_logger("gateway")


@dataclass(frozen=True, slots=True)
class ServedPrediction:
    prediction_id: str
    label: int
    model_id: str
    arm: Arm


@dataclass(slots=True)
class ShadowJob:
    """Deferred candidate scoring of rows already answered by production."""

    controller: RolloutController
    candidate: OnnxModel
    features: NDArray[np.float32]
    pending: list[PendingPrediction]
    baseline_latency_ms: float


@dataclass(frozen=True, slots=True)
class PredictResult:
    predictions: list[ServedPrediction]
    rollout_id: str | None
    shadow_job: ShadowJob | None = None


@dataclass(slots=True)
class FeedbackSummary:
    received: int = 0
    matched: int = 0
    paired: int = 0
    stale: int = 0
    unmatched: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "received": self.received,
            "matched": self.matched,
            "paired": self.paired,
            "stale": self.stale,
            "unmatched": self.unmatched,
        }


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    prediction_id: str
    label: int
    outcome: float | None = None


def to_features(instances: Sequence[Sequence[float]]) -> NDArray[np.float32]:
    """Validate and convert request rows to a ``float32 [n, d]`` matrix."""
    try:
        arr = np.asarray(instances, dtype=np.float32)
    except ValueError as exc:
        raise ConfigError(f"instances must be a rectangular numeric matrix: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ConfigError(f"instances must be a non-empty 2-D matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ConfigError("instances must be finite")
    return arr


@dataclass
class _State:
    production_id: str
    controller: RolloutController | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class Gateway:
    """Thread-safe serving core shared by the HTTP layer, the CLI and the demo."""

    def __init__(
        self,
        registry: ModelRegistry,
        production_id: str,
        *,
        default_policy: RolloutPolicy,
        audit: AuditLog | None = None,
        feedback: FeedbackStore | None = None,
        counters: Counters | None = None,
    ) -> None:
        registry.get(production_id)  # fail fast on an unknown model
        self.registry = registry
        self.default_policy = default_policy
        self.audit = audit if audit is not None else AuditLog()
        self.feedback_store = feedback if feedback is not None else FeedbackStore()
        self.counters = counters if counters is not None else Counters()
        self._state = _State(production_id)
        self._lock = threading.RLock()
        self._failure_logged: set[str] = set()  # rollouts whose first failure was logged

    # ------------------------------------------------------------------ state

    @property
    def production_id(self) -> str:
        with self._lock:
            return self._state.production_id

    @property
    def controller(self) -> RolloutController | None:
        with self._lock:
            return self._state.controller

    def active_controller(self) -> RolloutController | None:
        ctrl = self.controller
        return ctrl if ctrl is not None and ctrl.running else None

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._state.history)

    def start_rollout(
        self,
        candidate_id: str,
        policy: RolloutPolicy | None = None,
        rollout_id: str | None = None,
    ) -> RolloutController:
        self.registry.get(candidate_id)
        with self._lock:
            if self.active_controller() is not None:
                raise RolloutStateError("a rollout is already running; abort it first")
            if candidate_id == self._state.production_id:
                raise RolloutStateError(f"{candidate_id!r} is already the production model")
            ctrl = RolloutController(
                policy or self.default_policy,
                self._state.production_id,
                candidate_id,
                audit=self.audit,
                rollout_id=rollout_id,
            )
            self._state.controller = ctrl
            return ctrl

    def abort(self, reason: str) -> Decision:
        with self._lock:
            ctrl = self.active_controller()
            if ctrl is None:
                raise RolloutStateError("no rollout is running")
            decision = ctrl.abort(reason)
            self._finish(ctrl, decision)
            return decision

    # ---------------------------------------------------------------- serving

    def predict(
        self, features: NDArray[np.float32], routing_keys: Sequence[str] | None = None
    ) -> PredictResult:
        n = features.shape[0]
        if routing_keys is not None and len(routing_keys) != n:
            raise ConfigError("routing_keys must have one entry per instance")
        with self._lock:
            ctrl = self.active_controller()
            production = self.registry.get(self._state.production_id)
        if ctrl is None:
            pred = production.predict(features)
            self.counters.inc(
                "modelcanary_predictions_total", n, model=production.model_id, arm="production"
            )
            return PredictResult(
                [
                    ServedPrediction(uuid.uuid4().hex, int(lbl), production.model_id, Arm.BASELINE)
                    for lbl in pred.labels
                ],
                None,
            )
        candidate = self.registry.get(ctrl.candidate_id)
        keys = (
            list(routing_keys) if routing_keys is not None else [uuid.uuid4().hex for _ in range(n)]
        )
        if ctrl.shadow_active:
            return self._predict_shadow(ctrl, production, candidate, features)
        return self._predict_canary(ctrl, production, candidate, features, keys)

    def _remember(
        self, ctrl: RolloutController, shadow: bool, arm: Arm, label: int
    ) -> PendingPrediction:
        item = PendingPrediction(
            prediction_id=uuid.uuid4().hex,
            rollout_id=ctrl.rollout_id,
            shadow=shadow,
            served_arm=arm,
            served_label=label,
            created_at=self.feedback_store.now(),
        )
        self.feedback_store.put(item)
        return item

    def _predict_shadow(
        self,
        ctrl: RolloutController,
        production: OnnxModel,
        candidate: OnnxModel,
        features: NDArray[np.float32],
    ) -> PredictResult:
        pred = production.predict(features)
        n = features.shape[0]
        pending = [self._remember(ctrl, True, Arm.BASELINE, int(lbl)) for lbl in pred.labels]
        ctrl.record_traffic(n, 0)
        self.counters.inc(
            "modelcanary_predictions_total", n, model=production.model_id, arm="baseline"
        )
        out = [
            ServedPrediction(p.prediction_id, p.served_label, production.model_id, Arm.BASELINE)
            for p in pending
        ]
        job = ShadowJob(ctrl, candidate, features, pending, pred.latency_ms)
        return PredictResult(out, ctrl.rollout_id, job)

    def run_shadow(self, job: ShadowJob) -> None:
        """Score the candidate on rows production already answered (background task)."""
        ctrl = job.controller
        if not ctrl.running:
            return
        n = job.features.shape[0]
        try:
            pred = job.candidate.predict(job.features)
        except InferenceError as exc:
            self._log_failure(ctrl, exc)
            for p in job.pending:
                p.candidate_failed = True
            ctrl.record_candidate_attempts(n, n)
            self.counters.inc(
                "modelcanary_inference_failures_total", n, model=job.candidate.model_id
            )
        else:
            for p, lbl in zip(job.pending, pred.labels, strict=True):
                p.candidate_label = int(lbl)
            ctrl.record_candidate_attempts(n, 0)
            ctrl.observe_paired(MetricSource.LATENCY, [job.baseline_latency_ms], [pred.latency_ms])
            ctrl.record_traffic(0, 0, shadow_scored=n)
            self.counters.inc("modelcanary_shadow_scored_total", n)
        self._maybe_decide(ctrl)

    def _predict_canary(
        self,
        ctrl: RolloutController,
        production: OnnxModel,
        candidate: OnnxModel,
        features: NDArray[np.float32],
        keys: list[str],
    ) -> PredictResult:
        n = features.shape[0]
        weight = ctrl.candidate_weight
        to_candidate = np.array(
            [assign_arm(ctrl.rollout_id, k, weight) is Arm.CANDIDATE for k in keys], dtype=bool
        )
        labels = np.zeros(n, dtype=np.int64)
        served_by = np.full(n, Arm.BASELINE.value, dtype=object)
        fallback = np.zeros(n, dtype=bool)
        cand_rows = np.flatnonzero(to_candidate)
        if cand_rows.size:
            try:
                cpred = candidate.predict(features[cand_rows])
            except InferenceError as exc:
                self._log_failure(ctrl, exc)
                fallback[cand_rows] = True
                ctrl.record_candidate_attempts(int(cand_rows.size), int(cand_rows.size))
                self.counters.inc(
                    "modelcanary_inference_failures_total", cand_rows.size, model=candidate.model_id
                )
            else:
                labels[cand_rows] = cpred.labels
                served_by[cand_rows] = Arm.CANDIDATE.value
                ctrl.record_candidate_attempts(int(cand_rows.size), 0)
                ctrl.observe_arm(MetricSource.LATENCY, Arm.CANDIDATE, [cpred.latency_ms])
        base_rows = np.flatnonzero(~to_candidate | fallback)
        if base_rows.size:
            bpred = production.predict(features[base_rows])
            labels[base_rows] = bpred.labels
            if not fallback.any():  # a fallback call's latency is not a clean baseline sample
                ctrl.observe_arm(MetricSource.LATENCY, Arm.BASELINE, [bpred.latency_ms])
        out = []
        for i in range(n):
            arm = Arm(served_by[i])
            item = self._remember(ctrl, False, arm, int(labels[i]))
            model_id = candidate.model_id if arm is Arm.CANDIDATE else production.model_id
            out.append(ServedPrediction(item.prediction_id, item.served_label, model_id, arm))
        n_cand = int((served_by == Arm.CANDIDATE.value).sum())
        ctrl.record_traffic(n - n_cand, n_cand)
        if n_cand:
            self.counters.inc(
                "modelcanary_predictions_total", n_cand, model=candidate.model_id, arm="candidate"
            )
        if n - n_cand:
            self.counters.inc(
                "modelcanary_predictions_total",
                n - n_cand,
                model=production.model_id,
                arm="baseline",
            )
        self._maybe_decide(ctrl)
        return PredictResult(out, ctrl.rollout_id)

    def _log_failure(self, ctrl: RolloutController, exc: InferenceError) -> None:
        """Log the first candidate failure of a rollout loudly, later ones at debug level."""
        first_line = str(exc).splitlines()[0]
        if ctrl.rollout_id in self._failure_logged:
            log.debug("candidate_inference_failed rollout=%s error=%s", ctrl.rollout_id, first_line)
            return
        self._failure_logged.add(ctrl.rollout_id)
        log.warning(
            "candidate_inference_failed rollout=%s error=%s (further failures at debug level)",
            ctrl.rollout_id,
            first_line,
        )

    # --------------------------------------------------------------- feedback

    def feedback(self, items: Sequence[FeedbackItem]) -> FeedbackSummary:
        summary = FeedbackSummary(received=len(items))
        touched: dict[str, RolloutController] = {}
        ctrl = self.controller
        for item in items:
            pending = self.feedback_store.pop(item.prediction_id)
            if pending is None or ctrl is None:
                summary.unmatched += 1
                continue
            result = apply_feedback(ctrl, pending, item.label, item.outcome)
            summary.matched += 1
            summary.paired += int(result.paired)
            summary.stale += int(result.stale)
            if not result.stale:
                touched[ctrl.rollout_id] = ctrl
        for name, value in summary.to_dict().items():
            if name != "received" and value:
                self.counters.inc("modelcanary_feedback_total", value, result=name)
        for c in touched.values():
            self._maybe_decide(c)
        return summary

    # -------------------------------------------------------------- decisions

    def _maybe_decide(self, ctrl: RolloutController) -> None:
        decision = ctrl.maybe_evaluate()
        if decision is not None:
            with self._lock:
                self._finish(ctrl, decision)

    def _finish(self, ctrl: RolloutController, decision: Decision) -> None:
        if ctrl.status is RolloutStatus.PROMOTED:
            previous = self._state.production_id
            self._state.production_id = ctrl.candidate_id
            log.info("production_switched from=%s to=%s", previous, ctrl.candidate_id)
        self._state.history.append(ctrl.snapshot())
        self.counters.inc("modelcanary_rollout_decisions_total", action=decision.action.value)

    # ---------------------------------------------------------------- reports

    def status(self) -> dict[str, Any]:
        with self._lock:
            ctrl = self._state.controller
            return {
                "production": self._state.production_id,
                "rollout": ctrl.snapshot() if ctrl is not None else None,
                "pending_feedback": len(self.feedback_store),
                "rollouts_finished": len(self._state.history),
            }

    def gauges(self) -> dict[str, float]:
        ctrl = self.active_controller()
        return {
            "modelcanary_rollout_running": 1.0 if ctrl is not None else 0.0,
            "modelcanary_candidate_weight": ctrl.candidate_weight if ctrl is not None else 0.0,
            "modelcanary_pending_feedback": float(len(self.feedback_store)),
        }
