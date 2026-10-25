"""The rollout state machine.

A :class:`RolloutController` owns one progressive rollout of a candidate model
against the current baseline.  It is transport-agnostic: the FastAPI gateway
and the simulator both push observations into it and call :meth:`evaluate`.

Decision order at every check point (first match wins):

1. **Failure guardrail** -- candidate inference failure rate above the limit
   -> roll back immediately (no statistics needed).
2. **Statistical rollback** -- any metric monitor says ``WORSE``.
3. **Promotion** -- every metric says ``NOT_WORSE``, the current stage is at or
   past ``promote_min_stage`` and the minimum total and candidate-served request
   counts have been reached (so a shadow-cleared model still meets real users
   before it takes all traffic).
4. **Stage advance** -- the current stage has served its ``min_requests``.
5. **Budget** -- ``max_requests`` reached without a verdict -> the policy's
   ``on_inconclusive`` action (rollback by default).

Every state change is written to the hash-chained audit log together with the
statistics that justified it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from numpy.typing import ArrayLike

from modelcanary.audit import AuditLog
from modelcanary.engine.monitors import Monitor, MonitorSnapshot, Strategy, Verdict, build_monitor
from modelcanary.engine.pairing import Arm, PairBuffer
from modelcanary.engine.policy import MetricSource, OnInconclusive, RolloutPolicy, StageSpec
from modelcanary.errors import RolloutStateError
from modelcanary.logging_setup import get_logger

log = get_logger("controller")


class RolloutStatus(StrEnum):
    RUNNING = "running"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    ABORTED = "aborted"


class Action(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class Decision:
    """A terminal decision and the evidence behind it."""

    action: Action
    reason: str
    stage: str
    requests: int
    candidate_requests: int
    inconclusive: bool = False
    snapshots: tuple[MonitorSnapshot, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "stage": self.stage,
            "requests": self.requests,
            "candidate_requests": self.candidate_requests,
            "inconclusive": self.inconclusive,
            "metrics": [s.to_dict() for s in self.snapshots],
        }


class RolloutController:
    """Progressive delivery of ``candidate_id`` against ``baseline_id``.

    Thread-safe: all public methods take an internal lock, so the gateway can
    feed observations from request handlers and background shadow tasks.
    """

    def __init__(
        self,
        policy: RolloutPolicy,
        baseline_id: str,
        candidate_id: str,
        *,
        strategy: Strategy = Strategy.SEQUENTIAL,
        audit: AuditLog | None = None,
        rollout_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy
        self.baseline_id = baseline_id
        self.candidate_id = candidate_id
        self.strategy = strategy
        self.rollout_id = rollout_id or f"ro-{uuid.uuid4().hex[:12]}"
        self._audit = audit
        self._clock = clock
        self._lock = threading.RLock()
        self._monitors: dict[MetricSource, Monitor] = {
            spec.source: build_monitor(
                spec, strategy, policy.metric_alpha_rollback, policy.metric_alpha_promotion
            )
            for spec in policy.metrics
        }
        self._buffers = {
            source: PairBuffer(policy.pair_buffer_capacity) for source in self._monitors
        }
        self._last_verdicts = dict.fromkeys(self._monitors, Verdict.CONTINUE)
        self.status = RolloutStatus.RUNNING
        self.decision: Decision | None = None
        self._stage_index = 0
        self.requests = 0
        self.candidate_requests = 0
        self.shadow_scored = 0
        self._stage_requests = 0
        self.candidate_attempts = 0
        self.candidate_failures = 0
        self._events_since_check = 0
        self.started_at = clock()
        self._record(
            "rollout_started",
            {
                "baseline": baseline_id,
                "candidate": candidate_id,
                "strategy": strategy.value,
                "stage": self.stage.name,
                "policy": policy.model_dump(mode="json"),
            },
        )
        log.info(
            "rollout_started id=%s baseline=%s candidate=%s strategy=%s",
            self.rollout_id,
            baseline_id,
            candidate_id,
            strategy.value,
        )

    # ------------------------------------------------------------------ state

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def stage(self) -> StageSpec:
        return self.policy.stages[self._stage_index]

    @property
    def running(self) -> bool:
        return self.status is RolloutStatus.RUNNING

    @property
    def candidate_weight(self) -> float:
        """Share of served traffic routed to the candidate right now."""
        return self.stage.weight if self.running else 0.0

    @property
    def shadow_active(self) -> bool:
        return self.running and self.stage.shadow

    def has_metric(self, source: MetricSource) -> bool:
        return source in self._monitors

    def verdicts(self) -> dict[str, Verdict]:
        with self._lock:
            return {m.spec.name: m.verdict for m in self._monitors.values()}

    def snapshots(self) -> tuple[MonitorSnapshot, ...]:
        with self._lock:
            return tuple(m.snapshot() for m in self._monitors.values())

    # -------------------------------------------------------------- ingestion

    def record_traffic(
        self, baseline_served: int, candidate_served: int, shadow_scored: int = 0
    ) -> None:
        """Count served requests (stage progress and the max-requests budget)."""
        with self._lock:
            if not self.running:
                return
            served = baseline_served + candidate_served
            self.requests += served
            self._stage_requests += served
            self.candidate_requests += candidate_served
            self.shadow_scored += shadow_scored
            self._events_since_check += served

    def record_candidate_attempts(self, attempts: int, failures: int) -> None:
        """Count candidate inference attempts and failures (exceptions / timeouts)."""
        if failures > attempts or failures < 0:
            raise ValueError("failures must be in [0, attempts]")
        with self._lock:
            if self.running:
                self.candidate_attempts += attempts
                self.candidate_failures += failures

    def observe_paired(
        self, source: MetricSource, baseline: ArrayLike, candidate: ArrayLike
    ) -> None:
        """Observations where both models scored the same requests (shadow mode)."""
        with self._lock:
            monitor = self._monitors.get(source)
            if monitor is None or not self.running:
                return
            monitor.observe(baseline, candidate)
            self._events_since_check += 1

    def observe_arm(self, source: MetricSource, arm: Arm, values: ArrayLike) -> None:
        """Observations from one arm only (canary mode); paired FIFO internally."""
        with self._lock:
            monitor = self._monitors.get(source)
            if monitor is None or not self.running:
                return
            base, cand = self._buffers[source].push(arm, values)
            if base.size:
                monitor.observe(base, cand)
            self._events_since_check += 1

    def maybe_evaluate(self) -> Decision | None:
        """Evaluate only if ``check_every`` events accumulated since the last check."""
        with self._lock:
            if self._events_since_check < self.policy.check_every:
                return None
            return self.evaluate()

    # --------------------------------------------------------------- decisions

    def evaluate(self) -> Decision | None:
        """Run the decision procedure once; return a terminal decision if one was taken."""
        with self._lock:
            if not self.running:
                return None
            self._events_since_check = 0
            g = self.policy.guardrails
            for monitor in self._monitors.values():
                monitor.check()
            self._record_verdict_changes()

            if (
                self.candidate_attempts >= g.min_failure_attempts
                and self.candidate_failures / self.candidate_attempts > g.max_candidate_failure_rate
            ):
                rate = self.candidate_failures / self.candidate_attempts
                return self._finish(
                    Action.ROLLBACK,
                    f"guardrail: candidate failure rate {rate:.3f} > "
                    f"{g.max_candidate_failure_rate:.3f}",
                )

            worse = [m.spec.name for m in self._monitors.values() if m.verdict is Verdict.WORSE]
            if worse:
                return self._finish(
                    Action.ROLLBACK, f"metric worse than baseline: {', '.join(worse)}"
                )

            all_ok = all(m.verdict is Verdict.NOT_WORSE for m in self._monitors.values())
            if (
                all_ok
                and self._stage_index >= self.policy.promote_min_stage
                and self.requests >= g.min_requests_before_promote
                and self.candidate_requests >= g.min_candidate_requests_before_promote
            ):
                return self._finish(Action.PROMOTE, "all metrics non-inferior")

            if (
                self._stage_index + 1 < len(self.policy.stages)
                and self._stage_requests >= self.stage.min_requests
            ):
                self._advance_stage()

            if self.requests >= g.max_requests:
                action = (
                    Action.ROLLBACK
                    if g.on_inconclusive is OnInconclusive.ROLLBACK
                    else Action.PROMOTE
                )
                return self._finish(
                    action, f"inconclusive after {self.requests} requests", inconclusive=True
                )
            return None

    def abort(self, reason: str) -> Decision:
        """Operator abort: stop the rollout and keep the baseline."""
        with self._lock:
            if not self.running:
                raise RolloutStateError(f"rollout {self.rollout_id} is already {self.status.value}")
            decision = self._make_decision(Action.ROLLBACK, f"aborted: {reason}")
            self.status = RolloutStatus.ABORTED
            self.decision = decision
            self._record("aborted", decision.to_dict())
            return decision

    def _advance_stage(self) -> None:
        previous = self.stage.name
        self._stage_index += 1
        self._stage_requests = 0
        self._record(
            "stage_advanced",
            {
                "from": previous,
                "to": self.stage.name,
                "weight": self.stage.weight,
                "shadow": self.stage.shadow,
                "requests": self.requests,
            },
        )
        log.info("stage_advanced id=%s stage=%s", self.rollout_id, self.stage.name)

    def _record_verdict_changes(self) -> None:
        for source, monitor in self._monitors.items():
            if monitor.verdict is not self._last_verdicts[source]:
                self._record(
                    "verdict_changed",
                    {
                        "from": self._last_verdicts[source].value,
                        "requests": self.requests,
                        **monitor.snapshot().to_dict(),
                    },
                )
                self._last_verdicts[source] = monitor.verdict

    def _make_decision(self, action: Action, reason: str, inconclusive: bool = False) -> Decision:
        return Decision(
            action=action,
            reason=reason,
            stage=self.stage.name,
            requests=self.requests,
            candidate_requests=self.candidate_requests,
            inconclusive=inconclusive,
            snapshots=tuple(m.snapshot() for m in self._monitors.values()),
        )

    def _finish(self, action: Action, reason: str, inconclusive: bool = False) -> Decision:
        decision = self._make_decision(action, reason, inconclusive)
        self.status = (
            RolloutStatus.PROMOTED if action is Action.PROMOTE else RolloutStatus.ROLLED_BACK
        )
        self.decision = decision
        self._record("promoted" if action is Action.PROMOTE else "rolled_back", decision.to_dict())
        log.info(
            "rollout_decided id=%s action=%s reason=%r requests=%d",
            self.rollout_id,
            action.value,
            reason,
            self.requests,
        )
        return decision

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._audit is not None:
            self._audit.append(self.rollout_id, event, payload)

    # ---------------------------------------------------------------- reports

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly status used by the HTTP API and the CLI."""
        with self._lock:
            return {
                "rollout_id": self.rollout_id,
                "baseline": self.baseline_id,
                "candidate": self.candidate_id,
                "strategy": self.strategy.value,
                "status": self.status.value,
                "stage": self.stage.name,
                "stage_index": self._stage_index,
                "candidate_weight": self.candidate_weight,
                "shadow": self.shadow_active,
                "requests": self.requests,
                "candidate_requests": self.candidate_requests,
                "shadow_scored": self.shadow_scored,
                "candidate_attempts": self.candidate_attempts,
                "candidate_failures": self.candidate_failures,
                "pairs_dropped": {s.value: b.dropped for s, b in self._buffers.items()},
                "metrics": [m.snapshot().to_dict() for m in self._monitors.values()],
                "decision": self.decision.to_dict() if self.decision else None,
            }
