"""Vectorised traffic simulator that drives a real :class:`RolloutController`.

The simulator generates requests in batches, routes them according to the
controller's *current* stage (shadow or canary weight), draws per-request
outcomes from the scenario's ground truth and feeds them through exactly the
same ingestion API the HTTP gateway uses.  Labels are assumed to arrive
immediately (feedback delay is a gateway concern, see ``serving.feedback``).

What is observable follows production reality:

* shadow: both models score every request, so error and latency arrive as
  *pairs*; the business outcome only exists for the model whose answer the
  user saw (the baseline), so the business metric gets no data in shadow;
* canary: each request is served by one arm; all three metrics arrive per arm
  and are paired FIFO inside the controller.  Candidate inference failures fall
  back to the baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from modelcanary.audit import AuditLog
from modelcanary.engine.controller import Action, RolloutController
from modelcanary.engine.monitors import Strategy
from modelcanary.engine.pairing import Arm
from modelcanary.engine.policy import MetricSource, RolloutPolicy
from modelcanary.sim.scenarios import ArmProfile, Scenario, Truth, derive_truth

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RunResult:
    scenario: str
    truth: Truth
    strategy: Strategy
    seed: int
    action: Action
    reason: str
    inconclusive: bool
    requests: int
    candidate_requests: int
    stage: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["truth"] = self.truth.value
        d["strategy"] = self.strategy.value
        d["action"] = self.action.value
        return d


class TrafficGenerator:
    """Draws per-request outcomes for one scenario from a seeded generator."""

    def __init__(self, scenario: Scenario, rng: np.random.Generator) -> None:
        self.scenario = scenario
        self.rng = rng

    def errors(self, profile: ArmProfile, n: int) -> FloatArray:
        return (self.rng.random(n) < profile.error_rate).astype(np.float64)

    def latencies(self, profile: ArmProfile, n: int) -> FloatArray:
        z = self.rng.standard_normal(n)
        return profile.latency_median_ms * np.exp(profile.latency_sigma * z)

    def conversions(self, profile: ArmProfile, n: int) -> FloatArray:
        return (self.rng.random(n) < profile.conversion_rate).astype(np.float64)

    def paired_errors(self, n: int) -> tuple[FloatArray, FloatArray]:
        """Correlated error indicators for the same requests (shared difficulty)."""
        s = self.scenario
        u = self.rng.random(n)
        shared = self.rng.random(n) < s.error_correlation
        u_cand = np.where(shared, u, self.rng.random(n))
        base = (u < s.baseline.error_rate).astype(np.float64)
        cand = (u_cand < s.candidate.error_rate).astype(np.float64)
        return base, cand

    def paired_latencies(self, n: int) -> tuple[FloatArray, FloatArray]:
        """Lognormal latencies sharing a per-request factor (request size)."""
        s = self.scenario
        rho = s.latency_correlation
        z0 = self.rng.standard_normal(n)
        zb = np.sqrt(rho) * z0 + np.sqrt(1 - rho) * self.rng.standard_normal(n)
        zc = np.sqrt(rho) * z0 + np.sqrt(1 - rho) * self.rng.standard_normal(n)
        base = s.baseline.latency_median_ms * np.exp(s.baseline.latency_sigma * zb)
        cand = s.candidate.latency_median_ms * np.exp(s.candidate.latency_sigma * zc)
        return base, cand


def _feed_arm(
    ctrl: RolloutController, gen: TrafficGenerator, arm: Arm, profile: ArmProfile, n: int
) -> None:
    if n <= 0:
        return
    ctrl.observe_arm(MetricSource.ERROR, arm, gen.errors(profile, n))
    ctrl.observe_arm(MetricSource.LATENCY, arm, gen.latencies(profile, n))
    ctrl.observe_arm(MetricSource.BUSINESS, arm, gen.conversions(profile, n))


def step(ctrl: RolloutController, gen: TrafficGenerator, batch_size: int) -> None:
    """Simulate one batch of requests under the controller's current stage."""
    s = gen.scenario
    rng = gen.rng
    if ctrl.shadow_active:
        failures = int(rng.binomial(batch_size, s.candidate.failure_rate))
        ok = batch_size - failures
        e_b, e_c = gen.paired_errors(ok)
        l_b, l_c = gen.paired_latencies(ok)
        ctrl.observe_paired(MetricSource.ERROR, e_b, e_c)
        ctrl.observe_paired(MetricSource.LATENCY, l_b, l_c)
        ctrl.record_candidate_attempts(batch_size, failures)
        ctrl.record_traffic(batch_size, 0, shadow_scored=ok)
        return
    routed = int(rng.binomial(batch_size, ctrl.candidate_weight))
    failures = int(rng.binomial(routed, s.candidate.failure_rate))
    served_cand = routed - failures
    served_base = batch_size - routed + failures  # failed candidate calls fall back
    _feed_arm(ctrl, gen, Arm.BASELINE, s.baseline, served_base)
    _feed_arm(ctrl, gen, Arm.CANDIDATE, s.candidate, served_cand)
    ctrl.record_candidate_attempts(routed, failures)
    ctrl.record_traffic(served_base, served_cand)


def simulate_rollout(
    scenario: Scenario,
    policy: RolloutPolicy,
    seed: int,
    strategy: Strategy = Strategy.SEQUENTIAL,
    *,
    batch_size: int = 500,
    audit: AuditLog | None = None,
) -> RunResult:
    """Run one rollout to a decision and score it against the ground truth."""
    rng = np.random.default_rng(seed)
    gen = TrafficGenerator(scenario, rng)
    ctrl = RolloutController(
        policy,
        "baseline",
        f"candidate-{scenario.name}",
        strategy=strategy,
        audit=audit,
        rollout_id=f"sim-{scenario.name}-{strategy.value}-{seed}",
        clock=lambda: 0.0,
    )
    decision = None
    while decision is None:
        step(ctrl, gen, batch_size)
        decision = ctrl.evaluate()
    return RunResult(
        scenario=scenario.name,
        truth=derive_truth(scenario, policy),
        strategy=strategy,
        seed=seed,
        action=decision.action,
        reason=decision.reason,
        inconclusive=decision.inconclusive,
        requests=decision.requests,
        candidate_requests=decision.candidate_requests,
        stage=decision.stage,
    )
