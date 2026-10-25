"""Many-seed evaluation: error rates and cost of each decision strategy.

For every (scenario, strategy, seed) the simulator runs a full rollout and the
decision is scored against the scenario's ground truth:

* **false promotion** -- promoted although the truth is ``WORSE``;
* **false rollback**  -- rolled back although the truth is ``BETTER`` or ``EQUAL``
  (budget-exhausted "inconclusive" rollbacks are counted here too and also
  reported separately);
* **requests to decision** and **candidate exposure** (requests served by the
  candidate) measure cost and blast radius.

Rates carry Wilson 95% intervals.  Seeds are partitioned: tuning uses seeds in
``TUNING_SEED_RANGE`` and the evaluation refuses to run on any of them.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from modelcanary.engine.controller import Action
from modelcanary.engine.monitors import Strategy
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.sim.runner import RunResult, simulate_rollout
from modelcanary.sim.scenarios import Scenario, Truth

TUNING_SEED_RANGE = range(0, 100_000)
EVAL_SEED_START = 1_000_000


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def eval_seeds(count: int, start: int = EVAL_SEED_START) -> list[int]:
    seeds = list(range(start, start + count))
    if any(s in TUNING_SEED_RANGE for s in seeds):
        raise ConfigError("evaluation seeds overlap the tuning range")
    return seeds


@dataclass(frozen=True, slots=True)
class _Job:
    scenario: Scenario
    policy: RolloutPolicy
    strategy: Strategy
    seeds: tuple[int, ...]
    batch_size: int


def _run_job(job: _Job) -> list[RunResult]:
    return [
        simulate_rollout(job.scenario, job.policy, seed, job.strategy, batch_size=job.batch_size)
        for seed in job.seeds
    ]


def run_grid(
    policy: RolloutPolicy,
    scenarios: Sequence[Scenario],
    strategies: Sequence[Strategy],
    seeds: Sequence[int],
    *,
    workers: int = 1,
    batch_size: int = 500,
    chunk: int = 25,
) -> list[RunResult]:
    """Run every (scenario, strategy, seed) combination, optionally in parallel."""
    seed_chunks = [tuple(seeds[i : i + chunk]) for i in range(0, len(seeds), chunk)]
    jobs = [
        _Job(sc, policy, st, ch, batch_size)
        for sc, st, ch in product(scenarios, strategies, seed_chunks)
    ]
    results: list[RunResult] = []
    if workers <= 1:
        for job in jobs:
            results.extend(_run_job(job))
    else:
        workers = min(workers, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(_run_job, jobs):
                results.extend(batch)
    return results


@dataclass(frozen=True, slots=True)
class Rate:
    count: int
    n: int

    @property
    def value(self) -> float:
        return self.count / self.n if self.n else math.nan

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.count, self.n)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci
        return {"count": self.count, "n": self.n, "rate": self.value, "ci95": [lo, hi]}

    def fmt(self) -> str:
        if not self.n:
            return "n/a"
        lo, hi = self.ci
        return f"{self.value:.3f} [{lo:.3f}, {hi:.3f}]"


@dataclass(frozen=True, slots=True)
class StrategySummary:
    strategy: Strategy
    false_promotion: Rate
    false_rollback: Rate
    inconclusive: Rate
    mean_requests_good: float  # truth better/equal
    mean_requests_worse: float
    mean_candidate_exposure_worse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "false_promotion": self.false_promotion.to_dict(),
            "false_rollback": self.false_rollback.to_dict(),
            "inconclusive": self.inconclusive.to_dict(),
            "mean_requests_good": self.mean_requests_good,
            "mean_requests_worse": self.mean_requests_worse,
            "mean_candidate_exposure_worse": self.mean_candidate_exposure_worse,
        }


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    scenario: str
    truth: Truth
    strategy: Strategy
    runs: int
    promoted: Rate
    mean_requests: float
    median_requests: float
    p90_requests: float
    mean_candidate_requests: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "truth": self.truth.value,
            "strategy": self.strategy.value,
            "runs": self.runs,
            "promoted": self.promoted.to_dict(),
            "mean_requests": self.mean_requests,
            "median_requests": self.median_requests,
            "p90_requests": self.p90_requests,
            "mean_candidate_requests": self.mean_candidate_requests,
        }


def _mean(xs: Iterable[float]) -> float:
    arr = np.fromiter(xs, dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def summarize_strategies(results: Sequence[RunResult]) -> list[StrategySummary]:
    out = []
    for strategy in Strategy:
        rs = [r for r in results if r.strategy is strategy]
        if not rs:
            continue
        good = [r for r in rs if r.truth in (Truth.BETTER, Truth.EQUAL)]
        worse = [r for r in rs if r.truth is Truth.WORSE]
        out.append(
            StrategySummary(
                strategy=strategy,
                false_promotion=Rate(sum(r.action is Action.PROMOTE for r in worse), len(worse)),
                false_rollback=Rate(sum(r.action is Action.ROLLBACK for r in good), len(good)),
                inconclusive=Rate(sum(r.inconclusive for r in rs), len(rs)),
                mean_requests_good=_mean(r.requests for r in good),
                mean_requests_worse=_mean(r.requests for r in worse),
                mean_candidate_exposure_worse=_mean(r.candidate_requests for r in worse),
            )
        )
    return out


def summarize_scenarios(results: Sequence[RunResult]) -> list[ScenarioSummary]:
    keys = sorted({(r.scenario, r.strategy) for r in results}, key=lambda k: (k[0], k[1].value))
    out = []
    for scenario, strategy in keys:
        rs = [r for r in results if r.scenario == scenario and r.strategy is strategy]
        req = np.array([r.requests for r in rs], dtype=np.float64)
        out.append(
            ScenarioSummary(
                scenario=scenario,
                truth=rs[0].truth,
                strategy=strategy,
                runs=len(rs),
                promoted=Rate(sum(r.action is Action.PROMOTE for r in rs), len(rs)),
                mean_requests=float(req.mean()),
                median_requests=float(np.median(req)),
                p90_requests=float(np.quantile(req, 0.9)),
                mean_candidate_requests=_mean(r.candidate_requests for r in rs),
            )
        )
    return out


def render_markdown(
    strategies: Sequence[StrategySummary], scenarios: Sequence[ScenarioSummary], header: str
) -> str:
    lines = [header, "", "### Error rates by strategy", ""]
    lines.append(
        "| strategy | false-promotion rate [95% CI] | false-rollback rate [95% CI] "
        "| inconclusive | mean requests (good) | mean requests (worse) "
        "| mean candidate exposure (worse) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in strategies:
        lines.append(
            f"| {s.strategy.value} | {s.false_promotion.fmt()} | {s.false_rollback.fmt()} "
            f"| {s.inconclusive.value:.3f} | {s.mean_requests_good:,.0f} "
            f"| {s.mean_requests_worse:,.0f} | {s.mean_candidate_exposure_worse:,.0f} |"
        )
    lines += ["", "### Per scenario", ""]
    lines.append(
        "| scenario | truth | strategy | runs | P(promote) | mean req | median req "
        "| p90 req | mean candidate req |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in scenarios:
        lines.append(
            f"| {r.scenario} | {r.truth.value} | {r.strategy.value} | {r.runs} "
            f"| {r.promoted.value:.3f} | {r.mean_requests:,.0f} | {r.median_requests:,.0f} "
            f"| {r.p90_requests:,.0f} | {r.mean_candidate_requests:,.0f} |"
        )
    return "\n".join(lines) + "\n"
