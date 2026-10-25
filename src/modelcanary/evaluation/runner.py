"""End-to-end evaluation: pilot tuning on tuning seeds, then the held-out grid."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcanary.engine.monitors import Strategy
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.evaluation.harness import (
    StrategySummary,
    eval_seeds,
    render_markdown,
    run_grid,
    summarize_scenarios,
    summarize_strategies,
)
from modelcanary.evaluation.tuning import tune_policy
from modelcanary.logging_setup import get_logger
from modelcanary.sim.scenarios import BASELINE, standard_scenarios

log = get_logger("evaluation")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seeds: int = 300
    pilot_seeds: int = 20
    workers: int = 1
    batch_size: int = 500
    scenarios: tuple[str, ...] = ()  # empty = the full standard suite

    def __post_init__(self) -> None:
        if self.seeds < 1 or self.pilot_seeds < 1:
            raise ConfigError("seeds and pilot_seeds must be >= 1")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    config: EvaluationConfig
    planning_sd: dict[str, float]
    strategies: list[StrategySummary]
    markdown: str
    data: dict[str, Any]
    elapsed_seconds: float

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "eval.md"
        json_path = out / "eval.json"
        md_path.write_text(self.markdown, encoding="utf-8")
        json_path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        return md_path, json_path


def run_evaluation(policy: RolloutPolicy, config: EvaluationConfig) -> EvaluationReport:
    """Tune the fixed-horizon plan on pilot seeds, then score all strategies on held-out seeds."""
    suite = standard_scenarios()
    names = config.scenarios or tuple(suite)
    unknown = [n for n in names if n not in suite]
    if unknown:
        raise ConfigError(f"unknown scenarios: {unknown}; choose from {sorted(suite)}")
    started = time.perf_counter()
    tuned, sds = tune_policy(policy, BASELINE, list(range(config.pilot_seeds)))
    log.info("pilot_planning_sd %s", " ".join(f"{k}={v:.4f}" for k, v in sds.items()))
    seeds = eval_seeds(config.seeds)
    results = run_grid(
        tuned,
        [suite[n] for n in names],
        list(Strategy),
        seeds,
        workers=config.workers,
        batch_size=config.batch_size,
    )
    strategies = summarize_strategies(results)
    scenarios = summarize_scenarios(results)
    elapsed = time.perf_counter() - started
    header = (
        f"## Evaluation: {config.seeds} held-out seeds x {len(names)} scenarios x "
        f"{len(Strategy)} strategies ({len(results)} simulated rollouts)\n\n"
        f"Pilot (A/A, tuning seeds 0..{config.pilot_seeds - 1}) planning sd: "
        + ", ".join(f"`{k}` = {v:.4f}" for k, v in sds.items())
        + f"\n\nEvaluation seeds start at {seeds[0]}; wall time {elapsed:.1f} s."
    )
    data = {
        "config": {
            "seeds": config.seeds,
            "pilot_seeds": config.pilot_seeds,
            "batch_size": config.batch_size,
            "scenarios": list(names),
            "first_eval_seed": seeds[0],
        },
        "planning_sd": sds,
        "policy": tuned.model_dump(mode="json"),
        "strategies": [s.to_dict() for s in strategies],
        "scenarios": [s.to_dict() for s in scenarios],
        "runs": len(results),
    }
    return EvaluationReport(
        config=config,
        planning_sd=sds,
        strategies=strategies,
        markdown=render_markdown(strategies, scenarios, header),
        data=data,
        elapsed_seconds=elapsed,
    )
