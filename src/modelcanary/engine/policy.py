"""Declarative rollout policy: metrics, traffic stages and guardrails.

Policies are plain data (pydantic models) loaded from TOML, so a rollout can be
reviewed and versioned like code.  Validation happens once at load time; the
controller trusts a :class:`RolloutPolicy` instance.
"""

from __future__ import annotations

import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from modelcanary.errors import ConfigError


class MetricSource(StrEnum):
    """Where a metric's per-request observations come from."""

    ERROR = "error"  # 1 if the prediction disagreed with the delayed label
    LATENCY = "latency"  # inference latency in milliseconds
    BUSINESS = "business"  # a business outcome reported with the feedback (e.g. conversion)


class Direction(StrEnum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class Method(StrEnum):
    """Sequential procedure used to monitor a metric."""

    SPRT = "sprt"  # discordant-pair SPRT, binary metrics only
    ASYMPTOTIC_CS = "asymptotic_cs"  # normal-mixture CS with plug-in variance
    BETTING_CS = "betting_cs"  # hedged-capital CS, bounded metrics only


class OnInconclusive(StrEnum):
    ROLLBACK = "rollback"
    PROMOTE = "promote"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetricSpec(_Frozen):
    """One guarded metric.

    ``tolerance`` is the non-inferiority margin in metric units: the candidate is
    "not worse" when its degradation is confidently below it.  ``rollback_margin``
    (default 0, must not exceed ``tolerance``) is the degradation the candidate
    must confidently exceed to be rolled back: 0 means "any confident
    regression", ``tolerance`` turns the metric into a pure non-inferiority
    test.  Degradation is ``candidate - baseline`` for lower-is-better metrics
    and the reverse otherwise.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    source: MetricSource
    direction: Direction
    method: Method
    tolerance: float = Field(gt=0)
    rollback_margin: float = Field(default=0.0, ge=0)
    odds_ratio: float | None = Field(default=None, gt=1)
    bounds: tuple[float, float] | None = None
    t_star: float = Field(default=2000.0, gt=0)
    min_pairs: int = Field(default=100, ge=2)
    planning_sd: float | None = Field(default=None, gt=0)
    grid_size: int = Field(default=500, ge=10)

    @model_validator(mode="after")
    def _check_method(self) -> Self:
        if self.method is Method.SPRT:
            if self.source is not MetricSource.ERROR:
                raise ValueError("method 'sprt' is only defined for the binary 'error' source")
            if self.odds_ratio is None:
                raise ValueError("method 'sprt' requires odds_ratio")
            if self.direction is not Direction.LOWER_IS_BETTER:
                raise ValueError("error rate must be lower_is_better")
            if self.rollback_margin != 0.0:
                raise ValueError("method 'sprt' encodes its rollback point in odds_ratio")
        if self.rollback_margin > self.tolerance:
            raise ValueError("rollback_margin must not exceed tolerance")
        if self.method is Method.BETTING_CS:
            if self.bounds is None:
                raise ValueError("method 'betting_cs' requires bounds = [lo, hi]")
            lo, hi = self.bounds
            if not hi > lo:
                raise ValueError("bounds must satisfy hi > lo")
        return self


class StageSpec(_Frozen):
    """A traffic stage.  ``weight`` is the candidate's share of *served* traffic."""

    name: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    shadow: bool = False
    min_requests: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.shadow and self.weight != 0.0:
            raise ValueError("a shadow stage serves no candidate traffic (weight must be 0)")
        if not self.shadow and self.weight == 0.0:
            raise ValueError("a non-shadow stage needs weight > 0")
        return self


class Guardrails(_Frozen):
    """Hard limits that act regardless of the statistical verdicts."""

    max_candidate_failure_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    min_failure_attempts: int = Field(default=50, ge=1)
    max_requests: int = Field(default=200_000, ge=1)
    min_requests_before_promote: int = Field(default=0, ge=0)
    min_candidate_requests_before_promote: int = Field(default=0, ge=0)
    on_inconclusive: OnInconclusive = OnInconclusive.ROLLBACK


class RolloutPolicy(_Frozen):
    """Everything the controller needs to run one progressive rollout."""

    metrics: tuple[MetricSpec, ...] = Field(min_length=1)
    stages: tuple[StageSpec, ...] = Field(min_length=1)
    guardrails: Guardrails = Guardrails()
    false_rollback_budget: float = Field(default=0.05, gt=0, lt=1)
    false_promotion_budget: float = Field(default=0.05, gt=0, lt=1)
    promote_min_stage: int = Field(default=0, ge=0)
    check_every: int = Field(default=50, ge=1)
    pair_buffer_capacity: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        sources = [m.source for m in self.metrics]
        if len(set(sources)) != len(sources):
            raise ValueError("at most one metric per source")
        if self.promote_min_stage >= len(self.stages):
            raise ValueError("promote_min_stage must index an existing stage")
        if self.false_rollback_budget + self.false_promotion_budget >= 1:
            raise ValueError("error budgets must sum to < 1")
        return self

    @property
    def metric_alpha_rollback(self) -> float:
        """Per-metric false-rollback level (Bonferroni split of the budget)."""
        return self.false_rollback_budget / len(self.metrics)

    @property
    def metric_alpha_promotion(self) -> float:
        """Per-metric false-promotion level (Bonferroni split of the budget)."""
        return self.false_promotion_budget / len(self.metrics)

    def with_metric_updates(self, updates: dict[str, dict[str, Any]]) -> RolloutPolicy:
        """Return a copy with per-metric fields replaced (used by tuning)."""
        unknown = set(updates) - {m.name for m in self.metrics}
        if unknown:
            raise ConfigError(f"unknown metrics: {sorted(unknown)}")
        metrics = tuple(
            MetricSpec.model_validate({**m.model_dump(), **updates.get(m.name, {})})
            for m in self.metrics
        )
        return self.model_copy(update={"metrics": metrics})

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RolloutPolicy:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"invalid rollout policy: {exc}") from exc

    @classmethod
    def from_toml(cls, path: str | Path) -> RolloutPolicy:
        try:
            raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read policy {path}: {exc}") from exc
        return cls.from_mapping(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> RolloutPolicy:
        """Load a policy from ``.toml`` or ``.json`` (chosen by extension)."""
        p = Path(path)
        if p.suffix.lower() == ".json":
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"cannot read policy {path}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"policy {path} must be a JSON object")
            return cls.from_mapping(raw)
        return cls.from_toml(p)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"
