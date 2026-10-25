"""Analytic operating characteristics of the policy's error-rate SPRT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from modelcanary.engine.policy import Method, RolloutPolicy
from modelcanary.errors import ConfigError
from modelcanary.stats.sprt import (
    exact_operating_characteristics,
    odds_ratio_to_discordant_p,
    wald_asn,
    wald_oc,
)


@dataclass(frozen=True, slots=True)
class OCRow:
    odds_ratio: float
    discordant_p: float
    p_rollback_exact: float
    p_rollback_wald: float
    discordant_pairs_exact: float
    discordant_pairs_wald: float


def sprt_oc_table(
    policy: RolloutPolicy,
    odds_ratios: Sequence[float] = (0.7, 0.85, 1.0, 1.1, 1.25, 1.5, 2.0),
    max_n: int = 20_000,
) -> tuple[float, float, float, list[OCRow]]:
    """Exact (lattice DP) and Wald-approximate OC/ASN of the error-rate SPRT.

    Returns ``(p1, alpha, beta, rows)`` where ``alpha``/``beta`` are the levels
    the monitor actually uses (``beta`` is half the per-metric promotion budget,
    the other half funds the discordance bound).
    """
    specs = [m for m in policy.metrics if m.method is Method.SPRT]
    h1_ratio = specs[0].odds_ratio if specs else None
    if h1_ratio is None:
        raise ConfigError("policy has no SPRT metric")
    p0 = 0.5
    p1 = odds_ratio_to_discordant_p(h1_ratio)
    alpha = policy.metric_alpha_rollback
    beta = policy.metric_alpha_promotion / 2.0
    rows = []
    for ratio in odds_ratios:
        p = odds_ratio_to_discordant_p(ratio)
        exact = exact_operating_characteristics(p, p0, p1, alpha, beta, max_n)
        rows.append(
            OCRow(
                odds_ratio=ratio,
                discordant_p=p,
                p_rollback_exact=exact.p_accept_h1,
                p_rollback_wald=1.0 - wald_oc(p, p0, p1, alpha, beta),
                discordant_pairs_exact=exact.expected_n,
                discordant_pairs_wald=wald_asn(p, p0, p1, alpha, beta),
            )
        )
    return p1, alpha, beta, rows


def render_oc_markdown(p1: float, alpha: float, beta: float, rows: Sequence[OCRow]) -> str:
    lines = [
        f"Error-rate SPRT on discordant pairs: H0 p=0.5 vs H1 p={p1:.4f}, "
        f"alpha={alpha:.4f}, beta={beta:.4f}",
        "",
        "| odds ratio | P(cand wrong / discordant) | P(rollback) exact | P(rollback) Wald "
        "| E[discordant pairs] exact | E[discordant pairs] Wald |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.odds_ratio:.2f} | {r.discordant_p:.4f} | {r.p_rollback_exact:.4f} "
            f"| {r.p_rollback_wald:.4f} | {r.discordant_pairs_exact:,.0f} "
            f"| {r.discordant_pairs_wald:,.0f} |"
        )
    return "\n".join(lines) + "\n"
