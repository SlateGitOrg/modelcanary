"""Fixed-horizon comparators: power analysis and one-shot z-tests on paired differences.

These are the baselines the sequential procedures are evaluated against.  A
fixed-horizon test is only valid when it is evaluated *once*, at the sample size
chosen in advance; :func:`paired_z_test` applied repeatedly is the classic
"peeking" mistake whose inflated error rates the evaluation also measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from modelcanary.errors import ConfigError


def z_quantile(q: float) -> float:
    """Standard normal quantile."""
    if not 0.0 < q < 1.0:
        raise ConfigError(f"quantile must be in (0, 1), got {q}")
    return float(norm.ppf(q))


def non_inferiority_sample_size(
    sigma: float, margin: float, alpha: float, power: float, true_degradation: float = 0.0
) -> int:
    """Pairs needed for a one-sided non-inferiority z-test.

    Tests ``H0: delta >= margin`` against ``H1: delta < margin`` at level
    ``alpha`` with the requested ``power`` when the true degradation is
    ``true_degradation``:  ``n = ((z_{1-alpha} + z_{power}) * sigma / (margin - true))^2``.
    """
    if sigma <= 0:
        raise ConfigError("sigma must be positive")
    gap = margin - true_degradation
    if gap <= 0:
        raise ConfigError("margin must exceed the assumed true degradation")
    z = z_quantile(1.0 - alpha) + z_quantile(power)
    return max(2, math.ceil((z * sigma / gap) ** 2))


@dataclass(frozen=True, slots=True)
class PairedZTest:
    """One-sided z statistics for the mean paired degradation ``delta``."""

    n: int
    mean: float
    standard_error: float
    z_superiority: float  # (mean - worse_margin) / se : large => candidate significantly worse
    z_non_inferiority: float  # (mean - margin) / se : very negative => within margin

    def candidate_worse(self, alpha: float) -> bool:
        """Reject ``H0: delta <= worse_margin`` at level ``alpha``."""
        return self.z_superiority > z_quantile(1.0 - alpha)

    def non_inferior(self, alpha: float) -> bool:
        """Reject ``H0: delta >= margin`` at level ``alpha``."""
        return self.z_non_inferiority < -z_quantile(1.0 - alpha)


def paired_z_test_from_moments(
    n: int, mean: float, variance: float, margin: float, worse_margin: float = 0.0
) -> PairedZTest:
    """z statistics from summary moments; ``worse_margin`` shifts the "worse" null."""
    if n < 2:
        raise ValueError("need at least two pairs")
    se = math.sqrt(max(variance, 1e-300) / n)
    return PairedZTest(n, mean, se, (mean - worse_margin) / se, (mean - margin) / se)


def paired_z_test(degradations: ArrayLike, margin: float) -> PairedZTest:
    """z-test on per-pair degradations ``d_i`` (positive = candidate worse)."""
    arr = np.asarray(degradations, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError("need at least two pairs")
    return paired_z_test_from_moments(arr.size, float(arr.mean()), float(arr.var(ddof=1)), margin)
