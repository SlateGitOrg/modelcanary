import math

import numpy as np
import pytest

from modelcanary.errors import ConfigError
from modelcanary.stats.fixed import (
    non_inferiority_sample_size,
    paired_z_test,
    z_quantile,
)


def test_sample_size_formula() -> None:
    n = non_inferiority_sample_size(sigma=1.0, margin=0.1, alpha=0.05, power=0.8)
    expected = ((z_quantile(0.95) + z_quantile(0.8)) / 0.1) ** 2
    assert n == math.ceil(expected)


def test_planned_sample_size_achieves_power_and_size() -> None:
    rng = np.random.default_rng(0)
    sigma, margin, alpha, power = 2.0, 0.5, 0.05, 0.8
    n = non_inferiority_sample_size(sigma, margin, alpha, power)
    reps = 3000
    data_equal = rng.normal(0.0, sigma, size=(reps, n))
    data_at_margin = rng.normal(margin, sigma, size=(reps, n))
    hits = np.mean([paired_z_test(row, margin).non_inferior(alpha) for row in data_equal])
    false = np.mean([paired_z_test(row, margin).non_inferior(alpha) for row in data_at_margin])
    assert hits == pytest.approx(power, abs=0.03)
    assert false == pytest.approx(alpha, abs=0.015)


def test_superiority_test_size_and_repeated_peeking_inflation() -> None:
    rng = np.random.default_rng(1)
    alpha, reps, looks, per_look = 0.05, 2000, 40, 25
    data = rng.normal(0, 1, size=(reps, looks * per_look))
    single = np.mean([paired_z_test(row, 0.1).candidate_worse(alpha) for row in data])
    assert single == pytest.approx(alpha, abs=0.015)
    peeked = 0
    for row in data:
        peeked += any(
            paired_z_test(row[: k * per_look], 0.1).candidate_worse(alpha)
            for k in range(1, looks + 1)
        )
    # Armitage et al. (1969): 40 looks at nominal 5% give roughly 20-25% type I error.
    assert peeked / reps > 3 * alpha


def test_fixed_validation() -> None:
    with pytest.raises(ConfigError):
        non_inferiority_sample_size(0.0, 0.1, 0.05, 0.8)
    with pytest.raises(ConfigError):
        non_inferiority_sample_size(1.0, 0.1, 0.05, 0.8, true_degradation=0.2)
    with pytest.raises(ConfigError):
        z_quantile(1.0)
    with pytest.raises(ValueError, match="two pairs"):
        paired_z_test([1.0], 0.1)
