"""Numerically stable running moments (Welford / Chan et al. parallel update)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


class RunningMoments:
    """Streaming count, mean and variance.

    ``update_many`` merges a whole batch with the Chan-Golub-LeVeque formula, so
    feeding observations one by one or in batches gives identical results up to
    floating-point rounding.
    """

    __slots__ = ("_m2", "_mean", "_n")

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n else math.nan

    @property
    def variance(self) -> float:
        """Unbiased sample variance (``nan`` with fewer than two observations)."""
        return self._m2 / (self._n - 1) if self._n > 1 else math.nan

    @property
    def population_variance(self) -> float:
        return self._m2 / self._n if self._n else math.nan

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def update_many(self, xs: ArrayLike) -> None:
        arr = np.asarray(xs, dtype=np.float64).ravel()
        k = arr.size
        if k == 0:
            return
        batch_mean = float(arr.mean())
        batch_m2 = float(((arr - batch_mean) ** 2).sum())
        n = self._n + k
        delta = batch_mean - self._mean
        self._mean += delta * k / n
        self._m2 += batch_m2 + delta * delta * self._n * k / n
        self._n = n
