"""Pairing of unpaired canary observations.

In canary mode each request is served by exactly one arm, so the two arms
produce independent streams.  :class:`PairBuffer` matches the i-th unmatched
baseline observation with the i-th unmatched candidate observation (FIFO).
Because the arms are independent and each stream is i.i.d. under a stationary
workload, the resulting pairs are i.i.d. with mean difference equal to the true
difference in arm means -- which is what the paired sequential tests need.

The baseline arm usually receives far more traffic (e.g. 95/5).  Its surplus is
kept in a bounded buffer and the *oldest* surplus is dropped, so paired
observations stay close in time and slow drifts in the workload affect both
members of a pair alike.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modelcanary.errors import ConfigError

_EMPTY = np.empty(0, dtype=np.float64)


class Arm(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class PairBuffer:
    """FIFO matcher with a bounded backlog per arm."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ConfigError("capacity must be >= 1")
        self.capacity = capacity
        self._pending: dict[Arm, NDArray[np.float64]] = {
            Arm.BASELINE: _EMPTY,
            Arm.CANDIDATE: _EMPTY,
        }
        self.dropped = 0
        self.paired = 0

    def backlog(self, arm: Arm) -> int:
        return int(self._pending[arm].size)

    def push(self, arm: Arm, values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add observations for ``arm``; return any newly completed (baseline, candidate) pairs."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size:
            self._pending[arm] = np.concatenate((self._pending[arm], arr))
        base, cand = self._pending[Arm.BASELINE], self._pending[Arm.CANDIDATE]
        m = min(base.size, cand.size)
        out_base, out_cand = base[:m], cand[:m]
        self._pending[Arm.BASELINE] = self._trim(base[m:])
        self._pending[Arm.CANDIDATE] = self._trim(cand[m:])
        self.paired += m
        return out_base, out_cand

    def _trim(self, rest: NDArray[np.float64]) -> NDArray[np.float64]:
        if rest.size > self.capacity:
            self.dropped += int(rest.size - self.capacity)
            return rest[-self.capacity :]
        return rest
