"""Delayed-feedback join.

Ground-truth labels and business outcomes arrive after the prediction (a click,
a chargeback, a human review).  Every prediction served during a rollout is
remembered in a bounded, TTL-limited :class:`FeedbackStore`; when feedback for
it arrives, :func:`apply_feedback` turns it into metric observations:

* shadow request -- both models predicted the same input, so the error
  indicators form a *pair* (``observe_paired``).  The business outcome belongs
  to the answer the user actually saw (the baseline's), so it has no candidate
  counterpart and is not used;
* canary request -- one model served it; error and business outcome are
  recorded for that arm (``observe_arm``) and paired FIFO by the controller.

Unmatched feedback (unknown or expired id) is counted, never guessed.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from modelcanary.engine.controller import RolloutController
from modelcanary.engine.pairing import Arm
from modelcanary.engine.policy import MetricSource
from modelcanary.errors import ConfigError


@dataclass(slots=True)
class PendingPrediction:
    """What the gateway remembers about one served prediction."""

    prediction_id: str
    rollout_id: str
    shadow: bool
    served_arm: Arm
    served_label: int
    created_at: float
    candidate_label: int | None = None  # shadow only; filled in by the background scorer
    candidate_failed: bool = False


@dataclass(frozen=True, slots=True)
class FeedbackOutcome:
    """How one feedback item was used."""

    matched: bool
    paired: bool = False
    stale: bool = False  # the rollout it belonged to is no longer running


class FeedbackStore:
    """Bounded insertion-ordered map with expiry; oldest entries are evicted first."""

    def __init__(
        self,
        capacity: int = 100_000,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or ttl_seconds <= 0:
            raise ConfigError("capacity must be >= 1 and ttl_seconds > 0")
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, PendingPrediction] = OrderedDict()
        self._lock = threading.Lock()
        self.evicted = 0
        self.expired = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def now(self) -> float:
        return self._clock()

    def put(self, item: PendingPrediction) -> None:
        with self._lock:
            self._expire()
            self._items[item.prediction_id] = item
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
                self.evicted += 1

    def get(self, prediction_id: str) -> PendingPrediction | None:
        with self._lock:
            return self._items.get(prediction_id)

    def pop(self, prediction_id: str) -> PendingPrediction | None:
        with self._lock:
            self._expire()
            return self._items.pop(prediction_id, None)

    def _expire(self) -> None:
        cutoff = self._clock() - self.ttl_seconds
        while self._items:
            oldest = next(iter(self._items.values()))
            if oldest.created_at >= cutoff:
                break
            self._items.popitem(last=False)
            self.expired += 1


def apply_feedback(
    controller: RolloutController,
    pending: PendingPrediction,
    label: int,
    outcome: float | None,
) -> FeedbackOutcome:
    """Convert one feedback item into metric observations on ``controller``."""
    if pending.rollout_id != controller.rollout_id or not controller.running:
        return FeedbackOutcome(matched=True, stale=True)
    served_error = float(pending.served_label != label)
    if pending.shadow:
        if pending.candidate_label is None:
            return FeedbackOutcome(matched=True)  # candidate failed or not scored yet
        candidate_error = float(pending.candidate_label != label)
        controller.observe_paired(MetricSource.ERROR, [served_error], [candidate_error])
        return FeedbackOutcome(matched=True, paired=True)
    controller.observe_arm(MetricSource.ERROR, pending.served_arm, [served_error])
    if outcome is not None:
        controller.observe_arm(MetricSource.BUSINESS, pending.served_arm, [outcome])
    return FeedbackOutcome(matched=True)
