"""Sticky traffic assignment.

A request's arm is a pure function of ``(salt, routing_key, weight)``: the key
is hashed to a uniform number in ``[0, 1)`` and compared with the candidate
weight.  Consequences:

* the same user (routing key) stays on the same arm for the whole stage, so a
  user never flips between model versions mid-session;
* when the weight grows from 5% to 25%, every user already on the candidate
  stays there (the candidate set only grows);
* the salt is the rollout id, so different rollouts draw independent splits.
"""

from __future__ import annotations

import hashlib
import struct

from modelcanary.engine.pairing import Arm

_SCALE = float(2**64)


def bucket(salt: str, routing_key: str) -> float:
    """Uniform hash of ``routing_key`` into ``[0, 1)``."""
    digest = hashlib.blake2b(f"{salt}\x00{routing_key}".encode(), digest_size=8).digest()
    value: int = struct.unpack(">Q", digest)[0]
    return value / _SCALE


def assign_arm(salt: str, routing_key: str, weight: float) -> Arm:
    """Arm serving ``routing_key`` when the candidate receives ``weight`` of traffic."""
    if weight <= 0.0:
        return Arm.BASELINE
    if weight >= 1.0:
        return Arm.CANDIDATE
    return Arm.CANDIDATE if bucket(salt, routing_key) < weight else Arm.BASELINE
