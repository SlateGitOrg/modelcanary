"""Tamper-evident decision audit trail.

Each record is appended as one JSON line and carries the SHA-256 of its own
canonical encoding chained to the previous record's hash::

    hash_i = sha256(hash_{i-1} || canonical_json(record_i without "hash"))

Editing, deleting or reordering any line breaks every later hash, which
:func:`verify_records` detects.  The chain makes tampering *evident*; it does
not prevent someone with write access from rewriting the whole file (see the
limitations in the README).
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelcanary.errors import AuditIntegrityError

GENESIS_HASH = "0" * 64


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars / non-finite floats into strict-JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool | int | str) or value is None:
        return value
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(prev_hash: str, body: Mapping[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    seq: int
    ts: str
    rollout_id: str
    event: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "rollout_id": self.rollout_id,
            "event": self.event,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "hash": self.hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditRecord:
        return cls(
            seq=int(data["seq"]),
            ts=str(data["ts"]),
            rollout_id=str(data["rollout_id"]),
            event=str(data["event"]),
            payload=dict(data["payload"]),
            prev_hash=str(data["prev_hash"]),
            hash=str(data["hash"]),
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    records: int
    first_bad_seq: int | None = None
    reason: str | None = None


def verify_records(records: Iterable[AuditRecord]) -> VerifyResult:
    """Check sequence numbers, back-links and hashes of a full chain."""
    prev = GENESIS_HASH
    count = 0
    for expected_seq, rec in enumerate(records):
        count += 1
        if rec.seq != expected_seq:
            return VerifyResult(False, count, rec.seq, f"expected seq {expected_seq}")
        if rec.prev_hash != prev:
            return VerifyResult(False, count, rec.seq, "prev_hash does not match previous record")
        if _hash(prev, rec.body()) != rec.hash:
            return VerifyResult(False, count, rec.seq, "hash mismatch (record modified)")
        prev = rec.hash
    return VerifyResult(True, count)


def read_records(path: str | Path) -> list[AuditRecord]:
    out: list[AuditRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                out.append(AuditRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise AuditIntegrityError(f"line {lineno}: malformed audit record: {exc}") from exc
    return out


def verify_file(path: str | Path) -> VerifyResult:
    try:
        records = read_records(path)
    except AuditIntegrityError as exc:
        return VerifyResult(False, 0, None, str(exc))
    return verify_records(records)


class AuditLog:
    """Append-only, hash-chained audit log with an optional JSONL file sink.

    The full chain is persisted to ``path`` (if given); the most recent
    ``memory_limit`` records are also kept in memory for the HTTP API.
    Thread-safe.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        memory_limit: int = 10_000,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[AuditRecord] = deque(maxlen=memory_limit)
        self._all_in_memory: list[AuditRecord] | None = [] if self.path is None else None
        self._seq = 0
        self._last_hash = GENESIS_HASH
        if self.path is not None and self.path.exists():
            existing = read_records(self.path)
            result = verify_records(existing)
            if not result.ok:
                raise AuditIntegrityError(
                    f"refusing to extend a broken chain at seq {result.first_bad_seq}: "
                    f"{result.reason}"
                )
            self._recent.extend(existing)
            if existing:
                self._seq = existing[-1].seq + 1
                self._last_hash = existing[-1].hash

    def append(self, rollout_id: str, event: str, payload: Mapping[str, Any]) -> AuditRecord:
        with self._lock:
            ts = datetime.fromtimestamp(self._clock(), tz=UTC).isoformat(timespec="milliseconds")
            clean = _jsonable(dict(payload))
            body = {
                "seq": self._seq,
                "ts": ts,
                "rollout_id": rollout_id,
                "event": event,
                "payload": clean,
                "prev_hash": self._last_hash,
            }
            record = AuditRecord(
                self._seq,
                ts,
                rollout_id,
                event,
                clean,
                self._last_hash,
                _hash(self._last_hash, body),
            )
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(canonical_json(record.to_dict()) + "\n")
            else:
                assert self._all_in_memory is not None
                self._all_in_memory.append(record)
            self._recent.append(record)
            self._seq += 1
            self._last_hash = record.hash
            return record

    def __len__(self) -> int:
        return self._seq

    def recent(self, limit: int = 100, rollout_id: str | None = None) -> list[AuditRecord]:
        with self._lock:
            items = [r for r in self._recent if rollout_id is None or r.rollout_id == rollout_id]
        return items[-limit:] if limit > 0 else []

    def verify(self) -> VerifyResult:
        if self.path is not None:
            return verify_file(self.path)
        with self._lock:
            assert self._all_in_memory is not None
            return verify_records(list(self._all_in_memory))
