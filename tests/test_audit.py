"""Hash-chained audit trail."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from modelcanary.audit import (
    GENESIS_HASH,
    AuditLog,
    canonical_json,
    read_records,
    verify_file,
    verify_records,
)
from modelcanary.errors import AuditIntegrityError


def _filled(path: Path | None = None, n: int = 5) -> AuditLog:
    log = AuditLog(path, clock=lambda: 1_700_000_000.0)
    for i in range(n):
        log.append("ro-1", "event", {"i": i, "x": np.float64(0.5), "arr": (1, 2)})
    return log


def test_chain_links_and_verifies_in_memory() -> None:
    log = _filled()
    recs = log.recent(10)
    assert recs[0].prev_hash == GENESIS_HASH
    assert all(b.prev_hash == a.hash for a, b in itertools.pairwise(recs))
    assert recs[0].payload == {"i": 0, "x": 0.5, "arr": [1, 2]}
    assert recs[0].ts.startswith("2023-11-14T22:13:20")
    assert log.verify().ok
    assert len(log) == 5


def test_non_finite_and_exotic_values_become_strict_json() -> None:
    log = AuditLog()
    rec = log.append("r", "e", {"nan": math.nan, "inf": np.inf, "obj": object, "n": np.int64(3)})
    assert rec.payload["nan"] is None
    assert rec.payload["inf"] is None
    assert rec.payload["n"] == 3
    assert isinstance(rec.payload["obj"], str)
    canonical_json(rec.to_dict())  # allow_nan=False would raise on NaN


def test_file_sink_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "audit" / "log.jsonl"
    _filled(path, 3)
    assert verify_file(path).ok
    resumed = AuditLog(path)
    assert len(resumed) == 3
    resumed.append("ro-2", "more", {})
    result = verify_file(path)
    assert result.ok
    assert result.records == 4
    assert resumed.verify().ok
    assert [r.rollout_id for r in resumed.recent(10, rollout_id="ro-2")] == ["ro-2"]
    assert resumed.recent(0) == []


@pytest.mark.parametrize("attack", ["edit", "delete", "swap"])
def test_tampering_is_detected(tmp_path: Path, attack: str) -> None:
    path = tmp_path / "log.jsonl"
    _filled(path, 5)
    lines = path.read_text(encoding="utf-8").splitlines()
    if attack == "edit":
        rec = json.loads(lines[2])
        rec["payload"]["i"] = 99
        lines[2] = json.dumps(rec)
    elif attack == "delete":
        del lines[2]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = verify_file(path)
    assert not result.ok
    assert result.first_bad_seq is not None
    with pytest.raises(AuditIntegrityError, match="broken chain"):
        AuditLog(path)


def test_recomputed_hash_still_breaks_the_next_link(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = _filled(path, 3)
    records = read_records(path)
    forged = records[1].payload | {"i": 42}
    # An attacker who re-hashes the edited record still breaks record 2's back-link.
    from modelcanary.audit import AuditRecord, _hash

    body = records[1].body() | {"payload": forged}
    new_hash = _hash(records[1].prev_hash, body)
    records[1] = AuditRecord(**{**body, "hash": new_hash})
    result = verify_records(records)
    assert not result.ok
    assert result.first_bad_seq == 2
    assert log.verify().ok  # the real file is untouched


def test_malformed_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"seq": 0}\n\nnot json\n', encoding="utf-8")
    result = verify_file(path)
    assert not result.ok
    assert result.reason is not None
    assert "malformed" in result.reason
