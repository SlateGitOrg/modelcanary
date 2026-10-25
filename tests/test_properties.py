"""Property-based tests for invariants that must hold for any input."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcanary.audit import AuditLog
from modelcanary.engine.pairing import Arm, PairBuffer
from modelcanary.serving.router import assign_arm
from modelcanary.stats.sprt import BernoulliSPRT, SPRTDecision

pushes = st.lists(
    st.tuples(st.sampled_from([Arm.BASELINE, Arm.CANDIDATE]), st.integers(0, 40)), max_size=30
)


@settings(max_examples=80, deadline=None)
@given(pushes, st.integers(1, 20))
def test_pair_buffer_conserves_observations(ops: list[tuple[Arm, int]], capacity: int) -> None:
    buf = PairBuffer(capacity)
    pushed = {Arm.BASELINE: 0, Arm.CANDIDATE: 0}
    paired = 0
    for arm, n in ops:
        b, c = buf.push(arm, np.arange(n, dtype=float))
        assert b.size == c.size
        paired += b.size
        pushed[arm] += n
        assert buf.backlog(arm) <= capacity
    assert paired == buf.paired
    total_pushed = pushed[Arm.BASELINE] + pushed[Arm.CANDIDATE]
    backlog = buf.backlog(Arm.BASELINE) + buf.backlog(Arm.CANDIDATE)
    assert total_pushed == 2 * paired + backlog + buf.dropped
    assert min(buf.backlog(Arm.BASELINE), buf.backlog(Arm.CANDIDATE)) == 0


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=20), st.text(max_size=20), st.floats(0, 1), st.floats(0, 1))
def test_router_is_monotone_in_weight(salt: str, key: str, w1: float, w2: float) -> None:
    lo, hi = min(w1, w2), max(w1, w2)
    if assign_arm(salt, key, lo) is Arm.CANDIDATE:
        assert assign_arm(salt, key, hi) is Arm.CANDIDATE


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=5),
            st.one_of(st.integers(), st.floats(allow_nan=True), st.text(max_size=5), st.none()),
            max_size=4,
        ),
        max_size=10,
    )
)
def test_audit_chain_always_verifies(payloads: list[dict[str, object]]) -> None:
    log = AuditLog(clock=lambda: 0.0)
    for p in payloads:
        log.append("ro", "event", p)
    assert log.verify().ok
    assert len(log) == len(payloads)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.booleans(), max_size=400), st.integers(1, 50))
def test_sprt_is_chunking_invariant(xs: list[bool], chunk: int) -> None:
    whole = BernoulliSPRT(0.5, 0.7, 0.1, 0.1)
    whole.update_many(np.asarray(xs, dtype=bool))
    parts = BernoulliSPRT(0.5, 0.7, 0.1, 0.1)
    for i in range(0, len(xs), chunk):
        parts.update_many(np.asarray(xs[i : i + chunk], dtype=bool))
    assert whole.decision is parts.decision
    assert whole.n == parts.n
    if whole.decision is SPRTDecision.CONTINUE:
        assert whole.n == len(xs)
