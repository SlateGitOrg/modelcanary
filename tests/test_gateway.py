"""Gateway core and HTTP API, end to end with real ONNX models."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelcanary.audit import AuditLog
from modelcanary.engine.controller import Action, RolloutStatus
from modelcanary.engine.pairing import Arm
from modelcanary.errors import ConfigError, ModelNotFoundError, RolloutStateError
from modelcanary.serving.app import create_app
from modelcanary.serving.feedback import FeedbackStore
from modelcanary.serving.gateway import FeedbackItem, Gateway, to_features
from modelcanary.serving.registry import ModelRegistry
from tests.conftest import labelled_rows, small_policy

STAGES = [
    {"name": "shadow", "weight": 0.0, "shadow": True, "min_requests": 300},
    {"name": "canary", "weight": 0.5},
]


def _policy(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "stages": STAGES,
        "promote_min_stage": 1,
        "guardrails": {
            "max_requests": 50_000,
            "min_failure_attempts": 20,
            "min_candidate_requests_before_promote": 100,
        },
        "metrics": [
            {
                "name": "error_rate",
                "source": "error",
                "direction": "lower_is_better",
                "method": "sprt",
                "odds_ratio": 2.0,
                "tolerance": 0.05,
                "min_pairs": 50,
            },
            {
                "name": "latency_ms",
                "source": "latency",
                "direction": "lower_is_better",
                "method": "asymptotic_cs",
                "tolerance": 50.0,
                "min_pairs": 20,
            },
        ],
    }
    base.update(overrides)
    return small_policy(**base)


@pytest.fixture
def gateway(model_dir: Path) -> Gateway:
    reg = ModelRegistry()
    reg.load_directory(model_dir)
    return Gateway(reg, "base", default_policy=_policy(), audit=AuditLog())


def _drive(gw: Gateway, steps: int, batch: int = 10, seed: int = 0) -> None:
    """Send traffic plus immediate feedback until the rollout decides."""
    for step in range(steps):
        ctrl = gw.active_controller()
        if ctrl is None:
            return
        x, y = labelled_rows(batch, seed=seed * 10_000 + step)
        keys = [f"user-{step}-{i}" for i in range(batch)]
        result = gw.predict(x, keys)
        if result.shadow_job is not None:
            gw.run_shadow(result.shadow_job)
        gw.feedback(
            [
                FeedbackItem(p.prediction_id, int(lbl))
                for p, lbl in zip(result.predictions, y, strict=True)
            ]
        )


def test_without_rollout_production_serves_everything(gateway: Gateway) -> None:
    x, _ = labelled_rows(5)
    result = gateway.predict(x)
    assert result.rollout_id is None
    assert {p.model_id for p in result.predictions} == {"base"}
    summary = gateway.feedback([FeedbackItem(result.predictions[0].prediction_id, 1)])
    assert summary.unmatched == 1
    assert gateway.status()["rollout"] is None


def test_near_identical_candidate_is_promoted_and_becomes_production(gateway: Gateway) -> None:
    ctrl = gateway.start_rollout("twin", rollout_id="ro-twin")
    _drive(gateway, 400)
    assert ctrl.status is RolloutStatus.PROMOTED
    assert gateway.production_id == "twin"
    assert ctrl.decision is not None
    assert ctrl.decision.stage == "canary"
    assert ctrl.candidate_requests >= 100
    assert gateway.history()[0]["status"] == "promoted"
    x, _ = labelled_rows(3)
    assert {p.model_id for p in gateway.predict(x).predictions} == {"twin"}
    assert gateway.audit.verify().ok
    events = [r.event for r in gateway.audit.recent(1000, rollout_id="ro-twin")]
    assert events[0] == "rollout_started"
    assert "stage_advanced" in events
    assert events[-1] == "promoted"


def test_weak_candidate_is_rolled_back_in_shadow_without_exposure(gateway: Gateway) -> None:
    ctrl = gateway.start_rollout("weak")
    _drive(gateway, 200)
    assert ctrl.status is RolloutStatus.ROLLED_BACK
    assert ctrl.decision is not None
    assert "error_rate" in ctrl.decision.reason
    assert ctrl.candidate_requests == 0
    assert gateway.production_id == "base"


def test_broken_candidate_trips_guardrail_in_shadow(gateway: Gateway) -> None:
    ctrl = gateway.start_rollout("broken")
    _drive(gateway, 10)
    assert ctrl.status is RolloutStatus.ROLLED_BACK
    assert ctrl.decision is not None
    assert ctrl.decision.reason.startswith("guardrail")
    assert gateway.counters.value("modelcanary_inference_failures_total", model="broken") >= 20


def test_canary_failures_fall_back_to_production(gateway: Gateway) -> None:
    policy = _policy(stages=[{"name": "all-in", "weight": 1.0}], promote_min_stage=0)
    ctrl = gateway.start_rollout("broken", policy)
    x, _ = labelled_rows(10)
    result = gateway.predict(x, [f"k{i}" for i in range(10)])
    assert {p.arm for p in result.predictions} == {Arm.BASELINE}
    assert {p.model_id for p in result.predictions} == {"base"}
    assert ctrl.candidate_failures == 10
    _drive(gateway, 5)
    assert ctrl.decision is not None
    assert ctrl.decision.reason.startswith("guardrail")


def test_canary_split_is_sticky(gateway: Gateway) -> None:
    policy = _policy(stages=[{"name": "half", "weight": 0.5}], promote_min_stage=0)
    gateway.start_rollout("twin", policy, rollout_id="ro-sticky")
    x, _ = labelled_rows(200)
    keys = [f"user-{i}" for i in range(200)]
    first = [p.arm for p in gateway.predict(x, keys).predictions]
    second = [p.arm for p in gateway.predict(x, keys).predictions]
    assert first == second
    share = sum(a is Arm.CANDIDATE for a in first) / len(first)
    assert 0.35 < share < 0.65


def test_stale_feedback_after_new_rollout(gateway: Gateway) -> None:
    gateway.start_rollout("weak", rollout_id="ro-a")
    x, y = labelled_rows(4)
    old = gateway.predict(x)
    gateway.abort("operator")
    gateway.start_rollout("twin", rollout_id="ro-b")
    summary = gateway.feedback(
        [FeedbackItem(p.prediction_id, int(lbl)) for p, lbl in zip(old.predictions, y, strict=True)]
    )
    assert summary.matched == 4
    assert summary.stale == 4


def test_shadow_feedback_before_scoring_is_unpaired(gateway: Gateway) -> None:
    gateway.start_rollout("twin")
    x, y = labelled_rows(3)
    result = gateway.predict(x)
    summary = gateway.feedback(
        [
            FeedbackItem(p.prediction_id, int(lbl))
            for p, lbl in zip(result.predictions, y, strict=True)
        ]
    )
    assert summary.matched == 3
    assert summary.paired == 0


def test_gateway_state_errors(gateway: Gateway) -> None:
    with pytest.raises(RolloutStateError, match="no rollout"):
        gateway.abort("nothing")
    with pytest.raises(ModelNotFoundError):
        gateway.start_rollout("missing")
    with pytest.raises(RolloutStateError, match="already the production"):
        gateway.start_rollout("base")
    gateway.start_rollout("twin")
    with pytest.raises(RolloutStateError, match="already running"):
        gateway.start_rollout("weak")
    x, _ = labelled_rows(2)
    with pytest.raises(ConfigError, match="routing_keys"):
        gateway.predict(x, ["only-one"])
    with pytest.raises(ModelNotFoundError):
        Gateway(gateway.registry, "missing", default_policy=_policy())


@pytest.mark.parametrize(
    "rows",
    [[[1.0, 2.0], [1.0]], [], [[float("nan"), 1.0]], [[]]],
)
def test_to_features_validation(rows: list[list[float]]) -> None:
    with pytest.raises(ConfigError):
        to_features(rows)


# --------------------------------------------------------------------- HTTP


@pytest.fixture
def client(gateway: Gateway) -> TestClient:
    return TestClient(create_app(gateway))


def test_http_end_to_end_rollout(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "production": "base"}
    models = client.get("/v1/models").json()
    assert [m["model_id"] for m in models["models"]] == ["base", "broken", "twin", "weak"]
    assert client.get("/v1/rollouts/current").status_code == 404

    created = client.post("/v1/rollouts", json={"candidate": "weak", "rollout_id": "ro-http"})
    assert created.status_code == 201
    assert created.json()["shadow"] is True

    for step in range(200):
        x, y = labelled_rows(10, seed=500 + step)
        resp = client.post("/v1/predict", json={"instances": x.tolist()})
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert {p["model_id"] for p in preds} == {"base"}
        items = [
            {"prediction_id": p["prediction_id"], "label": int(lbl)}
            for p, lbl in zip(preds, y, strict=True)
        ]
        client.post("/v1/feedback", json={"items": items})
        if client.get("/v1/rollouts/current").json()["status"] != "running":
            break
    state = client.get("/v1/rollouts/current").json()
    assert state["status"] == "rolled_back"
    assert state["shadow_scored"] > 0
    assert state["decision"]["action"] == Action.ROLLBACK.value
    finished = client.get("/v1/rollouts").json()
    assert finished["finished"][0]["rollout_id"] == "ro-http"

    audit = client.get("/v1/audit", params={"rollout_id": "ro-http", "limit": 500}).json()
    assert audit["records"][-1]["event"] == "rolled_back"
    assert client.get("/v1/audit/verify").json()["ok"] is True
    metrics = client.get("/metrics").text
    assert "modelcanary_shadow_scored_total" in metrics
    assert 'modelcanary_rollout_decisions_total{action="rollback"} 1' in metrics


def test_http_error_mapping(client: TestClient) -> None:
    assert client.post("/v1/rollouts", json={"candidate": "missing"}).status_code == 404
    bad_policy = {"candidate": "twin", "policy": {"metrics": [], "stages": []}}
    assert client.post("/v1/rollouts", json=bad_policy).status_code == 422
    assert client.post("/v1/rollouts/current/abort", json={"reason": "x"}).status_code == 409
    assert client.post("/v1/predict", json={"instances": []}).status_code == 422
    ragged = client.post("/v1/predict", json={"instances": [[1, 2, 3, 4], [1]]})
    assert ragged.status_code == 422
    wrong_width = client.post("/v1/predict", json={"instances": [[1.0, 2.0]]})
    assert wrong_width.status_code == 503  # production model cannot score it
    assert (
        client.post(
            "/v1/feedback", json={"items": [{"prediction_id": "x", "label": 1, "outcome": 3}]}
        ).status_code
        == 422
    )

    custom = _policy(stages=[{"name": "half", "weight": 0.5}], promote_min_stage=0)
    resp = client.post(
        "/v1/rollouts", json={"candidate": "twin", "policy": custom.model_dump(mode="json")}
    )
    assert resp.status_code == 201
    assert resp.json()["candidate_weight"] == 0.5
    assert client.post("/v1/rollouts", json={"candidate": "weak"}).status_code == 409
    aborted = client.post("/v1/rollouts/current/abort", json={"reason": "manual"})
    assert aborted.status_code == 200
    assert aborted.json()["reason"] == "aborted: manual"


def test_feedback_store_is_bounded_in_gateway(model_dir: Path) -> None:
    reg = ModelRegistry()
    reg.load_directory(model_dir)
    gw = Gateway(reg, "base", default_policy=_policy(), feedback=FeedbackStore(capacity=5))
    gw.start_rollout("twin")
    x, _ = labelled_rows(20)
    gw.predict(x)
    assert gw.status()["pending_feedback"] == 5
    assert gw.gauges()["modelcanary_rollout_running"] == 1.0
    assert np.isclose(gw.gauges()["modelcanary_candidate_weight"], 0.0)
