"""FastAPI application exposing a :class:`~modelcanary.serving.gateway.Gateway`.

Endpoints (all JSON unless noted):

=======  ===============================  =============================================
Method   Path                             Purpose
=======  ===============================  =============================================
GET      ``/healthz``                     liveness
GET      ``/v1/models``                   registered ONNX models and their hashes
POST     ``/v1/predict``                  route and serve predictions
POST     ``/v1/feedback``                 delayed labels / business outcomes
POST     ``/v1/rollouts``                 start a rollout of a candidate
GET      ``/v1/rollouts/current``         live rollout state and metric statistics
POST     ``/v1/rollouts/current/abort``   operator abort (keeps production)
GET      ``/v1/rollouts``                 finished rollouts
GET      ``/v1/audit``                    recent audit records
GET      ``/v1/audit/verify``             verify the audit hash chain
GET      ``/metrics``                     Prometheus text format
=======  ===============================  =============================================
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from modelcanary import __version__
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.errors import (
    ConfigError,
    InferenceError,
    ModelCanaryError,
    ModelNotFoundError,
    RolloutStateError,
)
from modelcanary.serving.gateway import FeedbackItem, Gateway, to_features

MAX_BATCH = 1024
MAX_FEEDBACK = 10_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(_Strict):
    instances: list[list[float]] = Field(min_length=1, max_length=MAX_BATCH)
    routing_keys: list[str] | None = Field(default=None, max_length=MAX_BATCH)


class PredictionOut(_Strict):
    prediction_id: str
    label: int
    model_id: str
    arm: str


class PredictResponse(_Strict):
    rollout_id: str | None
    predictions: list[PredictionOut]


class FeedbackIn(_Strict):
    prediction_id: str = Field(min_length=1, max_length=64)
    label: int
    outcome: float | None = Field(default=None, ge=0.0, le=1.0)


class FeedbackRequest(_Strict):
    items: list[FeedbackIn] = Field(min_length=1, max_length=MAX_FEEDBACK)


class StartRolloutRequest(_Strict):
    candidate: str = Field(min_length=1)
    policy: dict[str, Any] | None = None
    rollout_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")


class AbortRequest(_Strict):
    reason: str = Field(min_length=1, max_length=500)


def _status_for(exc: ModelCanaryError) -> int:
    if isinstance(exc, ModelNotFoundError):
        return 404
    if isinstance(exc, RolloutStateError):
        return 409
    if isinstance(exc, ConfigError):
        return 422
    if isinstance(exc, InferenceError):
        return 503
    return 500


def create_app(gateway: Gateway) -> FastAPI:
    """Build the HTTP application around an existing gateway."""
    app = FastAPI(title="modelcanary gateway", version=__version__)
    app.state.gateway = gateway

    @app.exception_handler(ModelCanaryError)
    async def _handle(_: Request, exc: ModelCanaryError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(exc), content={"detail": str(exc)})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "production": gateway.production_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "production": gateway.production_id,
            "models": [info.to_dict() for info in gateway.registry.infos()],
        }

    @app.post("/v1/predict")
    def predict(body: PredictRequest, background: BackgroundTasks) -> PredictResponse:
        features = to_features(body.instances)
        result = gateway.predict(features, body.routing_keys)
        if result.shadow_job is not None:
            background.add_task(gateway.run_shadow, result.shadow_job)
        return PredictResponse(
            rollout_id=result.rollout_id,
            predictions=[
                PredictionOut(
                    prediction_id=p.prediction_id,
                    label=p.label,
                    model_id=p.model_id,
                    arm=p.arm.value,
                )
                for p in result.predictions
            ],
        )

    @app.post("/v1/feedback")
    def feedback(body: FeedbackRequest) -> dict[str, int]:
        items = [FeedbackItem(i.prediction_id, i.label, i.outcome) for i in body.items]
        return gateway.feedback(items).to_dict()

    @app.post("/v1/rollouts", status_code=201)
    def start_rollout(body: StartRolloutRequest) -> dict[str, Any]:
        policy = RolloutPolicy.from_mapping(body.policy) if body.policy is not None else None
        ctrl = gateway.start_rollout(body.candidate, policy, body.rollout_id)
        return ctrl.snapshot()

    @app.get("/v1/rollouts/current")
    def current() -> dict[str, Any]:
        ctrl = gateway.controller
        if ctrl is None:
            raise HTTPException(status_code=404, detail="no rollout has been started")
        return ctrl.snapshot()

    @app.post("/v1/rollouts/current/abort")
    def abort(body: AbortRequest) -> dict[str, Any]:
        return gateway.abort(body.reason).to_dict()

    @app.get("/v1/rollouts")
    def rollouts() -> dict[str, Any]:
        return {"production": gateway.production_id, "finished": gateway.history()}

    @app.get("/v1/audit")
    def audit(
        limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
        rollout_id: str | None = None,
    ) -> dict[str, Any]:
        records = gateway.audit.recent(limit, rollout_id)
        return {"records": [r.to_dict() for r in records]}

    @app.get("/v1/audit/verify")
    def audit_verify() -> dict[str, Any]:
        result = gateway.audit.verify()
        return {
            "ok": result.ok,
            "records": result.records,
            "first_bad_seq": result.first_bad_seq,
            "reason": result.reason,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return gateway.counters.render(gateway.gauges())

    return app
