"""Offline end-to-end demo: four progressive rollouts through the real HTTP gateway.

Everything runs in-process: the FastAPI app is driven through Starlette's
``TestClient`` (real routing, validation, background shadow tasks and JSON
serialisation, no socket).  Traffic is replayed from the held-out digits pool;
labels are returned as feedback immediately.  The business outcome is
**simulated**: a user "converts" with probability 0.30 when the served
prediction was correct and 0.10 when it was wrong.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from modelcanary.audit import AuditLog, verify_file
from modelcanary.demo.models import DemoData, DemoModel, build_models
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.serving.app import create_app
from modelcanary.serving.gateway import Gateway
from modelcanary.serving.registry import ModelRegistry

POLICY_PATH = Path(__file__).resolve().parents[3] / "configs" / "demo.toml"
CONVERT_IF_RIGHT = 0.30
CONVERT_IF_WRONG = 0.10
FEEDBACK_BATCH = 25


def _fmt(v: float | None, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _drive(
    client: TestClient, data: DemoData, rng: np.random.Generator, max_requests: int
) -> dict[str, Any]:
    """Send single-row requests with delayed feedback until the rollout decides."""
    pending: list[dict[str, Any]] = []
    stage = None
    for sent in range(1, max_requests + 1):
        i = int(rng.integers(data.pool_y.size))
        resp = client.post(
            "/v1/predict",
            json={
                "instances": [data.pool_x[i].tolist()],
                "routing_keys": [f"user-{rng.integers(1 << 62)}"],
            },
        )
        resp.raise_for_status()
        pred = resp.json()["predictions"][0]
        correct = pred["label"] == int(data.pool_y[i])
        converted = rng.random() < (CONVERT_IF_RIGHT if correct else CONVERT_IF_WRONG)
        pending.append(
            {
                "prediction_id": pred["prediction_id"],
                "label": int(data.pool_y[i]),
                "outcome": float(converted),
            }
        )
        if len(pending) >= FEEDBACK_BATCH:
            client.post("/v1/feedback", json={"items": pending}).raise_for_status()
            pending = []
            state = client.get("/v1/rollouts/current").json()
            if state["stage"] != stage:
                stage = state["stage"]
                print(
                    f"    {sent:>6,} requests  stage={stage:<10} "
                    f"candidate weight={state['candidate_weight']:.2f}"
                )
            if state["status"] != "running":
                return dict(state)
    return dict(client.get("/v1/rollouts/current").json())


def _print_decision(state: dict[str, Any]) -> None:
    d = state["decision"]
    print(
        f"  => {d['action'].upper()} ({d['reason']}) at stage {d['stage']!r} after "
        f"{d['requests']:,} requests; candidate served {d['candidate_requests']:,}, "
        f"shadow-scored {state['shadow_scored']:,}"
    )
    print(
        "     metric        pairs  baseline   candidate  degradation  interval              verdict"
    )
    for m in d["metrics"]:
        if m["method"] == "sprt":
            interval = f"(SPRT llr {_fmt(m['statistic'], 2)})"
        elif m["lower"] is None:
            interval = "(no data)"
        else:
            interval = f"[{_fmt(m['lower'])}, {_fmt(m['upper'])}]"
        print(
            f"     {m['metric']:<12} {m['pairs']:>6,}  {_fmt(m['baseline_mean']):>8}  "
            f"{_fmt(m['candidate_mean']):>9}  {_fmt(m['degradation']):>11}  "
            f"{interval:<20}  {m['verdict']}"
        )


def run_demo(out_dir: Path, seed: int = 7) -> int:
    started = time.perf_counter()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print("modelcanary demo: progressive delivery of real scikit-learn -> ONNX models")
    print("=" * 78)
    data, models = build_models(out_dir / "models", seed)
    print(
        f"data: UCI digits via scikit-learn, {data.train_size} training rows, "
        f"{data.pool_y.size} held-out rows replayed as traffic (with replacement)"
    )
    print("models (true error rate on the traffic pool):")
    for m in models:
        print(f"  {m.model_id:<12} {_fmt(m.pool_error_rate, 3):>6}  {m.description}")

    registry = ModelRegistry()
    registry.load_directory(out_dir / "models")
    policy = RolloutPolicy.from_file(POLICY_PATH)
    audit_path = out_dir / "audit.jsonl"
    gateway = Gateway(registry, "lr-v1", default_policy=policy, audit=AuditLog(audit_path))
    client = TestClient(create_app(gateway))
    rng = np.random.default_rng(seed)

    plan: list[tuple[str, str]] = [
        ("svc-v2", "more accurate, slightly slower"),
        ("knn-v3", "as accurate, much slower inference"),
        ("tree-v4", "fast, much less accurate"),
        ("svc-v5-32px", "cannot score 64-feature requests"),
    ]
    summary: list[dict[str, Any]] = []
    for n, (candidate, story) in enumerate(plan, start=1):
        production = client.get("/healthz").json()["production"]
        print(f"\n[{n}] rollout {candidate} vs production {production}: {story}")
        client.post(
            "/v1/rollouts", json={"candidate": candidate, "rollout_id": f"demo-{n}-{candidate}"}
        ).raise_for_status()
        state = _drive(client, data, rng, policy.guardrails.max_requests + 1000)
        _print_decision(state)
        summary.append(
            {
                "rollout_id": state["rollout_id"],
                "candidate": candidate,
                "production_before": production,
                "action": state["decision"]["action"],
                "reason": state["decision"]["reason"],
                "stage": state["decision"]["stage"],
                "requests": state["decision"]["requests"],
                "candidate_requests": state["decision"]["candidate_requests"],
            }
        )

    print(f"\nproduction model now: {client.get('/healthz').json()['production']}")
    verify = client.get("/v1/audit/verify").json()
    print(
        f"audit trail: {verify['records']} hash-chained records in {audit_path.name}; "
        f"verified ok={verify['ok']}"
    )
    tampered = out_dir / "audit.tampered.jsonl"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if '"event":"rolled_back"' in line)
    lines[target] = lines[target].replace('"action":"rollback"', '"action":"promote"')
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = verify_file(tampered)
    print(
        f"tamper check: flipping one rollback to 'promote' -> ok={bad.ok}, "
        f"first bad record seq={bad.first_bad_seq} ({bad.reason})"
    )
    metrics = client.get("/metrics").text
    print("prometheus /metrics excerpt:")
    for line in metrics.splitlines():
        if line.startswith("modelcanary_rollout_decisions_total") or line.startswith(
            "modelcanary_inference_failures_total"
        ):
            print(f"  {line}")
    (out_dir / "summary.json").write_text(
        json.dumps({"models": [_model_dict(m) for m in models], "rollouts": summary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\nwrote {out_dir}/summary.json and {audit_path.name}; "
        f"demo took {time.perf_counter() - started:.1f} s"
    )
    return 0


def _model_dict(m: DemoModel) -> dict[str, Any]:
    return {
        "model_id": m.model_id,
        "description": m.description,
        "pool_error_rate": m.pool_error_rate,
    }
