"""Command-line interface and the end-to-end demo."""

import json
from pathlib import Path
from typing import Any

import pytest

from modelcanary import cli
from modelcanary.engine.policy import RolloutPolicy
from tests.conftest import DEFAULT_POLICY

POLICY = str(DEFAULT_POLICY)


def test_simulate_prints_trail(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    code = cli.main(
        ["simulate", "--scenario", "worse-latency", "--policy", POLICY, "--audit", str(audit)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "decision  : ROLLBACK" in out
    assert "verdict_changed" in out
    assert "audit chain verified: True" in out
    assert cli.main(["audit", str(audit)]) == 0
    assert "hash chain intact" in capsys.readouterr().out


def test_simulate_json_and_unknown_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["simulate", "--scenario", "better", "--json", "--policy", POLICY]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["action"] == "promote"
    assert data["truth"] == "better"
    assert cli.main(["simulate", "--scenario", "nope", "--policy", POLICY]) == 2


def test_audit_detects_tampering(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    audit = tmp_path / "a.jsonl"
    cli.main(["simulate", "--scenario", "broken", "--policy", POLICY, "--audit", str(audit)])
    text = audit.read_text(encoding="utf-8").replace('"guardrail', '"GUARDRAIL', 1)
    audit.write_text(text, encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["audit", str(audit)]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_oc_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["oc", "--policy", POLICY]) == 0
    out = capsys.readouterr().out
    assert "| 1.00 | 0.5000 |" in out
    assert "P(rollback) exact" in out


def test_tune_writes_loadable_policy(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "tuned.json"
    assert cli.main(["tune", "--pilot-seeds", "2", "--policy", POLICY, "--out", str(out)]) == 0
    assert "planning_sd" in capsys.readouterr().out
    tuned = RolloutPolicy.from_file(out)
    assert tuned.metrics[0].planning_sd is not None
    assert cli.main(["simulate", "--scenario", "equal", "--policy", str(out)]) == 0


def test_evaluate_small_grid(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = cli.main(
        [
            "evaluate",
            "--seeds", "3",
            "--pilot-seeds", "2",
            "--scenario", "worse-latency",
            "--scenario", "broken",
            "--policy", POLICY,
            "--out", str(tmp_path),
        ]
    )  # fmt: skip
    assert code == 0
    assert "### Error rates by strategy" in capsys.readouterr().out
    data = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert data["runs"] == 2 * 3 * 3
    assert data["config"]["first_eval_seed"] == 1_000_000
    assert (tmp_path / "eval.md").exists()
    assert cli.main(["evaluate", "--seeds", "1", "--scenario", "nope", "--policy", POLICY]) == 1


def test_bad_policy_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert cli.main(["oc", "--policy", str(bad)]) == 1
    assert "must be a JSON object" in capsys.readouterr().err
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert cli.main(["oc", "--policy", str(broken)]) == 1


def test_serve_wires_gateway(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    code = cli.main(
        [
            "serve",
            "--models",
            str(model_dir),
            "--production",
            "base",
            "--policy",
            POLICY,
            "--port",
            "9999",
        ]
    )
    assert code == 0
    assert captured["port"] == 9999
    assert captured["app"].state.gateway.production_id == "base"
    with pytest.raises(SystemExit):
        cli.main(["serve", "--models", str(model_dir)])


def test_demo_end_to_end(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "demo"
    assert cli.main(["demo", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "verified ok=True" in text
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    actions = {r["candidate"]: (r["action"], r["reason"]) for r in summary["rollouts"]}
    assert actions["svc-v2"][0] == "promote"
    assert actions["knn-v3"] == ("rollback", "metric worse than baseline: latency_ms")
    assert actions["tree-v4"] == ("rollback", "metric worse than baseline: error_rate")
    assert actions["svc-v5-32px"][1].startswith("guardrail")
