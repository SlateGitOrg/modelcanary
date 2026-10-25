"""ONNX registry, sticky router, feedback store and Prometheus counters."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from modelcanary.engine.pairing import Arm
from modelcanary.errors import ConfigError, InferenceError, ModelNotFoundError
from modelcanary.serving.feedback import FeedbackStore, PendingPrediction
from modelcanary.serving.registry import ModelRegistry, OnnxModel
from modelcanary.serving.router import assign_arm, bucket
from modelcanary.serving.telemetry import Counters
from tests.conftest import labelled_rows


def test_registry_loads_directory_and_predicts(model_dir: Path) -> None:
    reg = ModelRegistry()
    assert reg.load_directory(model_dir) == ["base", "broken", "twin", "weak"]
    assert "base" in reg
    assert reg.ids() == ["base", "broken", "twin", "weak"]
    model = reg.get("base")
    x, y = labelled_rows(500)
    pred = model.predict(x)
    assert pred.labels.dtype == np.int64
    assert (pred.labels == y).mean() > 0.95
    assert pred.latency_ms > 0
    info = model.info.to_dict()
    assert info["input_shape"][1] == 4
    assert len(info["sha256"]) == 64
    assert info["outputs"][0] == "label"
    assert len(reg.infos()) == 4


def test_registry_errors(model_dir: Path, tmp_path: Path) -> None:
    reg = ModelRegistry()
    reg.register_path("base", model_dir / "base.onnx")
    with pytest.raises(ConfigError, match="already registered"):
        reg.register_path("base", model_dir / "base.onnx")
    with pytest.raises(ModelNotFoundError):
        reg.get("nope")
    with pytest.raises(ConfigError, match="cannot read"):
        OnnxModel.from_path("x", tmp_path / "missing.onnx")
    with pytest.raises(ConfigError, match="cannot load"):
        OnnxModel("garbage", b"not an onnx model")
    with pytest.raises(ConfigError, match="does not exist"):
        reg.load_directory(tmp_path / "nowhere")


def test_shape_mismatch_raises_inference_error(model_dir: Path) -> None:
    broken = OnnxModel.from_path("broken", model_dir / "broken.onnx")
    x, _ = labelled_rows(3)
    with pytest.raises(InferenceError, match="broken"):
        broken.predict(x)
    assert broken.predict(x[:, :3]).labels.shape == (3,)


def test_bucket_is_uniform_and_deterministic() -> None:
    values = np.array([bucket("salt", f"user-{i}") for i in range(20_000)])
    assert stats.kstest(values, "uniform").pvalue > 0.001
    assert bucket("salt", "u") == bucket("salt", "u")
    assert bucket("salt", "u") != bucket("other", "u")


def test_assignment_is_sticky_and_monotone_in_weight() -> None:
    keys = [f"k{i}" for i in range(5000)]
    at5 = {k for k in keys if assign_arm("ro", k, 0.05) is Arm.CANDIDATE}
    at25 = {k for k in keys if assign_arm("ro", k, 0.25) is Arm.CANDIDATE}
    assert at5 <= at25  # growing the weight never moves a user back to baseline
    assert len(at25) / len(keys) == pytest.approx(0.25, abs=0.02)
    assert assign_arm("ro", "k", 0.0) is Arm.BASELINE
    assert assign_arm("ro", "k", 1.0) is Arm.CANDIDATE


def _pending(pid: str, t: float) -> PendingPrediction:
    return PendingPrediction(pid, "ro", False, Arm.BASELINE, 1, created_at=t)


def test_feedback_store_ttl_and_capacity() -> None:
    now = [0.0]
    store = FeedbackStore(capacity=3, ttl_seconds=10.0, clock=lambda: now[0])
    for i in range(4):
        store.put(_pending(f"p{i}", store.now()))
    assert len(store) == 3
    assert store.evicted == 1
    assert store.get("p0") is None
    assert store.get("p1") is not None
    now[0] = 5.0
    store.put(_pending("late", store.now()))
    now[0] = 12.0  # p1..p3 (t=0) expire, "late" (t=5) survives
    assert store.pop("p2") is None
    assert store.expired >= 2
    item = store.pop("late")
    assert item is not None
    assert item.prediction_id == "late"
    assert store.pop("late") is None
    with pytest.raises(ConfigError):
        FeedbackStore(capacity=0)


def test_counters_render_prometheus_text() -> None:
    c = Counters()
    c.inc("modelcanary_predictions_total", 3, model="a", arm="baseline")
    c.inc("modelcanary_predictions_total", 2, model="a", arm="baseline")
    c.inc("custom_total", model='we"ird')
    assert c.value("modelcanary_predictions_total", model="a", arm="baseline") == 5
    text = c.render({"modelcanary_rollout_running": 1.0})
    assert "# TYPE modelcanary_predictions_total counter" in text
    assert 'modelcanary_predictions_total{arm="baseline",model="a"} 5' in text
    assert 'custom_total{model="we\\"ird"} 1' in text
    assert "modelcanary_rollout_running 1" in text
