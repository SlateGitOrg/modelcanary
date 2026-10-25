"""ONNX Runtime model registry.

Each model is an ONNX classifier (for example a scikit-learn estimator exported
with ``skl2onnx``) whose first output is the predicted label.  Inference is
timed with a monotonic clock around ``InferenceSession.run`` only, so the
latency metric measures the model, not HTTP or JSON overhead.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from modelcanary.errors import ConfigError, InferenceError, ModelNotFoundError
from modelcanary.logging_setup import get_logger

log = get_logger("registry")


@dataclass(frozen=True, slots=True)
class Prediction:
    labels: NDArray[np.int64]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_id: str
    sha256: str
    input_name: str
    input_shape: tuple[int | str | None, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "sha256": self.sha256,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "outputs": list(self.outputs),
        }


class OnnxModel:
    """A loaded ONNX classifier with a single float tensor input."""

    def __init__(self, model_id: str, model_bytes: bytes, intra_op_threads: int = 1) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(
                model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises several unrelated types
            raise ConfigError(f"cannot load ONNX model {model_id!r}: {exc}") from exc
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ConfigError(f"model {model_id!r} must have exactly one input, has {len(inputs)}")
        self._input_name: str = inputs[0].name
        outputs = tuple(o.name for o in self._session.get_outputs())
        self.info = ModelInfo(
            model_id=model_id,
            sha256=hashlib.sha256(model_bytes).hexdigest(),
            input_name=self._input_name,
            input_shape=tuple(inputs[0].shape),
            outputs=outputs,
        )

    @classmethod
    def from_path(cls, model_id: str, path: str | Path) -> OnnxModel:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot read model file {path}: {exc}") from exc
        return cls(model_id, data)

    @property
    def model_id(self) -> str:
        return self.info.model_id

    def predict(self, features: NDArray[np.float32]) -> Prediction:
        """Predict labels for a ``[n, d]`` batch; raises :class:`InferenceError` on failure."""
        start = time.perf_counter()
        try:
            outputs = self._session.run(None, {self._input_name: features})
        except Exception as exc:  # InvalidArgument, Fail, RuntimeException, ...
            raise InferenceError(f"model {self.model_id!r} failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        labels = np.asarray(outputs[0]).astype(np.int64).ravel()
        if labels.shape[0] != features.shape[0]:
            raise InferenceError(
                f"model {self.model_id!r} returned {labels.shape[0]} labels "
                f"for {features.shape[0]} rows"
            )
        return Prediction(labels, latency_ms)


class ModelRegistry:
    """Thread-safe mapping of model id to loaded :class:`OnnxModel`."""

    def __init__(self) -> None:
        self._models: dict[str, OnnxModel] = {}
        self._lock = threading.Lock()

    def register(self, model: OnnxModel) -> None:
        with self._lock:
            if model.model_id in self._models:
                raise ConfigError(f"model {model.model_id!r} is already registered")
            self._models[model.model_id] = model
        log.info("model_registered id=%s sha256=%s", model.model_id, model.info.sha256[:12])

    def register_path(self, model_id: str, path: str | Path) -> OnnxModel:
        model = OnnxModel.from_path(model_id, path)
        self.register(model)
        return model

    def load_directory(self, directory: str | Path) -> list[str]:
        """Register every ``*.onnx`` file in ``directory`` under its file stem."""
        root = Path(directory)
        if not root.is_dir():
            raise ConfigError(f"model directory {root} does not exist")
        ids = []
        for path in sorted(root.glob("*.onnx")):
            self.register_path(path.stem, path)
            ids.append(path.stem)
        return ids

    def get(self, model_id: str) -> OnnxModel:
        with self._lock:
            try:
                return self._models[model_id]
            except KeyError:
                raise ModelNotFoundError(f"unknown model {model_id!r}") from None

    def __contains__(self, model_id: object) -> bool:
        with self._lock:
            return model_id in self._models

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def infos(self) -> list[ModelInfo]:
        with self._lock:
            return [self._models[k].info for k in sorted(self._models)]
