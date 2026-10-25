"""Train the demo's scikit-learn models and export them to ONNX.

Data: the UCI "Optical Recognition of Handwritten Digits" set as bundled with
scikit-learn (``load_digits``: 1,797 8x8 images, 10 classes; no download).
It is split once, stratified, into a training half and a held-out *traffic
pool*; the demo replays the pool as production traffic, so every model's
true error rate on that traffic is known exactly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skl2onnx import to_onnx
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True, slots=True)
class DemoModel:
    model_id: str
    description: str
    path: Path
    pool_error_rate: float | None  # None when the model cannot score the pool (broken)


@dataclass(frozen=True, slots=True)
class DemoData:
    pool_x: NDArray[np.float32]
    pool_y: NDArray[np.int64]
    train_size: int


def _export(est: ClassifierMixin, sample: NDArray[np.float32], path: Path) -> None:
    onx = to_onnx(est, sample[:1], options={"zipmap": False}, target_opset=17)
    path.write_bytes(onx.SerializeToString())


def build_models(out_dir: Path, seed: int) -> tuple[DemoData, list[DemoModel]]:
    """Fit the four demo models, export them to ``out_dir`` and return the traffic pool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_digits(return_X_y=True)
    x = x.astype(np.float32)
    x_train, x_pool, y_train, y_pool = train_test_split(
        x, y, test_size=0.5, stratify=y, random_state=seed
    )
    specs: list[tuple[str, str, ClassifierMixin, int]] = [
        (
            "lr-v1",
            "logistic regression (current production)",
            LogisticRegression(max_iter=5000),
            64,
        ),
        ("svc-v2", "RBF support-vector classifier", SVC(), 64),
        ("knn-v3", "3-nearest-neighbours: accurate but slow", KNeighborsClassifier(3), 64),
        (
            "tree-v4",
            "depth-5 decision tree: fast but inaccurate",
            DecisionTreeClassifier(max_depth=5, random_state=seed),
            64,
        ),
        ("svc-v5-32px", "SVC exported for the top half of the image (schema mismatch)", SVC(), 32),
    ]
    models = []
    with warnings.catch_warnings():
        # skl2onnx reads SVC.probA_/probB_, which scikit-learn 1.9 marks as deprecated.
        warnings.simplefilter("ignore", FutureWarning)
        for model_id, description, est, width in specs:
            est.fit(x_train[:, :width], y_train)
            path = out_dir / f"{model_id}.onnx"
            _export(est, x_train[:, :width], path)
            err = float((est.predict(x_pool[:, :width]) != y_pool).mean()) if width == 64 else None
            models.append(DemoModel(model_id, description, path, err))
    return DemoData(x_pool, y_pool.astype(np.int64), len(y_train)), models
