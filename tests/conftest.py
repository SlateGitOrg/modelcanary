"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modelcanary.engine.policy import RolloutPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "default.toml"


@pytest.fixture
def default_policy() -> RolloutPolicy:
    return RolloutPolicy.from_toml(DEFAULT_POLICY)


def small_policy(**overrides: Any) -> RolloutPolicy:
    """A compact two-stage policy for fast controller tests."""
    data: dict[str, Any] = {
        "false_rollback_budget": 0.05,
        "false_promotion_budget": 0.05,
        "check_every": 1,
        "guardrails": {"max_requests": 50_000, "min_failure_attempts": 20},
        "stages": [
            {"name": "shadow", "weight": 0.0, "shadow": True, "min_requests": 100},
            {"name": "canary", "weight": 0.5},
        ],
        "metrics": [
            {
                "name": "error_rate",
                "source": "error",
                "direction": "lower_is_better",
                "method": "sprt",
                "odds_ratio": 2.0,
                "tolerance": 0.05,
                "planning_sd": 0.4,
            },
            {
                "name": "latency_ms",
                "source": "latency",
                "direction": "lower_is_better",
                "method": "asymptotic_cs",
                "tolerance": 2.0,
                "min_pairs": 20,
                "planning_sd": 5.0,
            },
        ],
    }
    data.update(overrides)
    return RolloutPolicy.from_mapping(data)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny real scikit-learn classifiers exported to ONNX.

    * ``base`` / ``twin`` -- logistic regressions on the informative features;
    * ``weak`` -- a stump that only looks at a pure-noise feature (~50% error);
    * ``broken`` -- trained on 3 features, so serving 4-feature rows fails.
    """
    import numpy as np
    from skl2onnx import to_onnx
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    weak = Pipeline(
        [
            ("pick", ColumnTransformer([("noise", "passthrough", [3])])),
            ("tree", DecisionTreeClassifier(max_depth=1, random_state=0)),
        ]
    )
    models: dict[str, tuple[Any, Any]] = {
        "base": (LogisticRegression().fit(x, y), x),
        "twin": (LogisticRegression(C=0.5).fit(x, y), x),
        "weak": (weak.fit(x, y), x),
        "broken": (LogisticRegression().fit(x[:, :3], y), x[:, :3]),
    }
    out = tmp_path_factory.mktemp("models")
    for name, (est, sample) in models.items():
        onx = to_onnx(est, sample[:1], options={"zipmap": False}, target_opset=17)
        (out / f"{name}.onnx").write_bytes(onx.SerializeToString())
    return out


def labelled_rows(n: int, seed: int = 1) -> tuple[Any, Any]:
    """Rows and labels from the same distribution the fixture models were trained on."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    return x, (x[:, 0] + x[:, 1] > 0).astype(np.int64)
