"""modelcanary: progressive model delivery with sequential statistical guardrails.

The package is organised in layers, each usable on its own:

* :mod:`modelcanary.stats` -- sequential tests (Wald SPRT, mixture and betting
  confidence sequences) and the fixed-horizon comparators.
* :mod:`modelcanary.engine` -- metric monitors, pairing, rollout policy and the
  :class:`~modelcanary.engine.controller.RolloutController` state machine.
* :mod:`modelcanary.serving` -- ONNX Runtime model registry and the FastAPI gateway.
* :mod:`modelcanary.sim` / :mod:`modelcanary.evaluation` -- ground-truth simulator
  and the many-seed evaluation harness.
"""

from modelcanary.errors import ModelCanaryError

__all__ = ["ModelCanaryError", "__version__"]
__version__ = "0.1.0"
