# One-command gates. Everything runs offline once dependencies are installed.
UV ?= uv
RUN := $(UV) run --frozen --all-extras

.PHONY: help sync fmt fmt-check lint typecheck test cov check demo eval eval-quick oc serve-demo clean

help:
	@echo "make check   - format check + lint + mypy --strict + tests with coverage gate"
	@echo "make demo    - offline end-to-end demo with real scikit-learn -> ONNX models"
	@echo "make eval    - held-out evaluation (1000 seeds x 7 scenarios x 3 strategies)"
	@echo "make oc      - exact vs Wald operating characteristics of the error-rate SPRT"

sync:
	$(UV) sync --frozen --all-extras

fmt:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

fmt-check: sync
	$(RUN) ruff format --check .

lint: sync
	$(RUN) ruff check .

typecheck: sync
	$(RUN) mypy

test: sync
	$(RUN) pytest

cov: sync
	$(RUN) pytest --cov --cov-report=term

check: fmt-check lint typecheck cov

demo: sync
	$(RUN) modelcanary demo --out var/demo

eval: sync
	$(RUN) python -u -m modelcanary evaluate --seeds 1000 --pilot-seeds 20 --workers 3 --out results

eval-quick: sync
	$(RUN) python -u -m modelcanary evaluate --seeds 50 --pilot-seeds 5 --workers 3

oc: sync
	$(RUN) modelcanary oc

clean:
	rm -rf var .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov .hypothesis
