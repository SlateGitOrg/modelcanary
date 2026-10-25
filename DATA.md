# Data

## UCI Optical Recognition of Handwritten Digits (demo only)

- **What**: 1,797 8x8 grey-level images of handwritten digits, 10 classes, 64 integer features (0-16).
- **How it is obtained**: `sklearn.datasets.load_digits()`, which reads a copy bundled inside the
  scikit-learn wheel. Nothing is downloaded and nothing is committed to this repository.
- **Source**: E. Alpaydin, C. Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine
  Learning Repository, 1998, https://archive.ics.uci.edu/dataset/80 (the scikit-learn copy is the test
  partition of that dataset).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI
  repository page.
- **Use**: split once, stratified 50/50 with seed 7, into 898 training rows and an 899-row held-out
  pool. The demo replays the pool (sampling with replacement) as production traffic and returns
  the true labels as feedback.

## Simulated data

- **Business outcome in the demo** is simulated: a user converts with probability 0.30 when the
  served prediction was correct and 0.10 otherwise (`modelcanary/demo/run.py`).
- **Evaluation traffic** is fully synthetic (`modelcanary/sim`): Bernoulli errors and conversions,
  lognormal latencies, failure injection; ground truth is known by construction. Parameters are
  listed in `modelcanary/sim/scenarios.py` and in the README.
- **Test fixtures** (`tests/conftest.py`) train tiny scikit-learn models on Gaussian synthetic data
  generated from a fixed seed.
