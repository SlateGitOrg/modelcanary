import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcanary.stats.running import RunningMoments

floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(st.lists(floats, min_size=0, max_size=60), st.integers(min_value=1, max_value=7))
def test_batch_and_scalar_updates_agree_with_numpy(xs: list[float], chunk: int) -> None:
    one = RunningMoments()
    many = RunningMoments()
    for x in xs:
        one.update(x)
    for i in range(0, len(xs), chunk):
        many.update_many(xs[i : i + chunk])
    assert one.n == many.n == len(xs)
    if xs:
        ref = np.asarray(xs)
        assert math.isclose(many.mean, float(ref.mean()), rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(one.mean, many.mean, rel_tol=1e-9, abs_tol=1e-6)
    if len(xs) > 1:
        ref_var = float(np.var(xs, ddof=1))
        assert math.isclose(many.variance, ref_var, rel_tol=1e-7, abs_tol=1e-3)
        assert math.isclose(one.variance, ref_var, rel_tol=1e-7, abs_tol=1e-3)


def test_empty_moments_are_nan() -> None:
    m = RunningMoments()
    assert math.isnan(m.mean)
    assert math.isnan(m.variance)
    assert math.isnan(m.population_variance)
    m.update_many([])
    assert m.n == 0
