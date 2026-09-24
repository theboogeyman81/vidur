import numpy as np
import pytest

from evals.metrics.latency import percentile


@pytest.mark.parametrize("p", [0, 50, 95, 100])
def test_matches_numpy(p):
    xs = [310.0, 120.5, 980.0, 450.2, 610.0, 275.0, 1300.0]
    assert percentile(xs, p) == pytest.approx(float(np.percentile(xs, p)))


def test_single_value():
    assert percentile([42.0], 95) == 42.0


def test_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)
