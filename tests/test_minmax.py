from domus_gym.envs.minmax import MinMaxTransform

import numpy as np
from numpy.testing import assert_array_equal

from pytest import approx


def test_minmaxtransform():
    obs_min = np.array([100, 0, -100])
    obs_max = np.array([120, 2, 100])

    tr = MinMaxTransform(obs_min, obs_max)

    v = np.array([110, 0.1, 0])
    assert v == approx(tr.inverse_transform(tr.transform(v)))
    t = tr.transform(v)
    assert (t >= -np.ones(obs_min.shape)).all()
    assert (t <= np.ones(obs_min.shape)).all()
    assert tr.transform(obs_min) == approx(np.array([-1, -1, -1]))
    assert tr.transform(obs_max) == approx(np.array([1, 1, 1]))
    assert tr.transform(obs_min) == approx(-1)
    assert tr.transform(obs_max) == approx(1)
