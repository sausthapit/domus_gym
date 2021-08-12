from domus_gym.envs.minmax import MinMaxTransform

import numpy as np
from numpy.testing import assert_array_equal


def test_minmaxtransform():
    obs_min = np.array([100, 0, -100])
    obs_max = np.array([120, 2, 100])

    tr = MinMaxTransform(obs_min, obs_max)

    v = np.array([110, 0.1, 0])
    assert_array_equal(tr.inverse_transform(tr.transform(v)), v)
    t = tr.transform(v)
    assert (t >= np.zeros(obs_min.shape)).all()
    assert (t <= np.ones(obs_min.shape)).all()
