import numpy as np

from domus_gym.envs import DomusFullEnv


def test_full_env():
    env = DomusFullEnv(use_scenario=1)

    s = env.observation_space.sample()
    assert s is not None
    a = env.action_space.sample()
    assert a is not None

    s = env.reset()
    done = False
    while not done:
        # transform full env state to ctrl state

        # max heating and radiant panels
        a = np.array([0, 1, 1, 1, 1, 0, 0, 1, 400, 0, 0, 1, 1, 6000])
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)

        if not done:
            print(s)
            assert env.observation_space.contains(s)
