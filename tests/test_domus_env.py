from domus_gym.envs.domus_env import DomusEnv


def test_domus_env():
    env = DomusEnv()

    # can we sample the observation space?
    s = env.observation_space.sample()
    assert s is not None
    a = env.action_space.sample()
    assert a is not None

    s = env.reset()
    assert s is not None
    assert env.observation_space.contains(s)

    s1 = env.step(a)
    assert s1 is not None
    assert env.observation_space.contains(s1)
