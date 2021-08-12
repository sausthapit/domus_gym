from domus_gym.envs import DomusContEnv

from domus_mlsim import (
    SimpleHvac,
)


def test_domus_cont_env():
    env = DomusContEnv()

    # can we sample the observation space?
    s = env.observation_space.sample()
    assert s is not None
    a = env.action_space.sample()
    assert a is not None

    ctrl = SimpleHvac()
    s = env.reset()
    for i in range(100):
        a = ctrl.step(env.obs_tr.inverse_transform(s))

        act = env.act_tr.transform(a)

        assert env.action_space.contains(act)
        s, rew, done, kw = env.step(act)
        if not done:
            assert env.observation_space.contains(s)
        else:
            s = env.reset()
