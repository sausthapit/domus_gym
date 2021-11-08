from domus_gym.envs import DomusDv0ContEnv
from domus_mlsim import SimpleHvac


def test_domus_dv0_env():
    env = DomusDv0ContEnv()

    # can we sample the observation space?
    s = env.observation_space.sample()
    assert s is not None
    a = env.action_space.sample()
    assert a is not None

    ctrl = SimpleHvac()
    s = env.reset()
    for _ in range(100):
        a = ctrl.step(env.hvac_action(s))

        act = env.act_tr.transform(a)

        assert env.action_space.contains(act)
        s, rew, done, kw = env.step(act)
        if not done:
            assert env.observation_space.contains(s)
        else:
            s = env.reset()


def test_domus_dv0_args():
    env = DomusDv0ContEnv(use_random_scenario=True)
    assert env.use_random_scenario
    env = DomusDv0ContEnv(use_scenario=1)
    assert env.use_scenario == 1
    env = DomusDv0ContEnv(fixed_episode_length=100)
    assert env.fixed_episode_length == 100
