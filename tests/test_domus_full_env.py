import numpy as np
from pytest import approx

from domus_gym.envs import Config, DomusFullEnv


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
        a = env.act_tr.transform(a)
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)

        if not done:
            print(s)
            assert env.observation_space.contains(s)


def test_new_air():
    env = DomusFullEnv(use_scenario=1)

    s = env.reset()
    # try new air mode 1 =>
    for mode in range(1, len(DomusFullEnv.NewAirMode)):

        a = np.array([mode, 1, 1, 1, 1, 0, 0, 1, 400, 0, 0, 1, 1, 6000])
        a = env.act_tr.transform(a)
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)


def test_seat():
    env = DomusFullEnv(use_scenario=1)

    s = env.reset()
    # try new air mode 1 =>
    for mode in range(1, len(DomusFullEnv.Seat)):

        a = np.array([0, 1, 1, 1, 1, mode, 0, 1, 400, 0, 0, 1, 1, 6000])
        a = env.act_tr.transform(a)
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)


def test_smart_vent():
    env = DomusFullEnv(use_scenario=1)

    s = env.reset()
    # try new air mode 1 =>
    for mode in range(1, len(DomusFullEnv.SmartVent)):

        a = np.array([0, 1, 1, 1, 1, 0, mode, 1, 400, 0, 0, 1, 1, 6000])
        a = env.act_tr.transform(a)
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)


def test_convert_action():
    env = DomusFullEnv(use_random_scenario=True)
    env.seed(1)
    for _ in range(100):
        _ = env._convert_action(env.action_space.sample())


def test_config():
    # check that when we turn off new air mode, changing new air mode has no effect
    env = DomusFullEnv(configuration=Config.seat + Config.windowheating, use_scenario=1)
    s_mode = []
    for mode in range(len(DomusFullEnv.NewAirMode)):
        s = env.reset()
        a = np.array([mode, 1, 1, 1, 1, 0, 0, 1, 400, 0, 0, 1, 1, 6000])
        a = env.act_tr.transform(a)
        assert env.action_space.contains(a)
        s, rew, done, info = env.step(a)
        s_mode.append(s)

    for mode in range(1, len(DomusFullEnv.NewAirMode)):
        assert s_mode[0] == approx(s_mode[mode])
