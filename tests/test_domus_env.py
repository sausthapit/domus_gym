import numpy as np
from domus_gym.envs import DomusEnv
from domus_gym.envs.domus_env import BLOWER_MIN
from pytest import approx

from domus_mlsim import (
    SimpleHvac,
    hcm_reduced,
    estimate_cabin_temperature_dv1,
    KELVIN,
    DV1_XT_COLUMNS,
    HVAC_UT_COLUMNS,
)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


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

    s1, rew, done, info = env.step(a)
    assert s1 is not None
    assert env.observation_space.contains(s1)
    assert isinstance(done, bool), f"done is of type {type(done)}"
    # check that all keywords are included in info
    assert "comfort" in info
    assert "energy" in info
    assert "safety" in info
    assert isinstance(rew, float)

    ctrl = SimpleHvac()
    s = env.reset()
    for _ in range(100):
        a = ctrl.step(env.obs_tr.inverse_transform(s))

        #        print(f"a={a}")
        # convert an continuous control value into a discrete one
        act = [find_nearest_idx(ag, value) for ag, value in zip(env.action_grid, a)]

        assert env.action_space.contains(act)
        s, rew, done, info = env.step(act)
        if not done:
            assert env.observation_space.contains(s)
        else:
            s = env.reset()
    #       print(f"s={s}")
    assert -1 <= s[0] <= 1
    # temperature should have decreased
    assert s[1] < 305


def test_domus_env_init():
    env = DomusEnv(use_random_scenario=True)
    env.seed(1)
    obs = env.reset()
    same_count = 0
    N = 1000
    for _ in range(N):
        test_obs = env.reset()
        if (test_obs == obs).all():
            same_count += 1
    assert same_count / N == approx(1 / 30, abs=0.08)


def test_domus_env_specific_scenario():
    env = DomusEnv(use_scenario=1)
    obs = env.reset()
    state = env.obs_tr.inverse_transform(obs)
    # 1,272.15,0.99,272.15,0.99,0,0,0,100,FALSE,FALSE,FALSE,0.55
    assert state == approx(
        np.array(
            [
                0.99,
                272.15,
                22 + KELVIN,
                272.15,
                272.15,
            ]
        )
    )


def get_episode_len(env):
    s = env.reset()
    done = False
    ep_len = 0
    while not done:
        a = np.array([BLOWER_MIN, 0, 0, 0, 0, 0])
        act = [find_nearest_idx(ag, value) for ag, value in zip(env.action_grid, a)]
        assert env.action_space.contains(act)
        s, rew, done, info = env.step(act)
        ep_len += 1
    return ep_len


# long running test commented out for the moment
def test_seed():
    env = DomusEnv()
    env.seed(1)
    first_ep_len = get_episode_len(env)
    assert env.episode_length <= first_ep_len <= env.episode_length + 1
    env.seed(1)
    assert first_ep_len == get_episode_len(env)


def test_fixed_episode_len():
    env = DomusEnv(fixed_episode_length=10)
    _ = env.reset()
    assert env.episode_length == 10
    assert 10 == get_episode_len(env)


def partial_kw_to_array(columns, **kwargs):
    A = np.zeros(len(columns))
    for k, v in kwargs.items():
        A[columns.index(k)] = v
    return A


def make_b_x(air_temp, rh, ws):
    return partial_kw_to_array(
        DV1_XT_COLUMNS,
        t_drvr1=air_temp,
        t_drvr2=air_temp,
        t_drvr3=air_temp,
        t_psgr1=air_temp,
        t_psgr2=air_temp,
        t_psgr3=air_temp,
        m_drvr1=air_temp,
        m_drvr2=air_temp,
        m_drvr3=air_temp,
        m_psgr1=air_temp,
        m_psgr2=air_temp,
        m_psgr3=air_temp,
        rhc=rh,
        ws=ws,
    )


def test_hcm():
    env = DomusEnv()
    t = KELVIN + 22
    body_state = np.array(
        [
            [
                t - KELVIN,
                t - KELVIN,
                0,
            ],
            [
                t - KELVIN,
                t - KELVIN,
                0,
            ],
            [
                t - KELVIN,
                t - KELVIN,
                0,
            ],
        ]
    )

    assert (
        hcm_reduced(
            model=env.hcm_model,
            pre_out=22,
            body_state=body_state,
            rh=50,
        )
        == 1
    )


def test_comfort():
    env = DomusEnv()

    _ = env.reset()
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        ambient=KELVIN + 37,
    )
    assert env._comfort(env.b_x, h_u) == 0

    # should be comfortable
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        ambient=KELVIN + 22,
    )
    b_x = make_b_x(KELVIN + 22, 0.5, KELVIN + 22)

    assert env._comfort(b_x, h_u) == 1

    # winter temperature range
    ambient = 10
    for temp in range(12, 37):
        h_u = partial_kw_to_array(
            HVAC_UT_COLUMNS,
            ambient=KELVIN + ambient,
        )
        b_x = make_b_x(KELVIN + temp, 0.5, KELVIN + ambient)

        assert env._comfort(b_x, h_u) == (18 <= temp <= 27.5)

    # summer temperature range
    ambient = 30
    for temp in range(12, 37):
        h_u = partial_kw_to_array(
            HVAC_UT_COLUMNS,
            ambient=KELVIN + ambient,
        )
        b_x = make_b_x(KELVIN + temp, 0.5, KELVIN + ambient)

        assert env._comfort(b_x, h_u) == (20 <= temp <= 30)


def test_energy():
    env = DomusEnv()
    # max values
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        blw_power=17 * 18 + 94,
        cmp_power=3000,
        fan_power=400,
        hv_heater=6000,
    )
    assert env._energy(h_u) == approx(1)
    # zeros
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        blw_power=17 * 5 + 94,
    )
    assert env._energy(h_u) == approx(0)


def test_safety():
    env = DomusEnv()

    _ = env.reset()
    cab_t = estimate_cabin_temperature_dv1(env.b_x)
    assert env._safety(env.b_x, cab_t) == 1

    b_x = make_b_x(KELVIN + 22, 0.9, KELVIN + 2)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    assert env._safety(b_x, cab_t) == 0

    # tdp = T - (100 - rh) / 5
    # tdp = 10 - (100 - 80) / 5 = 10 - 4 = 6
    b_x = make_b_x(KELVIN + 10, 0.8, KELVIN + 8)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    assert cab_t == approx(KELVIN + 10)
    assert env._safety(b_x, cab_t) == 0
    b_x = make_b_x(KELVIN + 10, 0.8, KELVIN + 9.5)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    assert env._safety(b_x, cab_t) == approx(0.5)
    b_x = make_b_x(KELVIN + 10, 0.8, KELVIN + 11)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    assert env._safety(b_x, cab_t) == approx(1)


def test_reward():
    env = DomusEnv()

    _ = env.reset()
    # max energy, safe, comfort
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        ambient=KELVIN + 22,
        blw_power=17 * 18 + 94,
        cmp_power=3000,
        fan_power=400,
        hv_heater=6000,
    )
    b_x = make_b_x(KELVIN + 22, 0.5, KELVIN + 22)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    r, c, e, s = env._reward(b_x, h_u, cab_t)
    assert r == approx(0.523 * 1 - 0.477 * 1 + 2 * (1 - 1))
    assert c == approx(1)
    assert e == approx(1)
    assert s == approx(1)

    # max energy, not safe, comfort
    b_x = make_b_x(KELVIN + 22, 0.9, KELVIN + 2)
    r, c, e, s = env._reward(b_x, h_u, cab_t)
    assert r == approx(0.523 * 1 - 0.477 * 1 + 2 * (0 - 1))
    assert c == approx(1)
    assert e == approx(1)
    assert s == approx(0)

    # min energy, safe, comfort
    h_u = partial_kw_to_array(
        HVAC_UT_COLUMNS,
        ambient=KELVIN + 22,
        blw_power=17 * 5 + 94,
        cmp_power=0,
        fan_power=0,
        hv_heater=0,
    )
    b_x = make_b_x(KELVIN + 22, 0.5, KELVIN + 22)
    env.last_cab_t = KELVIN + 22
    r, c, e, s = env._reward(b_x, h_u, cab_t)
    assert r == approx(0.523 * 1 - 0.477 * 0 + 2 * (1 - 1))
    assert c == approx(1)
    assert e == approx(0)
    assert s == approx(1)

    assert env._phi(18 + KELVIN) == approx(-4)
    assert env._reward_shaped(22 + KELVIN, 18 + KELVIN) == approx(4)

    # min energy, safe, not comfort
    # also check reward shaping for 18 -> 22
    env.last_cab_t = KELVIN + 18
    b_x = make_b_x(KELVIN + 17, 0.5, KELVIN + 22)
    cab_t = estimate_cabin_temperature_dv1(b_x)
    r, c, e, s = env._reward(b_x, h_u, cab_t)
    assert r == approx(0.523 * 0 - 0.477 * 0 + 2 * (1 - 1) + 0.1 * (0.99 * -5 + 4))
    assert c == approx(0)
    assert e == approx(0)
    assert s == approx(1)


def test_last_cab_t():
    env = DomusEnv()
    env.reset()
    assert env.last_cab_t is None

    env = DomusEnv(use_random_scenario=True)
    env.reset()
    assert env.last_cab_t is None

    env = DomusEnv(use_scenario=1)
    env.reset()
    assert env.last_cab_t is None


def test_configured_passengers():
    env = DomusEnv(use_scenario=1)
    env.reset()
    assert env.configured_passengers == [0]
