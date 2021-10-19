from pytest import approx

from domus_gym.envs.acoustics import calc_sound_level


def test_calc_sound_level():

    assert calc_sound_level(0, 0) <= calc_sound_level(0, 1)

    assert calc_sound_level(0, 0) <= calc_sound_level(1, 0)

    assert calc_sound_level(0, 0) < calc_sound_level(100, 6000)

    assert calc_sound_level(0, 0) == approx((0, 0, 0), abs=1e-7)

    assert calc_sound_level(52, 196) == approx((36.9, 35, 23), rel=0.05)

    assert calc_sound_level(68, 400) == approx((50.2, 40, 47), rel=0.05)
