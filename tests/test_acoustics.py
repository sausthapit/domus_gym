from domus_gym.envs.acoustics import calc_sound_level


def test_calc_sound_level():

    assert calc_sound_level(0, 0) <= calc_sound_level(0, 1)

    assert calc_sound_level(0, 0) <= calc_sound_level(1, 0)

    assert calc_sound_level(0, 0) < calc_sound_level(100, 6000)
