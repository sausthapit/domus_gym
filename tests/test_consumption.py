# test cases provided by Sebastian

from pytest import approx

from domus_gym.envs import consumption


def test_consumption():

    speed1 = 30  # kmh

    mass1 = 1280  # kg

    power1 = consumption.power(speed1, mass1)  # -- > 3933.089 Watts
    assert power1 == approx(3933.089)

    speed2 = 90  # kmh

    mass2 = 1500  # kg

    power2 = consumption.power(speed2, mass2)  # -- > 19419.774 Watts
    assert power2 == approx(19419.774)

    # maximum

    speed3 = 120  # kmh

    mass3 = 1600  # kg

    power3 = consumption.power(speed3, mass3)  # -- > 31392.927 Watts
    assert power3 == approx(31392.927)

    # minimum

    speed4 = 0  # kmh

    mass4 = 1000  # kg

    power4 = consumption.power(speed4, mass4)  # -- > 0 Watts
    assert power4 == 0

    # power delta

    speed5 = 50  # kmh

    mass5 = 1300  # kg

    deltaMass5 = 40  # kg

    powerDelta5 = consumption.power_delta(
        speed5, mass5, deltaMass5
    )  # -- > 81.815 Watts
    assert powerDelta5 == approx(81.815)

    # consumption

    speed6 = 50  # kmh

    mass6 = 1300  # kg

    cons6 = consumption.spec_consumption(speed6, mass6)  # -- > 15.421 kWh/100km
    assert cons6 == approx(15.421, abs=1e-3)
