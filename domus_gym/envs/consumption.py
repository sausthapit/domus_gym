"""  -*- coding: utf-8 -*-

consumption

Estimate power consumption due to weight of vehicle and speed of travel. Estimate maximum range.

Revised to use module level variable for singleton as per https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

Author
------
Sebastian Moeller

Date
----
October 17, 2021
"""
import pickle

import pkg_resources
import scipy.interpolate as sint


def _load_sc():
    filename = pkg_resources.resource_filename(__name__, "consumption.pickle")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    cons = data["consumption_kwh_100km"]
    speeds = data["speeds_kmh"]
    masses = data["masses_kg"]
    return sint.interp2d(speeds, masses, cons)


sc = _load_sc()


def spec_consumption_delta(speed, mass, deltaMass):
    c1 = sc(speed, mass)[0]
    c2 = sc(speed, mass + deltaMass)[0]
    return c2 - c1


def spec_consumption(speed, mass):  # Specific Consumption kWh/100km
    return sc(speed, mass)[0]


def power(speed, mass):  # watts
    return sc(speed, mass)[0] * speed * 10.0


def power_delta(speed, mass, deltaMass):  # watts
    c1 = sc(speed, mass)[0]
    c2 = sc(speed, mass + deltaMass)[0]
    return (c2 - c1) * speed * 10.0


def max_range(speed, mass, comfortPower):  # maximum range km, comfortPower in W
    p = comfortPower + power(speed, mass)  # W
    return speed * 24.0 * 1000.0 / p
