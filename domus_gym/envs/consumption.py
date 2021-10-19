"""  -*- coding: utf-8 -*-

consumption

Estimate power consumption due to weight of vehicle and speed of travel.

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


class Consumption:
    def __init__(
        self,
        filename=pkg_resources.resource_filename(__name__, "consumption.pickle"),
    ):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        cons = data["consumption_kwh_100km"]
        speeds = data["speeds_kmh"]
        masses = data["masses_kg"]
        self.sc = sint.interp2d(speeds, masses, cons)

    def spec_consumption_delta(self, speed, mass, deltaMass):
        c1 = self.sc(speed, mass)[0]
        c2 = self.sc(speed, mass + deltaMass)[0]
        return c2 - c1

    def spec_consumption(self, speed, mass):  # Specific Consumption
        return self.sc(speed, mass)[0]

    def power(self, speed, mass):
        return self.sc(speed, mass)[0] * speed * 10.0

    def power_delta(self, speed, mass, deltaMass):
        c1 = self.sc(speed, mass)[0]
        c2 = self.sc(speed, mass + deltaMass)[0]
        return (c2 - c1) * speed * 10.0
