""" Domus Environment with access to entire state and action space

"""

from enum import IntEnum

import numpy as np
import gymnasium as gym
from gymnasium import spaces  # error, spaces, utils

from domus_mlsim import KELVIN, DV1Xt, HvacXt

from .domus_fullact_env import DomusFullActEnv

KMH_TO_MS = 1000 / 3600


class DomusFullEnv(DomusFullActEnv):

    State = IntEnum(
        "State",
        [
            # state variables needed to estimate comfort
            "m_drvr1",
            "m_drvr2",
            "m_drvr3",
            "m_psgr1",
            "m_psgr2",
            "m_psgr21",
            "m_psgr22",
            "m_psgr23",
            "m_psgr3",
            "m_psgr31",
            "m_psgr32",
            "m_psgr33",
            # air temperature
            "t_drvr1",
            "t_drvr2",
            "t_drvr3",
            "t_psgr1",
            "t_psgr2",
            "t_psgr21",
            "t_psgr22",
            "t_psgr23",
            "t_psgr3",
            "t_psgr31",
            "t_psgr32",
            "t_psgr33",
            # air velocity
            "v_drvr1",
            "v_drvr2",
            "v_drvr3",
            "v_psgr1",
            "v_psgr2",
            "v_psgr21",
            "v_psgr22",
            "v_psgr23",
            "v_psgr3",
            "v_psgr31",
            "v_psgr32",
            "v_psgr33",
            # windshield temperature
            "ws",
            "cab_RH",
        ],
        start=0,
    )

    def __init__(
        self,
        **kwargs,
    ):
        """Description:
            Simulation of the thermal environment of a Fiat 500e car
            cabin.

        This modifies DomusFullActEnv by expanding the state space so
        that it has complete access to the simulator internals.

        """
        super(DomusFullEnv, self).__init__(**kwargs)
        obs_min = np.array(
            [KELVIN - 10] * 12 + [KELVIN - 10] * 12 + [0] * 12 + [KELVIN - 20, 0],
            dtype=np.float32,
        )
        obs_max = np.array(
            [KELVIN + 60] * 12 + [KELVIN + 60] * 12 + [10] * 12 + [KELVIN + 60, 0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            high=obs_max, low=obs_min, shape=obs_min.shape, dtype=np.float32
        )

    def _convert_state(self):
        """given the current state, create a vector that can be used as input to the controller"""
        state = np.zeros((len(self.State)), dtype=np.float32)
        state[
            [
                self.State.m_drvr1,
                self.State.m_drvr2,
                self.State.m_drvr3,
                self.State.m_psgr1,
                self.State.m_psgr2,
                self.State.m_psgr21,
                self.State.m_psgr22,
                self.State.m_psgr23,
                self.State.m_psgr3,
                self.State.m_psgr31,
                self.State.m_psgr32,
                self.State.m_psgr33,
                self.State.t_drvr1,
                self.State.t_drvr2,
                self.State.t_drvr3,
                self.State.t_psgr1,
                self.State.t_psgr2,
                self.State.t_psgr21,
                self.State.t_psgr22,
                self.State.t_psgr23,
                self.State.t_psgr3,
                self.State.t_psgr31,
                self.State.t_psgr32,
                self.State.t_psgr33,
                self.State.v_drvr1,
                self.State.v_drvr2,
                self.State.v_drvr3,
                self.State.v_psgr1,
                self.State.v_psgr2,
                self.State.v_psgr21,
                self.State.v_psgr22,
                self.State.v_psgr23,
                self.State.v_psgr3,
                self.State.v_psgr31,
                self.State.v_psgr32,
                self.State.v_psgr33,
                self.State.ws,
            ]
        ] = self.b_x[
            [
                DV1Xt.m_drvr1,
                DV1Xt.m_drvr2,
                DV1Xt.m_drvr3,
                DV1Xt.m_psgr1,
                DV1Xt.m_psgr2,
                DV1Xt.m_psgr21,
                DV1Xt.m_psgr22,
                DV1Xt.m_psgr23,
                DV1Xt.m_psgr3,
                DV1Xt.m_psgr31,
                DV1Xt.m_psgr32,
                DV1Xt.m_psgr33,
                DV1Xt.t_drvr1,
                DV1Xt.t_drvr2,
                DV1Xt.t_drvr3,
                DV1Xt.t_psgr1,
                DV1Xt.t_psgr2,
                DV1Xt.t_psgr21,
                DV1Xt.t_psgr22,
                DV1Xt.t_psgr23,
                DV1Xt.t_psgr3,
                DV1Xt.t_psgr31,
                DV1Xt.t_psgr32,
                DV1Xt.t_psgr33,
                DV1Xt.v_drvr1,
                DV1Xt.v_drvr2,
                DV1Xt.v_drvr3,
                DV1Xt.v_psgr1,
                DV1Xt.v_psgr2,
                DV1Xt.v_psgr21,
                DV1Xt.v_psgr22,
                DV1Xt.v_psgr23,
                DV1Xt.v_psgr3,
                DV1Xt.v_psgr31,
                DV1Xt.v_psgr32,
                DV1Xt.v_psgr33,
                DV1Xt.ws,
            ]
        ]
        state[self.State.cab_RH] = self.h_x[HvacXt.cab_RH]
        return np.clip(
            state, a_min=self.observation_space.low, a_max=self.observation_space.high
        )
