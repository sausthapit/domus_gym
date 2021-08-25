""" Domus Environment with access to entire state and action space

"""

from enum import IntEnum

import numpy as np
from gym import spaces  # error, spaces, utils
from stable_baselines3.common.type_aliases import GymStepReturn

from domus_mlsim import (
    KELVIN,
    DV1Ut,
    DV1Xt,
    HvacUt,
    HvacXt,
    estimate_cabin_temperature_dv1,
)

from .domus_env import BLOWER_ADD, BLOWER_MULT, DomusEnv

KMH_TO_MS = 1000 / 3600


class DomusFullEnv(DomusEnv):
    metadata = {"render.modes": []}

    Action = IntEnum(
        "Action",
        [
            "new_air_mode",
            "radiant_panel_1",
            "radiant_panel_2",
            "radiant_panel_3",
            "radiant_panel_4",
            "seat",
            "smart_vent",
            "window_heating",
            "blw_power",
            "cmp_power",
            "fan_power",
            "recirc",
            "dist_defrost",
            "hv_heater",
        ],
        start=0,
    )

    InternalAction = IntEnum(
        "InternalAction",
        [
            "new_air_mode_Bi_Level_SO_Side_Low",
            "new_air_mode_Panel_Only_SO_Side_Low",
            "new_air_mode_Panel_Only_SO_Side_High",
            "new_air_mode_Panel_Only_SO_Middle_High",
            "new_air_mode_Defrost_SO_Defrost",
            "new_air_mode_Bi_Level_SO_Side_High",
            "new_air_mode_Floor_SO_Defrost",
            "new_air_mode_Floor_Defrost_SO_Defrost",
            "new_air_mode_Panel_Only_SO_Middle_Low",
            "new_air_mode_Bi_Level_SO_Middle_Low",
            "radiant_panel_1",
            "radiant_panel_2",
            "radiant_panel_3",
            "radiant_panel_4",
            "seat_off",
            "seat_ventilate",
            "smart_vent_diffuse_low",
            "window_heating",
            "blw_power",
            "cmp_power",
            "fan_power",
            "recirc",
            "dist_defrost",
            "hv_heater",
        ],
        start=0,
    )

    NewAirMode = IntEnum(
        "NewAirMode",
        [
            "default",
            "Bi_Level_SO_Side_Low",
            "Panel_Only_SO_Side_Low",
            "Panel_Only_SO_Side_High",
            "Panel_Only_SO_Middle_High",
            "Defrost_SO_Defrost",
            "Bi_Level_SO_Side_High",
            "Floor_SO_Defrost",
            "Floor_Defrost_SO_Defrost",
            "Panel_Only_SO_Middle_Low",
            "Bi_Level_SO_Middle_Low",
        ],
        start=0,
    )

    Seat = IntEnum(
        "Seat",
        [
            "default",
            "off",
            "ventilate",
        ],
        start=0,
    )

    SmartVent = IntEnum(
        "SmartVent",
        [
            "default",
            "diffuse_low",
        ],
        start=0,
    )

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

        This modifies DomusEnv by expanding the state and action space so that it has complete access to the simulator operation.

        """
        super(DomusFullEnv, self).__init__(**kwargs)
        act_min = np.array(
            [
                0,
            ]
            * 8
            + [
                5 * BLOWER_MULT + BLOWER_ADD,
            ]
            + [0] * 5
        )
        act_max = np.array(
            [
                len(self.NewAirMode),
            ]
            + [1] * 4
            + [
                len(self.Seat),
                len(self.SmartVent),
                1,
                18 * BLOWER_MULT + BLOWER_ADD,
                3000,
                400,
                1,
                1,
                6000,
            ]
        )
        obs_min = np.array(
            [KELVIN - 10] * 12 + [KELVIN - 10] * 12 + [0] * 12 + [KELVIN - 20, 0]
        )
        obs_max = np.array(
            [KELVIN + 60] * 12 + [KELVIN + 60] * 12 + [10] * 12 + [KELVIN + 60, 0]
        )
        self.action_space = spaces.Box(
            high=act_max, low=act_min, shape=act_min.shape, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            high=obs_max, low=obs_min, shape=obs_min.shape, dtype=np.float32
        )
        self.h_u = np.zeros((len(HvacUt)))
        self.b_u = np.zeros((len(DV1Ut)))

    def _convert_action(self, action: np.ndarray):
        """given some action in the original box space, convert it to an
        internal action vector

        action is a ndarray

        """
        assert self.action_space.contains(  # TODO move to test code
            action
        ), f"action {action} is not in the action_space {self.action_space}"
        rounded_action = np.around(action)
        iaction = np.zeros((len(self.InternalAction)))
        new_air_mode = rounded_action[self.Action.new_air_mode]
        # new_air_mode is 0 - n, translate this into one-hot
        if new_air_mode > 0:
            iaction[
                int(new_air_mode)
                + self.InternalAction.new_air_mode_Bi_Level_SO_Side_Low
                - 1
            ] = 1

        seat = rounded_action[self.Action.seat]
        if seat > 0:
            iaction[int(seat) + self.InternalAction.seat_off - 1] = 1

        iaction[
            [
                self.InternalAction.radiant_panel_1,
                self.InternalAction.radiant_panel_2,
                self.InternalAction.radiant_panel_3,
                self.InternalAction.radiant_panel_4,
                self.InternalAction.smart_vent_diffuse_low,
                self.InternalAction.window_heating,
                self.InternalAction.blw_power,
                self.InternalAction.cmp_power,
                self.InternalAction.fan_power,
                self.InternalAction.recirc,
                self.InternalAction.dist_defrost,
                self.InternalAction.hv_heater,
            ]
        ] = rounded_action[
            [
                self.Action.radiant_panel_1,
                self.Action.radiant_panel_2,
                self.Action.radiant_panel_3,
                self.Action.radiant_panel_4,
                self.Action.smart_vent,
                self.Action.window_heating,
                self.Action.blw_power,
                self.Action.cmp_power,
                self.Action.fan_power,
                self.Action.recirc,
                self.Action.dist_defrost,
                self.Action.hv_heater,
            ]
        ]

        return iaction

    def _convert_state(self):
        """given the current state, create a vector that can be used as input to the controller"""
        state = np.zeros((len(self.State)))
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

    def step(self, action: np.ndarray) -> GymStepReturn:
        self.episode_clock += 1
        int_action = self._convert_action(action)
        cab_t = estimate_cabin_temperature_dv1(self.b_x)

        # TODO only update ambient etc when it changes
        self.h_u[
            [HvacUt.ambient, HvacUt.humidity, HvacUt.solar, HvacUt.speed, HvacUt.cab_T]
        ] = [
            self.ambient_t,
            self.ambient_rh,
            self.solar1,
            self.car_speed,
            cab_t,
        ]
        self.h_u[
            [
                HvacUt.blw_power,
                HvacUt.cmp_power,
                HvacUt.fan_power,
                HvacUt.recirc,
                HvacUt.hv_heater,
            ]
        ] = int_action[
            [
                self.InternalAction.blw_power,
                self.InternalAction.cmp_power,
                self.InternalAction.fan_power,
                self.InternalAction.recirc,
                self.InternalAction.hv_heater,
            ]
        ]
        _, self.h_x = self.hvac_sim.step(self.h_u)

        self.b_u[
            [
                DV1Ut.recirc,
                DV1Ut.dist_defrost,
                DV1Ut.new_air_mode_Bi_Level_SO_Side_Low,
                DV1Ut.new_air_mode_Panel_Only_SO_Side_Low,
                DV1Ut.new_air_mode_Panel_Only_SO_Side_High,
                DV1Ut.new_air_mode_Panel_Only_SO_Middle_High,
                DV1Ut.new_air_mode_Defrost_SO_Defrost,
                DV1Ut.new_air_mode_Bi_Level_SO_Side_High,
                DV1Ut.new_air_mode_Floor_SO_Defrost,
                DV1Ut.new_air_mode_Floor_Defrost_SO_Defrost,
                DV1Ut.new_air_mode_Panel_Only_SO_Middle_Low,
                DV1Ut.new_air_mode_Bi_Level_SO_Middle_Low,
                DV1Ut.radiant_panel_1,
                DV1Ut.radiant_panel_2,
                DV1Ut.radiant_panel_3,
                DV1Ut.radiant_panel_4,
                DV1Ut.seat_off,
                DV1Ut.seat_ventilate,
                DV1Ut.smart_vent_diffuse_low,
                DV1Ut.window_heating,
            ]
        ] = int_action[
            [
                self.InternalAction.recirc,
                self.InternalAction.dist_defrost,
                self.InternalAction.new_air_mode_Bi_Level_SO_Side_Low,
                self.InternalAction.new_air_mode_Panel_Only_SO_Side_Low,
                self.InternalAction.new_air_mode_Panel_Only_SO_Side_High,
                self.InternalAction.new_air_mode_Panel_Only_SO_Middle_High,
                self.InternalAction.new_air_mode_Defrost_SO_Defrost,
                self.InternalAction.new_air_mode_Bi_Level_SO_Side_High,
                self.InternalAction.new_air_mode_Floor_SO_Defrost,
                self.InternalAction.new_air_mode_Floor_Defrost_SO_Defrost,
                self.InternalAction.new_air_mode_Panel_Only_SO_Middle_Low,
                self.InternalAction.new_air_mode_Bi_Level_SO_Middle_Low,
                self.InternalAction.radiant_panel_1,
                self.InternalAction.radiant_panel_2,
                self.InternalAction.radiant_panel_3,
                self.InternalAction.radiant_panel_4,
                self.InternalAction.seat_off,
                self.InternalAction.seat_ventilate,
                self.InternalAction.smart_vent_diffuse_low,
                self.InternalAction.window_heating,
            ]
        ]

        self.b_u[[DV1Ut.HvacMain,]] = self.h_x[
            [
                HvacXt.vent_T,
            ]
        ]
        # simplification to get dv1 working
        self.b_u[DV1Ut.vent_flow_rate] = np.interp(
            int_action[self.InternalAction.blw_power],
            np.array([5, 10, 18]) * 17 + 94,
            np.array([1, 3, 5]),
        )

        # TODO update only when it changes
        self.b_u[
            [
                DV1Ut.t_a,
                DV1Ut.rh_a,
                DV1Ut.rad1,
                DV1Ut.rad2,
                DV1Ut.VehicleSpeed,
            ]
        ] = [
            self.ambient_t,
            self.ambient_rh,
            self.solar1,
            self.solar2,
            self.car_speed * KMH_TO_MS,
        ]
        _, self.b_x = self.dv1_sim.step(self.b_u)

        rew, c, e, s = self._reward(self.b_x, self.h_u, cab_t)
        self.last_cab_t = cab_t
        return (
            self._convert_state(),
            rew,
            self._isdone(),
            {"comfort": c, "energy": e, "safety": s},
        )
