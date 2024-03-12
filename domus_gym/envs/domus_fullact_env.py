""" Domus Environment with access to action space but limited state space

"""

from enum import Enum, IntEnum, auto

import numpy as np
import gymnasium as gym
from gymnasium import spaces  # error, spaces, utils

from domus_mlsim import DV1Ut, HvacUt, HvacXt

from .domus_cont_env import DomusContEnv
from .domus_env import BLOWER_ADD, BLOWER_MULT
from .minmax import MinMaxTransform

KMH_TO_MS = 1000 / 3600


class Config(IntEnum):
    radiant = auto()
    seat = auto()
    windowheating = auto()
    newairmode = auto()


CONFIG_ALL = set([x for x in Config])


class DomusFullActEnv(DomusContEnv):
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

    def __init__(
        self,
        configuration=CONFIG_ALL,
        **kwargs,
    ):
        """Description:
            Simulation of the thermal environment of a Fiat 500e car
            cabin.

        This modifies DomusContEnv by expanding the action space.

        Parameters
        ----------

        configuration : set or string of 1s and 0s

          defaults to all options included

          use a set of Config enums to select a particular
          configuration. e.g., to turn on just the seat
          use `configuration=set([Config.seat])'

        """
        super(DomusFullActEnv, self).__init__(**kwargs)
        act_min = np.array(
            [
                0,
            ]
            * 7
            + [
                5 * BLOWER_MULT + BLOWER_ADD,
            ]
            + [0] * 5,
            dtype=np.float32,
        )
        act_max = np.array(
            [
                len(self.NewAirMode),
            ]
            + [1] * 4
            + [
                len(self.Seat),
                1,
                18 * BLOWER_MULT + BLOWER_ADD,
                3000,
                400,
                1,
                1,
                6000,
            ],
            dtype=np.float32,
        )
        self.act_tr = MinMaxTransform(act_min, act_max)
        self.action_space = spaces.Box(
            high=1, low=-1, shape=act_min.shape, dtype=np.float32
        )
        self._make_mask(configuration)

    def _make_mask(self, configuration):
        if isinstance(configuration, str):
            configuration = set(
                [cfg for cfg, p in zip(Config, configuration) if p == "1"]
            )
        self.mask = np.ones((len(self.Action),))
        if Config.radiant not in configuration:
            self.mask[self.Action.radiant_panel_1 : self.Action.radiant_panel_4 + 1] = 0
        if Config.seat not in configuration:
            self.mask[self.Action.seat] = 0
        if Config.windowheating not in configuration:
            self.mask[self.Action.window_heating] = 0
        if Config.newairmode not in configuration:
            self.mask[self.Action.new_air_mode] = 0

    def _mask_action(self, action: np.ndarray):
        return action * self.mask

    def _convert_action(self, action: np.ndarray):
        """given some action in the original box space, convert it to an
        internal action vector

        action is a ndarray

        """
        assert self.action_space.contains(  # TODO move to test code
            action
        ), f"action {action} is not in the action_space {self.action_space}"
        action = self.act_tr.inverse_transform(action)
        action = self._mask_action(action)
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
                self.InternalAction.window_heating,
                self.InternalAction.recirc,
                self.InternalAction.dist_defrost,
            ]
        ] = rounded_action[
            [
                self.Action.radiant_panel_1,
                self.Action.radiant_panel_2,
                self.Action.radiant_panel_3,
                self.Action.radiant_panel_4,
                self.Action.window_heating,
                self.Action.recirc,
                self.Action.dist_defrost,
            ]
        ]

        iaction[
            [
                self.InternalAction.blw_power,
                self.InternalAction.cmp_power,
                self.InternalAction.fan_power,
                self.InternalAction.hv_heater,
            ]
        ] = action[
            [
                self.Action.blw_power,
                self.Action.cmp_power,
                self.Action.fan_power,
                self.Action.hv_heater,
            ]
        ]

        return iaction

    def _step_hvac(self, int_action, cab_t):
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

    def _step_cabin(self, int_action):
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
                self.InternalAction.window_heating,
            ]
        ]

        self.b_u[
            [
                DV1Ut.HvacMain,
            ]
        ] = self.h_x[
            [
                HvacXt.vent_T,
            ]
        ]
        # simplification to get dv1 working
        self.b_u[DV1Ut.vent_flow_rate] = np.interp(
            int_action[self.InternalAction.blw_power],
            np.array([5, 10, 18], dtype=np.float32) * 17 + 94,
            np.array([1, 3, 5], dtype=np.float32),
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
