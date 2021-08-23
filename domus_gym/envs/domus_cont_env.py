""" Domus Continuous Action

"""

import numpy as np
from gym import spaces  # error, spaces, utils

from .domus_env import BLOWER_ADD, BLOWER_MULT, DomusEnv
from .minmax import MinMaxTransform


class DomusContEnv(DomusEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        **kwargs,
    ):
        """Description:
            Simulation of the thermal environment of a Fiat 500e car
            cabin.

        This modifies DomusEnv by making the action_space continuous

        """
        super(DomusContEnv, self).__init__(**kwargs)
        act_min = np.array(
            [
                5 * BLOWER_MULT + BLOWER_ADD,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        act_max = np.array(
            [
                18 * BLOWER_MULT + BLOWER_ADD,
                3000,
                6000,
                400,
                1,
                1,
            ]
        )
        self.act_tr = MinMaxTransform(act_min, act_max)
        self.action_space = spaces.Box(high=1, low=-1)

    def _convert_action(self, action: np.ndarray):
        """given some action, convert it first into the controller state
        SimpleHvac.Xt and then into control inputs to the cabin and
        hvac.

        action is a ndarray

        """
        assert self.action_space.contains(
            action
        ), f"action {action} is not in the action_space {self.action_space}"

        c_x = self.act_tr.inverse_transform(action)

        return c_x
