import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from domus_mlsim import (
    DV1Ut,
    DV1Xt,
    DV1_XT_COLUMNS,
    HVAC_XT_COLUMNS,
    HvacUt,
    HvacXt,
    KELVIN,
    estimate_cabin_temperature_dv1,
    kw_to_array,
    load_dv1,
    load_hvac,
    make_dv1_sim,
    make_hvac_sim,
    SimpleHvac,
    update_control_inputs_dv1,
    update_dv1_inputs,
    update_hvac_inputs,
)

# 1. import domus_mlsim harness
# 2. initially - set b_x / h_x to a hot starting environment
# 3.


class DomusEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Description:
            Simulation of the thermal environment of a Fiat 500e car
            cabin.

        Observation:

            The environment is not fully observable. The temperature
            of the passengers is summarised.

            [
                cabin_humidity,
                cabin_temperature,
                setpoint,
                vent_temperature,
                window_temperature
            ]

        Actions:

          recirc to 0, 0.5, or 1
          defrost to 0, or 1
          blower_level 5, 10, or 18
          window_heating 0 or 1
          compressor_power 0 -> 3000
          hv_heater 0 -> 6000
          fan_power 0 -> 200??

        Reward

          TODO Based on fitness function

        Starting state:

        Episode termination:

          randomly terminates with probability p so that mean episode
          length is 23 minutes.

        """
        obs_min = np.array(
            [
                0,
                KELVIN - 20,
                KELVIN + 15,
                KELVIN + 0,
                KELVIN - 20,
            ],
            dtype=np.float32,
        )
        obs_max = np.array(
            [
                1,
                KELVIN + 60,
                KELVIN + 28,
                KELVIN + 60,
                KELVIN + 60,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
        self.action_grid = [
            # recirc
            np.linspace(0, 1, 3),
            # defrost
            np.linspace(0, 1, 2),
            # blower_level 5, 10, or 18
            np.array([5, 10, 18]) * 17 + 94,
            # window_heating 0 or 1
            np.linspace(0, 1, 2),
            # compressor_power 0 -> 3000 (0 or 3000)
            np.linspace(0, 3000, 2),
            # hv_heater 0 -> 6000 (0 or 3000 or 6000)
            np.linspace(0, 6000, 3),
            # fan_power 0 -> 400
            np.linspace(0, 400, 3),
        ]
        self.action_space = spaces.MultiDiscrete([len(x) for x in self.action_grid])

        self.dv1_scaler_and_model = load_dv1()
        self.hvac_scaler_and_model = load_hvac()
        self.setpoint = 22 + KELVIN

    def _convert_state(self):
        """given the current state, create a vector that can be used as input to the controller"""
        c_u = np.zeros((len(SimpleHvac.Ut)))
        c_u[SimpleHvac.Ut.setpoint] = self.setpoint
        cab_t = estimate_cabin_temperature_dv1(self.b_x)
        update_control_inputs_dv1(c_u, self.b_x, self.h_x, cab_t)
        return c_u

    def _convert_action(self, action):
        """given some action, convert it first into the controller state
        SimpleHvac.Xt and then into control inputs to the cabin and
        hvac.

        action is a MultiDiscrete space and thus an array of integers.

        """
        assert self.action_space.contains(
            action
        ), f"action {action} is not in the action_space {self.action_space}"

        assert len(action) == len(
            self.action_grid
        ), f"wrong number of elements in action vector {action}"
        c_x = np.array([ag[i] for ag, i in zip(self.action_grid, action)])

        return c_x

    def step(self, action):

        c_x = self._convert_action(action)
        cab_t = estimate_cabin_temperature_dv1(self.b_x)

        # TODO pre-create h_u and b_u in init
        h_u = np.zeros((len(HvacUt)))
        h_u[[HvacUt.ambient, HvacUt.humidity, HvacUt.solar, HvacUt.speed]] = [
            self.ambient_t,
            self.ambient_rh,
            self.solar1,
            self.car_speed,
        ]
        update_hvac_inputs(h_u, c_x, cab_t)
        _, self.h_x = self.hvac_sim.step(h_u)

        b_u = np.zeros((len(DV1Ut)))
        b_u[[DV1Ut.t_a, DV1Ut.rh_a, DV1Ut.rad1, DV1Ut.rad2, DV1Ut.VehicleSpeed,]] = [
            self.ambient_t,
            self.ambient_rh,
            self.solar1,
            self.solar2,
            self.car_speed / 100 * 27.778,
        ]
        update_dv1_inputs(b_u, self.h_x, c_x)
        _, self.b_x = self.dv1_sim.step(b_u)

        return self._convert_state()

    def reset(self):

        # create a new state vector for the cabin and hvac
        self.ambient_t = KELVIN + 37
        self.ambient_rh = 0.5
        self.cabin_t = KELVIN + 37
        self.cabin_v = 0
        self.cabin_rh = 0.5
        self.solar1 = 200
        self.solar2 = 100
        self.car_speed = 50

        self.b_x = kw_to_array(
            DV1_XT_COLUMNS,
            t_drvr1=self.cabin_t,
            t_drvr2=self.cabin_t,
            t_drvr3=self.cabin_t,
            t_psgr1=self.cabin_t,
            t_psgr2=self.cabin_t,
            t_psgr3=self.cabin_t,
            t_psgr21=self.cabin_t,
            t_psgr22=self.cabin_t,
            t_psgr23=self.cabin_t,
            t_psgr31=self.cabin_t,
            t_psgr32=self.cabin_t,
            t_psgr33=self.cabin_t,
            v_drvr1=self.cabin_v,
            v_drvr2=self.cabin_v,
            v_drvr3=self.cabin_v,
            v_psgr1=self.cabin_v,
            v_psgr2=self.cabin_v,
            v_psgr3=self.cabin_v,
            v_psgr21=self.cabin_v,
            v_psgr22=self.cabin_v,
            v_psgr23=self.cabin_v,
            v_psgr31=self.cabin_v,
            v_psgr32=self.cabin_v,
            v_psgr33=self.cabin_v,
            m_drvr1=self.cabin_t,
            m_drvr2=self.cabin_t,
            m_drvr3=self.cabin_t,
            m_psgr1=self.cabin_t,
            m_psgr2=self.cabin_t,
            m_psgr3=self.cabin_t,
            m_psgr21=self.cabin_t,
            m_psgr22=self.cabin_t,
            m_psgr23=self.cabin_t,
            m_psgr31=self.cabin_t,
            m_psgr32=self.cabin_t,
            m_psgr33=self.cabin_t,
            rhc=self.cabin_rh,
            ws=self.cabin_t,
        )

        self.h_x = kw_to_array(
            HVAC_XT_COLUMNS,
            cab_RH=self.cabin_rh,
            evp_mdot=self.cabin_v,
            vent_T=self.cabin_t,
        )

        # create a new simulation for both dv1 and hvac
        self.dv1_sim = make_dv1_sim(self.dv1_scaler_and_model, self.b_x)
        self.hvac_sim = make_hvac_sim(self.hvac_scaler_and_model, self.h_x)

        # convert state to control input
        return self._convert_state()

        # cab_t = estimate_cabin_temperature_dv1(self.b_x)
        # c_u = np.zeros((len(SimpleHvac.Ut)))
        # c_u[SimpleHvac.Ut.setpoint] = self.setpoint
        # update_control_inputs_dv1(c_u, self.b_x, self.h_x, cab_t)
        # return c_u

    def render(self, mode="human"):
        pass

    def close(self):
        pass
