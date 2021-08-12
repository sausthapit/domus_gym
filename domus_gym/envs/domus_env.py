import gym
from gym import spaces  # error, spaces, utils

from .minmax import MinMaxTransform

from gym.utils import seeding
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

import numpy as np
from domus_mlsim import (
    DV1Ut,
    DV1Xt,
    DV1_XT_COLUMNS,
    HVAC_XT_COLUMNS,
    HvacUt,
    KELVIN,
    estimate_cabin_temperature_dv1,
    kw_to_array,
    hcm_reduced,
    load_hcm_model,
    load_dv1,
    load_hvac,
    make_dv1_sim,
    make_hvac_sim,
    SimpleHvac,
    update_control_inputs_dv1,
    update_dv1_inputs,
    update_hvac_inputs,
)

COMFORT_WEIGHT = 0.523
ENERGY_WEIGHT = -0.477

BLOWER_MULT = 17
BLOWER_ADD = 94

COMPRESSOR_MAX = 3000
HV_HEATER_MAX = 6000
FAN_MAX = 400
BLOWER_MIN = 5 * BLOWER_MULT + BLOWER_ADD
BLOWER_MAX = 18 * BLOWER_MULT + BLOWER_ADD

ENERGY_MIN = BLOWER_MIN
ENERGY_MAX = BLOWER_MAX + COMPRESSOR_MAX + HV_HEATER_MAX + FAN_MAX

DEWPOINT_LOWER = 2
DEWPOINT_UPPER = 5

# exponential distribution for 23 minute mean episode length
MEAN_EPISODE_MINS = 23
DONE_PROB = 1 - np.exp(-1 / (MEAN_EPISODE_MINS * 60))

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

          blower_level 5, 10, or 18
          compressor_power 0 -> 3000
          hv_heater 0 -> 6000
          fan_power 0 -> 200??
          recirc to 0, 0.5, or 1
          window_heating 0 or 1

        Reward

        + The reward function is provided by the AF as

         :math: f_n(t) = 0.523 c(t) - 0.477e_n(t) + 2 ( s(t) - 1 ),

        where $t$ is time, $f_n$ is the normalised fitness function,
        $c$ is the comfort, $e_n$ is the normalised energy use, $s$ is
        the safety index.

        + The energy $e(t)$ is the sum of component energies from:
        1. HVAC including:
         a. PTC heater
         b. AC compressor
         c. AC fan
         d. blower
        2. radiant panels (1--4), if available
        3. heated seats, if available
        4. windshield heating, if available

        Starting state:

        Episode termination:

          randomly terminates with probability p so that mean episode
          length is 23 minutes.

        """
        super(DomusEnv, self).__init__()
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
                KELVIN + 80,
                KELVIN + 60,
            ],
            dtype=np.float32,
        )

        self.obs_tr = MinMaxTransform(obs_min, obs_max)

        self.observation_space = spaces.Box(
            high=1, low=0, shape=obs_min.shape, dtype=np.float32
        )
        self.action_grid = [
            # blower_level 5, 10, or 18
            np.array([5, 10, 18]) * BLOWER_MULT + BLOWER_ADD,
            # compressor_power 0 -> 3000 (0 or 3000)
            np.linspace(0, 3000, 2),
            # hv_heater 0 -> 6000 (0 or 3000 or 6000)
            np.linspace(0, 6000, 3),
            # fan_power 0 -> 400
            np.linspace(0, 400, 3),
            # recirc
            np.linspace(0, 1, 3),
            # window_heating 0 or 1
            np.linspace(0, 1, 2),
        ]
        self.action_space = spaces.MultiDiscrete([len(x) for x in self.action_grid])

        self.dv1_scaler_and_model = load_dv1()
        self.hvac_scaler_and_model = load_hvac()
        self.setpoint = 22 + KELVIN
        _, _, ldamdl, scale = load_hcm_model()
        self.hcm_model = (ldamdl, scale)
        self.configured_passengers = [0, 1]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _convert_state(self):
        """given the current state, create a vector that can be used as input to the controller"""
        c_u = np.zeros((len(SimpleHvac.Ut)))
        c_u[SimpleHvac.Ut.setpoint] = self.setpoint
        cab_t = estimate_cabin_temperature_dv1(self.b_x)
        update_control_inputs_dv1(c_u, self.b_x, self.h_x, cab_t)
        return self.obs_tr.transform(c_u)

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

    def _body_state(self, b_x, n):
        """return the body state matrix for passenger n where 0 is the driver, etc"""
        if n == 0:
            v = b_x[
                [
                    DV1Xt.t_drvr1,
                    DV1Xt.m_drvr1,
                    DV1Xt.v_drvr1,
                    DV1Xt.t_drvr2,
                    DV1Xt.m_drvr2,
                    DV1Xt.v_drvr2,
                    DV1Xt.t_drvr3,
                    DV1Xt.m_drvr3,
                    DV1Xt.v_drvr3,
                ]
            ]
        elif n == 1:
            v = b_x[
                [
                    DV1Xt.t_psgr1,
                    DV1Xt.m_psgr1,
                    DV1Xt.v_psgr1,
                    DV1Xt.t_psgr2,
                    DV1Xt.m_psgr2,
                    DV1Xt.v_psgr2,
                    DV1Xt.t_psgr3,
                    DV1Xt.m_psgr3,
                    DV1Xt.v_psgr3,
                ]
            ]
        elif n == 2:
            v = b_x[
                [
                    DV1Xt.t_psgr21,
                    DV1Xt.m_psgr21,
                    DV1Xt.v_psgr21,
                    DV1Xt.t_psgr22,
                    DV1Xt.m_psgr22,
                    DV1Xt.v_psgr22,
                    DV1Xt.t_psgr23,
                    DV1Xt.m_psgr23,
                    DV1Xt.v_psgr23,
                ]
            ]
        elif n == 3:
            v = b_x[
                [
                    DV1Xt.t_psgr31,
                    DV1Xt.m_psgr31,
                    DV1Xt.v_psgr31,
                    DV1Xt.t_psgr32,
                    DV1Xt.m_psgr32,
                    DV1Xt.v_psgr32,
                    DV1Xt.t_psgr33,
                    DV1Xt.m_psgr33,
                    DV1Xt.v_psgr33,
                ]
            ]
        # hcm uses celsius not kelvin
        v = v - np.array([KELVIN, KELVIN, 0, KELVIN, KELVIN, 0, KELVIN, KELVIN, 0])
        return v.reshape((3, 3))

    def _comfort(self, b_x, h_u):
        # temporarily just assess driver and front passenger comfort

        # assess driver comfort
        hcm = [
            hcm_reduced(
                model=self.hcm_model,
                pre_out=h_u[HvacUt.ambient] - KELVIN,
                body_state=self._body_state(b_x, i),
                rh=b_x[DV1Xt.rhc] * 100,
            )
            for i in self.configured_passengers
        ]
        return np.mean(hcm)

    def _normalise_energy(self, energy):
        """normalise energy value to be between 0 and 1"""
        return (energy - ENERGY_MIN) / (ENERGY_MAX - ENERGY_MIN)

    def _energy(self, h_u):
        energy = np.sum(
            h_u[
                [HvacUt.blw_power, HvacUt.cmp_power, HvacUt.fan_power, HvacUt.hv_heater]
            ]
        )
        # TODO window heating, radiant panels, heated seats

        return self._normalise_energy(energy)

    def _safety(self, b_x, cab_t):
        """safety is defined based on window fogging. This is estimated from
        the windshield temperature and relative humidity. The relative
        humidity yields a dew-point temperature

        """
        cabin_temperature = cab_t
        cabin_humidity = b_x[DV1Xt.rhc]
        windshield_temperature = b_x[DV1Xt.ws]
        # use simple dewpoint calculation given on wikipedia
        # https://en.wikipedia.org/wiki/Dew_point#Simple_approximation
        dewpoint_temperature = cabin_temperature - (1 - cabin_humidity) * 20
        delta_t = windshield_temperature - dewpoint_temperature
        if delta_t < DEWPOINT_LOWER:
            return 0
        elif delta_t > DEWPOINT_UPPER:
            return 1
        else:
            return (delta_t - DEWPOINT_LOWER) / (DEWPOINT_UPPER - DEWPOINT_LOWER)

    def _reward(self, b_x, h_u, cab_t):
        """fitness function based on state

        according to the domus d1.2 assessment framework, the fitness
        function is based on the energy, comfort and safety

        :math:f_n(t) = 0.523 c(t) - 0.477e_n(t) + 2 ( s(t) - 1 ),

        """

        return (
            COMFORT_WEIGHT * self._comfort(b_x, h_u)
            + ENERGY_WEIGHT * self._energy(h_u)
            + 2 * (self._safety(b_x, cab_t) - 1)
        )

    def _isdone(self):
        """by default, the test returns a numpy bool rather than a python
        bool, which upsets check_env"""
        return bool(self.np_random.uniform() < DONE_PROB)

    def step(self, action: np.ndarray) -> GymStepReturn:

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

        return (
            self._convert_state(),
            self._reward(self.b_x, h_u, cab_t),
            self._isdone(),
            {},
        )

    def reset(self) -> GymObs:

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
