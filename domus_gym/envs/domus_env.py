import gym
from gym import error, spaces, utils
from gym.utils import seeding

import joblib
from domus_mlsim.harness import make_dv0_sim, make_dv1_sim
from domus_mlsim.cols import KELVIN

# 1. import domus_mlsim harness
# 2. initially - set b_x / h_x to a hot starting environment
# 3.

DV1_MODEL = ROOT / "model/dv1_lr.joblib"
HVAC_MODEL = ROOT / "model/hvac_lr.joblib"


class DomusEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        dv1_scaler, dv1_model = joblib.load(DV1_MODEL)
        hvac_scaler, hvac_model = joblib.load(HVAC_MODEL)

        cabin_t = KELVIN + 37
        cabin_v = 0
        cabin_rh = 0.5
        b_x = kw_to_array(
            DV1_XT_COLUMNS,
            t_drvr1=cabin_t,
            t_drvr2=cabin_t,
            t_drvr3=cabin_t,
            t_psgr1=cabin_t,
            t_psgr2=cabin_t,
            t_psgr3=cabin_t,
            t_psgr21=cabin_t,
            t_psgr22=cabin_t,
            t_psgr23=cabin_t,
            t_psgr31=cabin_t,
            t_psgr32=cabin_t,
            t_psgr33=cabin_t,
            v_drvr1=cabin_v,
            v_drvr2=cabin_v,
            v_drvr3=cabin_v,
            v_psgr1=cabin_v,
            v_psgr2=cabin_v,
            v_psgr3=cabin_v,
            v_psgr21=cabin_v,
            v_psgr22=cabin_v,
            v_psgr23=cabin_v,
            v_psgr31=cabin_v,
            v_psgr32=cabin_v,
            v_psgr33=cabin_v,
            m_drvr1=cabin_t,
            m_drvr2=cabin_t,
            m_drvr3=cabin_t,
            m_psgr1=cabin_t,
            m_psgr2=cabin_t,
            m_psgr3=cabin_t,
            m_psgr21=cabin_t,
            m_psgr22=cabin_t,
            m_psgr23=cabin_t,
            m_psgr31=cabin_t,
            m_psgr32=cabin_t,
            m_psgr33=cabin_t,
            rhc=cabin_rh,
            ws=cabin_t,
        )

        h_x = kw_to_array(
            HVAC_XT_COLUMNS, cab_RH=cabin_rh, evp_mdot=cabin_v, vent_T=cabin_t
        )

        self.dv1_mlsim = make_dv1_sim(dv1_model, dv1_scaler, b_x)
        self.hvac_mlsim = make_hvac_sim(hvac_model, hvac_scaler, h_x)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass
