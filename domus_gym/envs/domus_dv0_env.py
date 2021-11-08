import numpy as np

from domus_mlsim import (
    DV0_XT_COLUMNS,
    KELVIN,
    DV0Ut,
    DV0Xt,
    HvacUt,
    estimate_cabin_temperature_dv0,
    hcm_reduced,
    kw_to_array,
    load_dv0,
    make_dv0_sim,
    update_control_inputs_dv0,
    update_dv0_inputs,
)

from . import DomusContEnv
from .acoustics import calc_sound_level


class DomusDv0ContEnv(DomusContEnv):
    CABIN_ENERGY = np.zeros((len(DV0Ut)))

    def __init__(
        self,
        **kwargs,
    ):
        """Description:
            Simulation of the DV0 thermal environment of a Fiat 500e car
            cabin.

        This modifies DomusContEnv by overriding use of DV1 with DV0 where needed.

        """
        super().__init__(**kwargs)
        self.b_u = np.zeros((len(DV0Ut)))
        self.dv1_scaler_and_model = None
        self.dv0_scaler_and_model = load_dv0()

    def _convert_state(self):
        """given the current state, create a vector that can be used as input to the controller"""
        cab_t = estimate_cabin_temperature_dv0(self.b_x)
        update_control_inputs_dv0(self.c_u, self.b_x, self.h_x, cab_t)
        return self.obs_tr.transform(self.c_u)

    def _body_state(self, b_x, n):
        """return the body state matrix for passenger n where 0 is the driver, etc"""
        if n == 0:
            v = b_x[
                [
                    DV0Xt.t_drvr1,
                    DV0Xt.m_drvr1,
                    DV0Xt.v_drvr1,
                    DV0Xt.t_drvr2,
                    DV0Xt.m_drvr2,
                    DV0Xt.v_drvr2,
                    DV0Xt.t_drvr3,
                    DV0Xt.m_drvr3,
                    DV0Xt.v_drvr3,
                ]
            ]
        elif n == 1:
            v = b_x[
                [
                    DV0Xt.t_psgr1,
                    DV0Xt.m_psgr1,
                    DV0Xt.v_psgr1,
                    DV0Xt.t_psgr2,
                    DV0Xt.m_psgr2,
                    DV0Xt.v_psgr2,
                    DV0Xt.t_psgr3,
                    DV0Xt.m_psgr3,
                    DV0Xt.v_psgr3,
                ]
            ]
        elif n == 2:
            v = b_x[
                [
                    DV0Xt.t_psgr21,
                    DV0Xt.m_psgr21,
                    DV0Xt.v_psgr21,
                    DV0Xt.t_psgr22,
                    DV0Xt.m_psgr22,
                    DV0Xt.v_psgr22,
                    DV0Xt.t_psgr23,
                    DV0Xt.m_psgr23,
                    DV0Xt.v_psgr23,
                ]
            ]
        elif n == 3:
            v = b_x[
                [
                    DV0Xt.t_psgr31,
                    DV0Xt.m_psgr31,
                    DV0Xt.v_psgr31,
                    DV0Xt.t_psgr32,
                    DV0Xt.m_psgr32,
                    DV0Xt.v_psgr32,
                    DV0Xt.t_psgr33,
                    DV0Xt.m_psgr33,
                    DV0Xt.v_psgr33,
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
                pre_clo=self.pre_clo,
                pre_out=h_u[HvacUt.ambient] - KELVIN,
                body_state=self._body_state(b_x, i),
                rh=b_x[DV0Xt.rhc] * 100,
                sound=calc_sound_level(h_u[HvacUt.speed], h_u[HvacUt.blw_power])[0],
            )
            for i in self.configured_passengers
        ]
        return np.mean(hcm)

    def _ws_and_rh(self, b_x):
        return b_x[DV0Xt.ws], b_x[DV0Xt.rhc]

    def _step_cabin(self, c_x):
        self.b_u[
            [
                DV0Ut.t_a,
                DV0Ut.rh_a,
                DV0Ut.rad1,
                DV0Ut.rad2,
                DV0Ut.VehicleSpeed,
            ]
        ] = [
            self.ambient_t,
            self.ambient_rh,
            self.solar1,
            self.solar2,
            self.car_speed / 100 * 27.778,
        ]
        update_dv0_inputs(self.b_u, self.h_x, c_x)
        print(self.b_u)
        _, self.b_x = self.dv0_sim.step(self.b_u)

    def _make_cabin_state(self):
        return kw_to_array(
            DV0_XT_COLUMNS,
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

    def _make_cabin_sim(self):
        self.dv0_sim = make_dv0_sim(self.dv0_scaler_and_model, self.b_x)
