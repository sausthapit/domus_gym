"""acoustics.py

Convert blower power and vehicle speed into an estimate of the sound
level being heard by the driver.

Author
------
S. Moeller

Date
----
29 September 2021

"""

import numpy as np
import scipy.interpolate as sint

# speed ... vehicle speed in km per hour
# power ... power consumption of the HVAC blower in watts
# soundLevel ... soundLevel in dB


def PaToDB(p):
    p0 = 2.0e-5
    return max(0.0, 10.0 * np.log10((p * p + 1e-18) / p0 / p0))


def dBtoPa(dB):
    p0 = 2.0e-5
    return np.sqrt(10.0 ** (dB / 10.0)) * p0


def calc_sound_level(speed, power):
    """calc_sound_level - convert blower power and vehicle speed into sound level

    Parameters
    ----------

    speed : float

      vehicle speed in km per hour

    power : float

      blower power in watts

    Returns
    -------

      tuple containing: estimated sound level in dB, driving sound, blower sound

    """
    coeff = 4.85
    drivingSound = coeff * np.sqrt(speed)
    drivingSound_Pa = dBtoPa(drivingSound)

    blower_vdot_hi = 344.0  # level 12 of 12
    blower_power_hi = 400.0  # Watts estimated from CRF simulation data
    blower_SPL_hi = 47.0
    blower_SP_hi_Pa = dBtoPa(blower_SPL_hi)
    # blower_vdot_lo = 222.0 # level 8 of 12
    blower_vdot_lo = 169.0  # level 6 of 12
    blower_power_lo = (
        blower_power_hi * blower_vdot_lo / blower_vdot_hi
    )  # assumption power approx. linear to vdot, from data in DNTS HVAC Performance Report
    blower_SPL_lo = 23.0
    blower_SP_lo_Pa = dBtoPa(blower_SPL_lo)

    powersteps = np.array([0.0, blower_power_lo, blower_power_hi])
    soundpressure = np.array([0.0, blower_SP_lo_Pa, blower_SP_hi_Pa])
    f = sint.interp1d(powersteps, soundpressure, fill_value="extrapolate")
    blowerSound_Pa = f(power)
    blowerSound = PaToDB(blowerSound_Pa)

    soundLevel = PaToDB(drivingSound_Pa + blowerSound_Pa)

    return soundLevel, drivingSound, blowerSound
