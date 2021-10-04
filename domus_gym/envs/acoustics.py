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

# speed ... vehicle speed in km per hour
# power ... power consumption of the HVAC blower in watts
# soundLevel ... soundLevel in dB


def PaToDB(p):
    p0 = 2.0e-5
    return 10.0 * np.log10(p * p / p0 / p0)


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

      estimated sound level in dB

    """
    ds_lo = 43.0
    ds_lo_Pa = dBtoPa(ds_lo)
    ds_lo_speed = 50.0
    ds_hi = 75.0
    ds_hi_Pa = dBtoPa(ds_hi)
    ds_hi_speed = 105.0
    drivingSound_Pa = ds_lo_Pa + (ds_hi_Pa - ds_lo_Pa) * (speed - ds_lo_speed) / (
        ds_hi_speed - ds_lo_speed
    )
    drivingSound = PaToDB(drivingSound_Pa)

    bs_lo = 43.0
    bs_lo_Pa = dBtoPa(bs_lo)
    bs_lo_power = 50.0
    bs_hi = 75.0
    bs_hi_Pa = dBtoPa(bs_hi)
    bs_hi_power = 105.0
    blowerSound_Pa = bs_lo_Pa + (bs_hi_Pa - bs_lo_Pa) * (power - bs_lo_power) / (
        bs_hi_power - bs_lo_power
    )
    blowerSound = PaToDB(blowerSound_Pa)

    soundLevel = PaToDB(drivingSound_Pa + blowerSound_Pa)

    return soundLevel, drivingSound, blowerSound
