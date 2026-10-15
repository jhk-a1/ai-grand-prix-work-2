"""Types the ESKF needs, extracted so this directory runs on its own.

In the full stack these live in apex/common.py and apex/config.py alongside
the comms layer, controller and planner. Everything here is a verbatim subset
-- nothing was simplified for display.

Frames are NED/FRD throughout: +z is DOWN, thrust acts along body -z, and
altitude is -p[2].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImuSample:
    t: float                 # local monotonic receive time [s]
    t_sim_us: int            # simulator timestamp [us]
    accel: np.ndarray        # specific force, body FRD [m/s^2]
    gyro: np.ndarray         # angular rate, body FRD [rad/s]


@dataclass
class StateEstimate:
    t: float
    p: np.ndarray            # position, odom NED [m]
    v: np.ndarray            # velocity, odom NED [m/s]
    q: np.ndarray            # q_odom_body, scalar first [w,x,y,z]
    omega: np.ndarray        # body rates [rad/s], bias-corrected
    vision_healthy: bool = True
    pos_std: float = 0.0     # sqrt of max position variance -- confidence


@dataclass
class EstimatorConfig:
    """Process and measurement noise.

    The vision-loss ramp matters: this simulator publishes NO absolute
    position (verified by logging every MAVLink message it sends), so between
    gate sightings the filter is dead-reckoning IMU alone and its covariance
    must be allowed to grow honestly rather than claiming confidence it has
    not earned.
    """
    gravity: float = 9.80665
    sigma_acc: float = 0.35            # m/s^2/sqrt(Hz), accel white noise
    sigma_gyro: float = 0.02           # rad/s/sqrt(Hz)
    sigma_acc_bias: float = 0.01       # random walk
    sigma_gyro_bias: float = 0.0005
    blind_q_boost_max: float = 8.0     # inflate Q by up to this when blind
    blind_q_ramp_s: float = 0.5
    r_pnp_pos_base: float = 0.02       # m^2 at 5 m, scaled by range^2
    r_pnp_yaw: float = 0.03            # rad^2
    mahalanobis_thresh_3dof: float = 11.34   # chi2(3, 0.99)
    init_still_time_s: float = 0.4
