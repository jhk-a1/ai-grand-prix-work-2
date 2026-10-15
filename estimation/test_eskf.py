"""Tests for the error-state Kalman filter.

These check the properties that actually matter in flight, not just that the
code runs. The filter's job on this platform is unusual: the simulator
publishes no absolute position at all, so between gate sightings the filter is
integrating IMU alone and the thing that keeps it honest is its covariance
growing to match its real ignorance.

    pytest test_eskf.py -v
"""

import math

import numpy as np
import pytest

from eskf import Eskf
from types_ import EstimatorConfig, ImuSample
from mathutil import euler_to_quat, quat_to_euler

G = 9.80665


def imu(t, accel=None, gyro=None):
    """An IMU sample. At rest the accelerometer reads +1 g along body -z."""
    return ImuSample(t=t, t_sim_us=int(t * 1e6),
                     accel=np.array(accel if accel is not None else [0, 0, -G]),
                     gyro=np.array(gyro if gyro is not None else [0, 0, 0]))


def settled(cfg=None, t_end=0.6, dt=0.005):
    """A filter that has seen enough still samples to initialise."""
    f = Eskf(cfg or EstimatorConfig())
    t = 0.0
    while t < t_end:
        f.feed_init(imu(t))
        t += dt
    return f, t


# --------------------------------------------------------------- init
def test_initialises_from_stillness():
    f, _ = settled()
    assert f.estimate() is not None


def test_init_aligns_roll_and_pitch_to_gravity():
    """Sitting level, the filter must deduce level attitude from gravity."""
    f, _ = settled()
    roll, pitch, _ = quat_to_euler(f.estimate().q)
    assert abs(roll) < math.radians(2.0)
    assert abs(pitch) < math.radians(2.0)


def test_init_recovers_a_tilted_start():
    """The drone starts on a stand pitched ~18 deg nose-down, not level."""
    f = Eskf(EstimatorConfig())
    tilt = math.radians(-18.0)
    # specific force rotates with the body
    a = np.array([G * math.sin(tilt), 0.0, -G * math.cos(tilt)])
    t = 0.0
    while t < 0.6:
        f.feed_init(imu(t, accel=a))
        t += 0.005
    _, pitch, _ = quat_to_euler(f.estimate().q)
    assert abs(pitch - tilt) < math.radians(4.0), f"got {math.degrees(pitch):.1f} deg"


# --------------------------------------------------- propagation sanity
def test_at_rest_position_does_not_run_away():
    """Hovering with thrust exactly cancelling gravity must not drift fast."""
    f, t = settled()
    for _ in range(400):                      # 2 s at 200 Hz
        t += 0.005
        f.propagate(imu(t))
    assert np.linalg.norm(f.estimate().p) < 1.0


def test_free_fall_accelerates_downward_at_g():
    """Zero specific force is free fall: +z is DOWN, so v[2] goes positive.

    Started well above the pad, because the filter clamps z to +2 m (below
    the pad is estimator fiction) and zeroes the vertical velocity there.
    """
    f, t = settled()
    f.p = np.array([0.0, 0.0, -20.0])         # 20 m up, room to fall
    for _ in range(100):                      # 0.5 s
        t += 0.005
        f.propagate(imu(t, accel=[0.0, 0.0, 0.0]))
    assert f.estimate().v[2] == pytest.approx(0.5 * G, rel=0.15)


def test_arena_clamp_stops_the_estimate_falling_through_the_floor():
    """Below the pad is not a place the drone can be; the filter says so."""
    f, t = settled()
    for _ in range(400):
        t += 0.005
        f.propagate(imu(t, accel=[0.0, 0.0, 0.0]))
    assert f.estimate().p[2] <= 2.0 + 1e-6
    assert f.estimate().v[2] <= 0.0 + 1e-6


def test_gyro_integrates_into_attitude():
    f, t = settled()
    rate = 1.0                                # rad/s about body x
    for _ in range(200):                      # 1 s -> ~1 rad
        t += 0.005
        f.propagate(imu(t, gyro=[rate, 0.0, 0.0]))
    roll, _, _ = quat_to_euler(f.estimate().q)
    assert roll == pytest.approx(1.0, abs=0.15)


def test_quaternion_stays_normalised_over_a_long_run():
    """Thousands of manifold updates must not let |q| drift."""
    f, t = settled()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        t += 0.005
        f.propagate(imu(t, gyro=rng.normal(0, 0.5, 3)))
    assert np.linalg.norm(f.estimate().q) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------- covariance behaviour
def test_uncertainty_grows_while_blind():
    """With no measurements the filter must admit it is losing track.

    This is the property the whole architecture leans on -- a filter that
    reports steady confidence while dead-reckoning is actively dangerous,
    because the controller believes it.
    """
    f, t = settled()
    early = None
    for i in range(600):
        t += 0.005
        f.propagate(imu(t))
        if i == 20:
            early = f.estimate().pos_std
    assert f.estimate().pos_std > early


def test_position_update_pulls_the_estimate_toward_the_measurement():
    f, t = settled()
    for _ in range(100):
        t += 0.005
        f.propagate(imu(t))
    before = f.estimate().p.copy()
    # Small enough to pass the Mahalanobis gate against a freshly initialised
    # covariance -- a 5 m jump from a confident filter is correctly refused.
    target = before + np.array([0.05, 0.0, -0.05])
    assert f.update_position(target, r_scale=1.0) is True
    after = f.estimate().p
    assert np.linalg.norm(after - target) < np.linalg.norm(before - target)


def test_wild_outlier_is_rejected_by_mahalanobis_gate():
    """A 500 m jump is a detector failure, not a drone that teleported."""
    f, t = settled()
    for _ in range(100):
        t += 0.005
        f.propagate(imu(t))
    assert f.update_position(np.array([500.0, 500.0, -500.0]), r_scale=1.0) is False


def test_a_rejected_outlier_leaves_the_state_untouched():
    f, t = settled()
    for _ in range(100):
        t += 0.005
        f.propagate(imu(t))
    before = f.estimate().p.copy()
    f.update_position(np.array([900.0, -900.0, 900.0]), r_scale=1.0)
    assert np.allclose(f.estimate().p, before)


# ------------------------------------------------------------ robustness
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_imu_never_poisons_the_state(bad):
    """One bad sample must not turn the whole estimate into NaN forever."""
    f, t = settled()
    for _ in range(50):
        t += 0.005
        f.propagate(imu(t))
    t += 0.005
    f.propagate(imu(t, accel=[bad, 0.0, -G]))  # must be rejected, not absorbed
    assert f.n_bad_imu == 1
    for _ in range(50):
        t += 0.005
        f.propagate(imu(t))
    e = f.estimate()
    assert np.isfinite(e.p).all() and np.isfinite(e.v).all() \
        and np.isfinite(e.q).all()


@pytest.mark.parametrize("dt", [1e-9, 1e-4, 0.25, 5.0])
def test_absurd_timesteps_do_not_explode(dt):
    """Dropped frames and stalls produce dt values nothing was designed for."""
    f, t = settled()
    for _ in range(40):
        t += dt
        f.propagate(imu(t))
    e = f.estimate()
    assert np.isfinite(e.p).all() and np.isfinite(e.v).all()


def test_out_of_order_timestamps_are_survivable():
    f, t = settled()
    f.propagate(imu(t + 0.01))
    f.propagate(imu(t + 0.005))               # went backwards
    f.propagate(imu(t + 0.02))
    assert np.isfinite(f.estimate().p).all()


def test_rotation_matrix_is_orthonormal():
    f, t = settled()
    rng = np.random.default_rng(1)
    for _ in range(300):
        t += 0.005
        f.propagate(imu(t, gyro=rng.normal(0, 0.8, 3)))
    R = f.R_world_body
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
