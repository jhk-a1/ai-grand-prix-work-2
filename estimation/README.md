# Error-State Kalman Filter

15-state ESKF (position, velocity, attitude error, accel bias, gyro bias)
driven by IMU, with position and yaw updates from vision when a gate is in
sight.

## Why it is built this way

The simulator publishes **no absolute position**. That was established by
logging every MAVLink message it emits: `HIGHRES_IMU`,
`ACTUATOR_OUTPUT_STATUS`, `COLLISION`, `HEARTBEAT` and an encapsulated
race/video stream — and nothing else. No `LOCAL_POSITION_NED`, no `ODOMETRY`.

So between gate sightings this filter is dead-reckoning IMU alone, and
double-integrating accelerometer error grows without bound. Three consequences
shaped the implementation:

- **Covariance must grow honestly while blind.** `_blind_boost()` inflates the
  velocity process noise by up to 8× the longer vision has been absent. A
  filter that reports steady confidence while dead-reckoning is worse than no
  filter, because the controller believes it.
- **Measurements are gated, not trusted.** A 3-DOF Mahalanobis test against
  chi2(3, 0.99) rejects detector failures. A 500 m jump is a bad detection,
  not a drone that teleported.
- **Physical limits bound the fiction.** Velocity is clamped at 25 m/s and
  altitude to the arena, so a diverged estimate degrades into a gentle drift
  rather than a full-throttle runaway.

## Tests

`pytest test_eskf.py -v` — 21 tests covering gravity alignment at
initialisation (including the ~18° nose-down launch stand the drone actually
starts on), free-fall and rotation integration, quaternion normalisation over
5,000 updates, covariance growth while blind, outlier rejection, and
robustness to NaN/infinite IMU, absurd timesteps and out-of-order timestamps.

The NaN test found a real defect: a single non-finite accelerometer sample
propagated into `p`, `v` **and** the covariance, and since every later update
is a function of `P`, the filter returned NaN forever while still looking like
a running filter. Fixed by rejecting non-finite IMU at the top of `propagate`
and counting it in `n_bad_imu`.
