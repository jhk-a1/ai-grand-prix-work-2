"""Error-State Kalman Filter — the only source of truth about where we are.

Nominal state : p (odom NED), v, q_odom_body, accel bias, gyro bias.
Error state   : [dp, dv, dtheta, dba, dbg]  (15).

Propagation from HIGHRES_IMU (specific force + gyro, body FRD) at ~120 Hz.
Corrections:
  * position pseudo-measurement from a PnP gate observation against a mapped
    gate (3-DOF, Mahalanobis-gated),
  * yaw from the observed gate normal vs the mapped normal (1-DOF),
  * roll/pitch from gravity when quasi-static (keeps SEARCH phases honest).

Q adapts to vision health: the longer we fly blind, the more the filter is
willing to believe the next measurement (blind_q_boost).
"""

from __future__ import annotations

import math

import numpy as np

from types_ import EstimatorConfig, ImuSample, StateEstimate
from mathutil import (quat_exp, quat_mul, quat_normalize, quat_to_rot,
                           skew, wrap_angle)


class Eskf:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.g = np.array([0.0, 0.0, cfg.gravity])   # NED: down is +z

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.eye(15) * 0.01
        self.last_gyro = np.zeros(3)

        self.initialized = False
        self.n_bad_imu = 0
        self._init_accum: list[ImuSample] = []
        self._init_t0: float | None = None
        self.t = 0.0
        self.last_vision_t = -1e9
        self.rejected_streak = 0

    # ------------------------------------------------------------ initialise
    def feed_init(self, imu: ImuSample) -> bool:
        """Accumulate at-rest IMU to bootstrap attitude and gyro bias.

        Only counts QUASI-STATIC samples — initializing mid-fall after a sim
        respawn bakes a garbage attitude into the filter. Any motion restarts
        the stillness window.
        """
        still = (abs(float(np.linalg.norm(imu.accel)) - self.cfg.gravity) < 0.5
                 and float(np.linalg.norm(imu.gyro)) < 0.15)
        if not still:
            self._init_t0 = None
            self._init_accum.clear()
            return False
        if self._init_t0 is None:
            self._init_t0 = imu.t
        self._init_accum.append(imu)
        if imu.t - self._init_t0 < self.cfg.init_still_time_s:
            return False
        acc = np.mean([s.accel for s in self._init_accum], axis=0)
        gyr = np.mean([s.gyro for s in self._init_accum], axis=0)
        # At rest the accelerometer reads -g in body frame: specific force up.
        # Recover roll/pitch (yaw unobservable -> 0).
        ax, ay, az = acc
        roll = math.atan2(-ay, -az)
        pitch = math.atan2(ax, math.hypot(ay, az))
        from mathutil import euler_to_quat
        self.q = euler_to_quat(roll, pitch, 0.0)
        self.bg = gyr.copy()
        self.ba = np.zeros(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.P = np.diag([
            1e-4, 1e-4, 1e-4,
            1e-3, 1e-3, 1e-3,
            5e-3, 5e-3, 1e-2,
            1e-2, 1e-2, 1e-2,
            1e-5, 1e-5, 1e-5,
        ])
        self.t = imu.t
        self.initialized = True
        return True

    # -------------------------------------------------------------- predict
    def propagate(self, imu: ImuSample) -> None:
        if not self.initialized:
            self.feed_init(imu)
            return
        # Reject non-finite IMU before it touches the state. One NaN sample
        # otherwise propagates into p, v AND the covariance, and because every
        # later update is a function of P the filter never recovers -- it
        # reports NaN forever while still looking like a running filter.
        if not (np.all(np.isfinite(imu.accel)) and np.all(np.isfinite(imu.gyro))
                and math.isfinite(imu.t)):
            self.n_bad_imu += 1
            return
        dt = imu.t - self.t
        if dt <= 0.0:
            return
        dt = min(dt, 0.05)      # guard against hiccups
        self.t = imu.t

        acc = imu.accel - self.ba
        gyr = imu.gyro - self.bg
        self.last_gyro = gyr
        R = quat_to_rot(self.q)

        # Nominal state integration (specific force + gravity in NED).
        a_world = R @ acc + self.g
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        # Physical sanity: this quad cannot exceed ~25 m/s. Dead-reckoning
        # drift beyond that is estimator fiction — clamp it so downstream
        # control degrades gently instead of chasing phantoms.
        speed = float(np.linalg.norm(self.v))
        if speed > 25.0:
            self.v *= 25.0 / speed
        # Arena sanity: the venue is a warehouse. z (NED down) outside
        # [-30 m up, +2 m below pad] is estimator fiction — clamp and bleed
        # the runaway vertical velocity.
        if self.p[2] > 2.0:
            self.p[2] = 2.0
            self.v[2] = min(self.v[2], 0.0)
        elif self.p[2] < -30.0:
            self.p[2] = -30.0
            self.v[2] = max(self.v[2], 0.0)
        self.q = quat_normalize(quat_mul(self.q, quat_exp(gyr * dt)))

        # Error-state Jacobian (first order).
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(acc) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = np.eye(3) - skew(gyr) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        c = self.cfg
        boost = self._blind_boost(imu.t)
        Qc = np.zeros((15, 15))
        Qc[3:6, 3:6] = np.eye(3) * (c.sigma_acc ** 2) * boost
        Qc[6:9, 6:9] = np.eye(3) * (c.sigma_gyro ** 2)
        Qc[9:12, 9:12] = np.eye(3) * (c.sigma_acc_bias ** 2)
        Qc[12:15, 12:15] = np.eye(3) * (c.sigma_gyro_bias ** 2)
        self.P = F @ self.P @ F.T + Qc * dt
        self.P = 0.5 * (self.P + self.P.T)
        # Covariance explosion guard: without vision fixes P grows without
        # bound and eventually poisons every update numerically.
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-9, 100.0))

    def _blind_boost(self, now: float) -> float:
        dt_blind = now - self.last_vision_t
        if dt_blind <= 0.1:
            return 1.0
        frac = min(1.0, (dt_blind - 0.1) / self.cfg.blind_q_ramp_s)
        return 1.0 + frac * (self.cfg.blind_q_boost_max - 1.0)

    # -------------------------------------------------------------- updates
    def update_position(self, p_meas: np.ndarray, r_scale: float) -> bool:
        """3-DOF drone-position pseudo-measurement (from gate PnP vs map)."""
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        R = np.eye(3) * (self.cfg.r_pnp_pos_base * r_scale)
        innov = p_meas - self.p
        S = H @ self.P @ H.T + R
        try:
            maha = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return False
        if maha > self.cfg.mahalanobis_thresh_3dof:
            self.rejected_streak += 1
            if self.rejected_streak >= 15:
                # The filter has drifted so far that it rejects reality.
                # Reality wins: hard-reset position onto the measurement.
                self.p = p_meas.copy()
                self.v *= 0.5
                self.P[0:3, 0:3] = np.eye(3) * 1.0
                self.last_vision_t = self.t
                self.rejected_streak = 0
                return True
            return False
        self._apply(H, innov, S, R)
        self.last_vision_t = self.t
        self.rejected_streak = 0
        return True

    def update_yaw(self, yaw_meas: float) -> bool:
        """1-DOF yaw update (from gate-normal heading vs map)."""
        from mathutil import quat_to_euler
        _, _, yaw = quat_to_euler(self.q)
        innov = np.array([wrap_angle(yaw_meas - yaw)])
        H = np.zeros((1, 15))
        H[0, 8] = 1.0
        R = np.array([[self.cfg.r_pnp_yaw]])
        S = H @ self.P @ H.T + R
        if float(innov @ np.linalg.solve(S, innov)) > 6.63:   # chi2(1, 0.99)
            return False
        self._apply(H, innov, S, R)
        return True

    def update_gravity(self, accel: np.ndarray) -> bool:
        """Roll/pitch correction when quasi-static (|f| ~ g, low rates).

        Deliberately NOT a Kalman update: the yaw component is unobservable
        (H is rank-2) and with an unbounded P this became a numerical bomb.
        A small fixed-gain complementary correction cannot explode and is all
        the accuracy this needs.
        """
        # STRICT quasi-static gate: during forward acceleration the specific
        # force tilts, and treating it as gravity pitches the estimate up ->
        # the controller dives to compensate -> runaway dive into the floor.
        # Fire only when there is essentially no dynamics.
        f = accel - self.ba
        norm = np.linalg.norm(f)
        if abs(norm - self.cfg.gravity) > 0.35 or np.linalg.norm(self.last_gyro) > 0.25:
            return False
        R = quat_to_rot(self.q)
        pred = -(R.T @ self.g) / self.cfg.gravity   # expected f direction, body
        meas = f / norm
        # With right-multiplied error dtheta:  pred ~= meas - dtheta x meas,
        # so  e = meas x pred ~= -dtheta_perp  and the corrective rotation is
        # exp(+gain*e). (The opposite sign AMPLIFIES the error - verified the
        # hard way.)
        e = np.cross(meas, pred)
        gain = 0.06
        self.q = quat_normalize(quat_mul(self.q, quat_exp(gain * e)))
        return True

    def _apply(self, H: np.ndarray, innov: np.ndarray, S: np.ndarray,
               R: np.ndarray) -> None:
        try:
            K = np.linalg.solve(S.T, H @ self.P).T      # K = P H^T S^-1
        except np.linalg.LinAlgError:
            return
        dx = K @ innov
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_normalize(quat_mul(self.q, quat_exp(dx[6:9])))
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------- accessors
    def estimate(self) -> StateEstimate:
        pos_var = float(np.max(np.diag(self.P)[0:3])) if self.initialized else 1e9
        return StateEstimate(
            t=self.t,
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            omega=self.last_gyro.copy(),
            vision_healthy=(self.t - self.last_vision_t) < 0.5,
            pos_std=math.sqrt(max(pos_var, 0.0)),
        )

    @property
    def R_world_body(self) -> np.ndarray:
        return quat_to_rot(self.q)
