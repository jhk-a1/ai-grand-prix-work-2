"""Thousands of quadrotors flying a gate course in parallel, in pure PyTorch.

Isaac Sim earns its RTX requirement on rendering and contact-rich physics --
articulated arms, grasping, cloth. A quadrotor racing through gates has none of
that. Its state is 13 floats, its dynamics are about ten lines of arithmetic,
and "did it pass the gate" is a plane crossing plus a radius test. All of that
batches perfectly, so N drones cost almost exactly as much as one.

That means the interesting part of the Isaac/Swift pipeline -- massively
parallel experience collection for RL -- runs fine on a laptop CPU.

Frames match the rest of the stack: NED, +z DOWN, thrust along body -z.
Altitude is -p[2].

    from vec_quad import VecQuad
    env = VecQuad(n_envs=2048)
    obs = env.reset()
    obs, rew, done, info = env.step(actions)
"""

from __future__ import annotations

import math

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product, scalar-first [w, x, y, z]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix, v_world = R @ v_body."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),
    ], dim=-2)


def quat_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector -> quaternion, safe at zero."""
    ang = v.norm(dim=-1, keepdim=True)
    small = ang < 1e-8
    axis = v / torch.where(small, torch.ones_like(ang), ang)
    half = 0.5 * ang
    q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    ident = torch.zeros_like(q); ident[..., 0] = 1.0
    return torch.where(small, ident, q)


def default_course(device) -> tuple[torch.Tensor, torch.Tensor]:
    """A climbing, turning circuit -- roughly the shape of the real course.

    Returns (positions [G,3], normals [G,3]) in NED. The normal is the
    direction of travel THROUGH the gate, which is what makes a signed plane
    crossing meaningful.
    """
    pts = [
        (12.0,   0.0, -4.0),
        (24.0,   6.0, -6.0),
        (32.0,  18.0, -8.0),
        (30.0,  32.0, -8.0),
        (18.0,  40.0, -10.0),
        (4.0,   38.0, -12.0),
        (-6.0,  28.0, -12.0),
        (-8.0,  14.0, -9.0),
        (0.0,    4.0, -6.0),
        (10.0,  -6.0, -5.0),
    ]
    g = torch.tensor(pts, dtype=torch.float32, device=device)
    # Normal at each gate = direction from the previous gate to the next one,
    # so the gate faces the way the course actually flows through it.
    prev = torch.cat([torch.zeros(1, 3, device=device), g[:-1]], 0)
    nxt = torch.cat([g[1:], g[-1:] + (g[-1] - g[-2])[None]], 0)
    n = nxt - prev
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return g, n


class VecQuad:
    """A batch of independent quadrotors racing the same gate course."""

    def __init__(self, n_envs: int = 2048, device: str = "cpu",
                 dt: float = 0.01, max_steps: int = 500,
                 gate_radius: float = 0.75, seed: int = 0,
                 # plant -- same shape as apex.sim.twin, fitted to real runs
                 tau_rate: float = 0.055, drag: float = 0.012,
                 thrust_max_g: float = 4.0, max_rate: float = 6.0,
                 randomize: bool = True, spawn_on_course: bool = True,
                 course: str = "demo"):
        self.n = n_envs
        self.device = torch.device(device)
        self.dt = dt
        self.max_steps = max_steps
        self.gate_radius = gate_radius
        self.max_rate = max_rate
        self.g = 9.80665
        self.randomize = randomize
        self.spawn_on_course = spawn_on_course
        torch.manual_seed(seed)

        if course == "aigp":
            # Real course: sketch topology fused with measured leg lengths
            # and altitudes. Legs run 7-59 m, versus a uniform ~15 m on the
            # demo course, so the policy must handle both close-packed gates
            # and long ballistic throws.
            from course_aigp import aigp_course
            self.gates, self.normals = aigp_course(self.device)
        else:
            self.gates, self.normals = default_course(self.device)
        self.n_gates = self.gates.shape[0]

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.p, self.v, self.w = z(self.n, 3), z(self.n, 3), z(self.n, 3)
        self.q = z(self.n, 4); self.q[:, 0] = 1.0
        self.target = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.prev_act = z(self.n, 4)
        self.prev_dist = z(self.n)

        # Domain randomisation. The real plant's thrust curve is uncertain, so
        # rather than guess one number the policy is trained across a RANGE and
        # must cope with any of them. This is what makes sim-to-real work
        # without a perfect model.
        self.tau0, self.drag0, self.tmax0 = tau_rate, drag, thrust_max_g
        self.tau = torch.full((self.n,), tau_rate, device=self.device)
        self.drag = torch.full((self.n,), drag, device=self.device)
        self.tmax = torch.full((self.n,), thrust_max_g, device=self.device)

        # Arena bound, derived from the course rather than hard-coded. A
        # fixed 120 m was fine for the ~50 m demo course but silently killed
        # every drone on the real one, which reaches 243 m from the origin:
        # gates 7-15 all sat outside it and terminated on spawn, which reads
        # as "the policy cannot fly them" and is nothing of the kind.
        self.arena_r = float(self.gates.norm(dim=-1).max()) + 60.0
        _legs = (self.gates - torch.roll(self.gates, 1, 0)).norm(dim=-1)
        self.max_leg = float(_legs[1:].max())
        self.obs_dim = 17
        self.act_dim = 4

    # ------------------------------------------------------------------
    def _randomize(self, mask: torch.Tensor) -> None:
        if not self.randomize:
            return
        k = int(mask.sum())
        if k == 0:
            return
        u = lambda lo, hi: torch.rand(k, device=self.device) * (hi - lo) + lo
        self.tau[mask] = u(0.03, 0.12)
        self.drag[mask] = u(0.000, 0.030)
        self.tmax[mask] = u(2.0, 5.0)      # thrust-to-weight spread

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        k = int(mask.sum())
        if k:
            # Curriculum by initialisation. Spawning every drone at the origin
            # means it must discover a 12 m flight to gate 0 before it earns a
            # single point -- which is why the first attempt sat at 0.01
            # gates/episode. Scattering the batch along the WHOLE course means
            # every drone starts near some gate, so the progress signal is
            # immediate and later gates get explored from the start instead of
            # only after the earlier ones are solved.
            if self.spawn_on_course:
                gi = torch.randint(0, self.n_gates, (k,), device=self.device)
                base = self.gates[gi]
                # sit a little way back along the approach to that gate
                legs = (self.gates - torch.roll(self.gates, 1, 0)).norm(dim=-1)
                legs = legs.clamp(4.0, 60.0)[gi].unsqueeze(-1)
                back = self.normals[gi] * (0.15 + torch.rand(k, 1, device=self.device) * 0.7) * legs
                self.p[mask] = base - back + torch.randn(k, 3, device=self.device) * 0.8
                self.target[mask] = gi
                # launch roughly along the course direction, not from rest
                self.v[mask] = self.normals[gi] * (torch.rand(k, 1, device=self.device) * 6.0) \
                    + torch.randn(k, 3, device=self.device) * 0.5
            else:
                self.p[mask] = torch.randn(k, 3, device=self.device) * 0.5
                self.p[mask, 2] = -2.0 + torch.randn(k, device=self.device) * 0.3
                self.v[mask] = torch.randn(k, 3, device=self.device) * 0.5
                self.target[mask] = 0
            # Stepping back along a leg whose normal points upward puts the
            # spawn underground; on the real course that killed 6.5% of drones
            # before their first command.
            self.p[mask, 2] = self.p[mask, 2].clamp(max=-1.5)
            self.w[mask] = 0.0
            qq = torch.zeros(k, 4, device=self.device); qq[:, 0] = 1.0
            self.q[mask] = qq
            self.steps[mask] = 0
            self.prev_act[mask] = 0.0
            self._randomize(mask)
            self.prev_dist[mask] = (self.gates[self.target[mask]]
                                    - self.p[mask]).norm(dim=-1)
        return self._obs()

    # ------------------------------------------------------------------
    def _obs(self) -> torch.Tensor:
        R = quat_to_rot(self.q)                      # [N,3,3]
        Rt = R.transpose(-1, -2)                     # world -> body
        gate = self.gates[self.target]
        nxt = self.gates[(self.target + 1) % self.n_gates]
        to_g = gate - self.p
        dist = to_g.norm(dim=-1, keepdim=True)
        dir_b = torch.bmm(Rt, (to_g / dist.clamp_min(1e-6)).unsqueeze(-1)).squeeze(-1)
        nxt_b = torch.bmm(Rt, (nxt - self.p).unsqueeze(-1)).squeeze(-1) / 30.0
        v_b = torch.bmm(Rt, self.v.unsqueeze(-1)).squeeze(-1) / 20.0
        # Gravity in the body frame encodes attitude without any wrap-around,
        # which a raw quaternion does not (q and -q are the same rotation).
        g_w = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.n, 3)
        g_b = torch.bmm(Rt, g_w.unsqueeze(-1)).squeeze(-1)
        return torch.cat([dir_b, (dist / 30.0).clamp(0, 2), nxt_b, v_b, g_b,
                          self.prev_act], dim=-1)

    # ------------------------------------------------------------------
    def step(self, act: torch.Tensor):
        act = act.clamp(-1.0, 1.0)
        rates = act[:, :3] * self.max_rate
        # map [-1,1] -> [0, tmax] g of specific force
        thrust = (act[:, 3] + 1.0) * 0.5 * self.tmax * self.g

        # rate loop lag
        alpha = 1.0 - torch.exp(-self.dt / self.tau).unsqueeze(-1)
        self.w = self.w + (rates - self.w) * alpha
        self.q = quat_mul(self.q, quat_exp(self.w * self.dt))
        self.q = self.q / self.q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        R = quat_to_rot(self.q)
        zb = R[:, :, 2]                                   # body z in world
        a = -zb * thrust.unsqueeze(-1)                    # thrust along body -z
        a[:, 2] += self.g                                 # NED: gravity is +z
        sp = self.v.norm(dim=-1, keepdim=True)
        a = a - self.drag.unsqueeze(-1) * sp * self.v

        p_old = self.p.clone()
        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt
        self.steps += 1

        # --- gate crossing: signed plane test, then a radius test ---------
        gate = self.gates[self.target]
        nrm = self.normals[self.target]
        d_old = ((p_old - gate) * nrm).sum(-1)
        d_new = ((self.p - gate) * nrm).sum(-1)
        crossed = (d_old < 0) & (d_new >= 0)
        # perpendicular miss distance at the crossing
        perp = (self.p - gate) - d_new.unsqueeze(-1) * nrm
        through = crossed & (perp.norm(dim=-1) < self.gate_radius)

        dist = (gate - self.p).norm(dim=-1)
        # dense shaping: reward closing on the gate, and nothing else
        progress = (self.prev_dist - dist).clamp(-2.0, 2.0)
        self.prev_dist = dist

        rew = 2.0 * progress
        rew = rew + 10.0 * through.float()
        rew = rew - 0.002 * (act - self.prev_act).pow(2).sum(-1)   # smoothness
        rew = rew + 0.01                                           # alive

        self.target = torch.where(through,
                                  (self.target + 1) % self.n_gates, self.target)
        # a fresh target needs a fresh baseline or the first step reads as a
        # huge negative "progress"
        if bool(through.any()):
            g2 = self.gates[self.target]
            self.prev_dist = torch.where(through,
                                         (g2 - self.p).norm(dim=-1),
                                         self.prev_dist)

        # Both bounds are derived from the loaded course. Hard-coded 120 m /
        # 60 m were sized for the small demo course and silently killed every
        # drone at gates 7-15 of the real one (which reaches 243 m), and any
        # drone correctly flying its 59 m leg. That reads as "the policy can't
        # fly" and is nothing of the kind.
        crashed = (self.p[:, 2] > 0.0) | (self.p.norm(dim=-1) > self.arena_r) \
            | (dist > self.max_leg + 25.0)
        timeout = self.steps >= self.max_steps
        rew = rew - 10.0 * crashed.float()
        done = crashed | timeout

        self.prev_act = act
        obs = self._obs()
        info = {"through": through, "crashed": crashed, "timeout": timeout}
        if bool(done.any()):
            self.reset(done)
            obs = self._obs()
        return obs, rew, done, info
