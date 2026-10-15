"""Tests for the vectorised quadrotor environment.

An RL environment is uniquely dangerous to get wrong, because a broken one
does not raise -- it trains. The agent happily optimises whatever you actually
built, and the only symptom is a learning curve that plateaus somewhere
disappointing. Every serious bug found in this environment was of that kind:

  * an arena radius of 120 m, sized for a small test course, terminated every
    agent at gates 7-15 of a 243 m course before it could move. That read as
    "the policy cannot fly those gates" and was nothing of the kind.
  * a fixed spawn stand-off placed agents past the gate on short legs.
  * spawning everything at the origin meant a 12 m exploration problem had to
    be solved before any reward existed at all.

So these check physics, geometry and termination directly, rather than
inferring them from a reward curve.

    pytest test_vec_quad.py -v
"""

import time

import pytest
import torch

from vec_quad import VecQuad, quat_exp, quat_mul, quat_to_rot

G = 9.80665


def env(**kw):
    kw.setdefault("n_envs", 256)
    kw.setdefault("randomize", False)
    return VecQuad(**kw)


def hover_action(e, n):
    """Action whose thrust is exactly 1 g, given the un-randomised plant."""
    a = torch.zeros(n, 4)
    a[:, 3] = 2.0 * (1.0 / float(e.tmax[0])) - 1.0
    return a


# ----------------------------------------------------------------- physics
def test_hovering_holds_altitude():
    e = env(n_envs=64, spawn_on_course=False)
    e.reset()
    e.v[:] = 0.0                       # drop the randomised launch velocity
    z0 = e.p[:, 2].clone()
    a = hover_action(e, 64)
    for _ in range(200):               # 2 s
        e.step(a)
    assert float((e.p[:, 2] - z0).abs().mean()) < 0.05


def test_zero_thrust_falls_at_g():
    """Half a second only: this env TERMINATES on ground contact rather than
    clamping, and the default spawn is just 2 m up."""
    e = env(n_envs=64, spawn_on_course=False)
    e.reset(); e.v[:] = 0.0
    a = torch.zeros(64, 4); a[:, 3] = -1.0
    for _ in range(50):                # 0.5 s, ~1.2 m of fall
        e.step(a)
    assert float(e.v[:, 2].mean()) == pytest.approx(0.5 * G, rel=0.1)


def test_hitting_the_ground_terminates_the_episode():
    """Below the pad is a crash, not a clamped position."""
    e = env(n_envs=32, spawn_on_course=False)
    e.reset(); e.v[:] = 0.0
    a = torch.zeros(32, 4); a[:, 3] = -1.0
    crashed = torch.zeros(32, dtype=torch.bool)
    for _ in range(120):               # long enough to reach the floor
        _, _, _, info = e.step(a)
        crashed |= info["crashed"]
    assert bool(crashed.all()), "free fall into the floor must terminate"


def test_full_thrust_climbs():
    e = env(n_envs=64, spawn_on_course=False)
    e.reset(); e.v[:] = 0.0
    a = torch.zeros(64, 4); a[:, 3] = 1.0
    for _ in range(100):
        e.step(a)
    assert float(e.v[:, 2].mean()) < 0.0        # NED: negative is upward


def test_commanded_body_rate_is_tracked():
    e = env(n_envs=16, spawn_on_course=False, tau_rate=0.02)
    e.reset()
    a = torch.zeros(16, 4); a[:, 0] = 0.5
    for _ in range(100):
        e.step(a)
    assert float(e.w[:, 0].mean()) == pytest.approx(0.5 * e.max_rate, rel=0.1)


def test_quaternions_stay_normalised():
    e = env(n_envs=128)
    e.reset()
    g = torch.Generator().manual_seed(0)
    for _ in range(600):
        e.step(torch.rand(128, 4, generator=g) * 2 - 1)
    assert torch.allclose(e.q.norm(dim=-1), torch.ones(128), atol=1e-5)


def test_state_stays_finite_under_random_input():
    e = env(n_envs=256)
    e.reset()
    g = torch.Generator().manual_seed(1)
    for _ in range(500):
        o, r, d, i = e.step(torch.rand(256, 4, generator=g) * 2 - 1)
        assert torch.isfinite(o).all() and torch.isfinite(r).all()


@pytest.mark.parametrize("bad", [1e9, -1e9])
def test_enormous_actions_do_not_poison_observations(bad):
    e = env(n_envs=32)
    e.reset()
    a = torch.full((32, 4), bad)
    for _ in range(5):
        o, r, d, i = e.step(a)
    assert torch.isfinite(o).all()


# ----------------------------------------------------------- quaternion math
def test_quat_mul_identity():
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(quat_mul(q, ident), q, atol=1e-6)


def test_quat_to_rot_is_orthonormal():
    g = torch.Generator().manual_seed(2)
    q = torch.rand(64, 4, generator=g) * 2 - 1
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rot(q)
    assert torch.allclose(R @ R.transpose(-1, -2),
                          torch.eye(3).expand(64, 3, 3), atol=1e-5)


def test_quat_exp_of_zero_is_identity():
    out = quat_exp(torch.zeros(8, 3))
    want = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(8, 4)
    assert torch.allclose(out, want, atol=1e-7)


# -------------------------------------------------------- course + geometry
@pytest.mark.parametrize("course", ["demo", "aigp"])
def test_arena_contains_every_gate(course):
    """The costliest bug in this project: a bound smaller than the course."""
    e = env(course=course)
    assert float(e.gates.norm(dim=-1).max()) < e.arena_r


@pytest.mark.parametrize("course", ["demo", "aigp"])
def test_nothing_dies_on_spawn(course):
    e = env(n_envs=2000, course=course, randomize=True)
    e.reset()
    _, _, _, info = e.step(torch.zeros(2000, 4))
    assert int(info["crashed"].sum()) == 0


@pytest.mark.parametrize("course", ["demo", "aigp"])
def test_spawn_covers_the_whole_course(course):
    """Every gate must be a start point, or late gates are never explored."""
    e = env(n_envs=3000, course=course)
    e.reset()
    assert len(set(e.target.tolist())) == e.n_gates


def test_gate_normals_are_unit_length():
    e = env(course="aigp")
    assert torch.allclose(e.normals.norm(dim=-1),
                          torch.ones(e.n_gates), atol=1e-5)


def test_abort_distance_clears_the_longest_leg():
    """Otherwise an agent correctly flying a 59 m leg is killed halfway."""
    e = env(course="aigp")
    legs = (e.gates - torch.roll(e.gates, 1, 0)).norm(dim=-1)[1:]
    assert e.max_leg + 25.0 > float(legs.max())


# ------------------------------------------------------- gate crossing logic
def test_flying_through_a_gate_is_counted():
    e = env(n_envs=1, course="demo")
    e.reset()
    g, n = e.gates[0], e.normals[0]
    e.target[:] = 0
    e.p[0] = g - n * 0.6
    e.v[0] = n * 20.0
    e.prev_dist[0] = 0.6
    hit = False
    for _ in range(12):
        _, _, _, info = e.step(torch.zeros(1, 4))
        hit |= bool(info["through"][0])
    assert hit


def test_passing_wide_of_the_opening_is_not_counted():
    """Crossing the gate PLANE outside its radius is a miss, not a pass."""
    e = env(n_envs=1, course="demo")
    e.reset()
    g, n = e.gates[0], e.normals[0]
    side = torch.tensor([-n[1], n[0], 0.0])
    side = side / side.norm().clamp_min(1e-6)
    e.target[:] = 0
    e.p[0] = g - n * 0.6 + side * 6.0
    e.v[0] = n * 20.0
    e.prev_dist[0] = (g - e.p[0]).norm()
    for _ in range(12):
        _, _, _, info = e.step(torch.zeros(1, 4))
        assert not bool(info["through"][0])


def test_target_advances_after_a_pass():
    e = env(n_envs=1, course="demo")
    e.reset()
    g, n = e.gates[0], e.normals[0]
    e.target[:] = 0
    e.p[0] = g - n * 0.6
    e.v[0] = n * 20.0
    e.prev_dist[0] = 0.6
    for _ in range(12):
        _, _, _, info = e.step(torch.zeros(1, 4))
        if bool(info["through"][0]):
            break
    assert int(e.target[0]) == 1


# ------------------------------------------------------------------ shapes
def test_observation_and_action_shapes():
    e = env(n_envs=64)
    o = e.reset()
    assert o.shape == (64, e.obs_dim)
    o, r, d, i = e.step(torch.zeros(64, e.act_dim))
    assert o.shape == (64, e.obs_dim)
    assert r.shape == (64,) and d.shape == (64,)


def test_reset_mask_only_touches_selected_envs():
    e = env(n_envs=32)
    e.reset()
    keep = e.p[16:].clone()
    m = torch.zeros(32, dtype=torch.bool); m[:16] = True
    e.reset(m)
    assert torch.allclose(e.p[16:], keep)


def test_determinism_for_a_fixed_seed():
    def run():
        e = env(n_envs=32, seed=7)
        o = e.reset()
        for _ in range(40):
            o, r, _, _ = e.step(torch.full((32, 4), 0.1))
        return o, r
    o1, r1 = run()
    o2, r2 = run()
    assert torch.allclose(o1, o2) and torch.allclose(r1, r2)


# ------------------------------------------------------------- performance
def test_throughput_is_worth_having():
    """The whole premise is that this is fast enough without a GPU."""
    e = env(n_envs=2048, course="aigp")
    e.reset()
    a = torch.zeros(2048, 4)
    for _ in range(5):
        e.step(a)
    t0 = time.perf_counter()
    for _ in range(100):
        e.step(a)
    sps = 2048 * 100 / (time.perf_counter() - t0)
    print(f"\n  {sps:,.0f} env-steps/s on CPU")
    assert sps > 20_000, f"only {sps:,.0f} steps/s"
