"""Draw what the policy actually does -- the course, and drones flying it.

Numbers in a training log tell you a policy improved. They do not tell you
whether it learned to fly or learned to exploit the reward. A picture does.

    python -m apex.rl.viz --ckpt models/pilot.pt --out flight.png
    python -m apex.rl.viz --ckpt models/pilot.pt --gif flight.gif
    python -m apex.rl.viz --random --out untrained.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from vec_quad import VecQuad
from ppo import ActorCritic


def gate_square(centre, normal, half=0.75):
    """Corners of a gate, drawn as a square perpendicular to its normal."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(n @ tmp) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, tmp); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u)
    c = np.asarray(centre)
    pts = [c + half*(u+v), c + half*(u-v), c + half*(-u-v), c + half*(-u+v)]
    pts.append(pts[0])
    return np.array(pts)


def rollout(ckpt: str | None, n_drones: int, steps: int, seed: int,
            course: str = "demo"):
    env = VecQuad(n_envs=n_drones, max_steps=steps + 10, seed=seed,
                  randomize=True, course=course)
    net = None
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        net = ActorCritic(blob["obs_dim"], blob["act_dim"], blob["hidden"])
        net.load_state_dict(blob["model"]); net.eval()

    obs = env.reset()
    traj = np.zeros((steps, n_drones, 3), dtype=np.float32)
    alive = np.ones((steps, n_drones), dtype=bool)
    gates_hit = np.zeros(n_drones, dtype=int)
    live = torch.ones(n_drones, dtype=torch.bool)
    with torch.no_grad():
        for t in range(steps):
            if net is None:
                a = torch.zeros(n_drones, 4).uniform_(-0.4, 0.4)
            else:
                a = net.dist(obs).mean          # deterministic, no exploration
            traj[t] = env.p.numpy()
            alive[t] = live.numpy()
            obs, rew, done, info = env.step(a)
            gates_hit += (info["through"] & live).numpy().astype(int)
            live &= ~done                        # freeze the line at first death
    return env, traj, alive, gates_hit


def draw_course(ax, env):
    g = env.gates.numpy(); n = env.normals.numpy()
    for i, (c, nn) in enumerate(zip(g, n)):
        sq = gate_square(c, nn)
        # NED -> plot as (x, y, altitude)
        ax.plot(sq[:, 0], sq[:, 1], -sq[:, 2], color="#ff6b35", lw=2.2)
        ax.text(c[0], c[1], -c[2] + 1.2, str(i), color="#ff6b35",
                fontsize=9, fontweight="bold", ha="center")
    ax.plot(g[:, 0], g[:, 1], -g[:, 2], ls=":", color="#999", lw=1.0)


def style(ax, env, title):
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    g = env.gates.numpy()
    ax.set_xlim(g[:, 0].min() - 12, g[:, 0].max() + 12)
    ax.set_ylim(g[:, 1].min() - 12, g[:, 1].max() + 12)
    ax.set_zlim(0, max(16.0, float((-g[:, 2]).max()) + 6))
    ax.view_init(elev=26, azim=-58)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--random", action="store_true", help="untrained policy")
    ap.add_argument("--drones", type=int, default=40)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default="flight.png")
    ap.add_argument("--gif", type=str, default=None)
    ap.add_argument("--course", type=str, default="demo",
                    choices=["demo", "aigp"])
    args = ap.parse_args()

    ckpt = None if args.random else args.ckpt
    env, traj, alive, hits = rollout(ckpt, args.drones, args.steps,
                                     args.seed, args.course)
    tag = "untrained (random actions)" if ckpt is None else args.ckpt
    print(f"{args.drones} drones, {args.steps} steps, policy: {tag}")
    print(f"gates passed -- mean {hits.mean():.2f}, best {hits.max()}, "
          f"{int((hits > 0).sum())}/{args.drones} passed at least one")

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw_course(ax, env)
    order = np.argsort(hits)
    for k in order:
        m = alive[:, k]
        if m.sum() < 2:
            continue
        frac = hits[k] / max(hits.max(), 1)
        col = plt.cm.viridis(0.15 + 0.85 * frac)
        ax.plot(traj[m, k, 0], traj[m, k, 1], -traj[m, k, 2],
                color=col, lw=1.0, alpha=0.75)
    style(ax, env, f"{args.drones} drones  |  best {hits.max()} gates  |  "
                   f"mean {hits.mean():.2f}  |  {tag}")
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    if args.gif:
        fig2 = plt.figure(figsize=(11, 8))
        ax2 = fig2.add_subplot(111, projection="3d")
        stride = max(1, args.steps // 220)
        frames = range(1, args.steps, stride)

        def update(t):
            ax2.clear(); draw_course(ax2, env)
            lo = max(0, t - 160)
            for k in range(args.drones):
                m = alive[lo:t, k]
                if m.sum() < 2:
                    continue
                seg = traj[lo:t, k][m]
                ax2.plot(seg[:, 0], seg[:, 1], -seg[:, 2],
                         color=plt.cm.viridis(0.2 + 0.8*k/args.drones),
                         lw=0.9, alpha=0.65)
                ax2.scatter(seg[-1, 0], seg[-1, 1], -seg[-1, 2], s=9,
                            color="#ffffff", edgecolors="#333", linewidths=0.3)
            style(ax2, env, f"t = {t*env.dt:5.2f} s   |   {tag}")

        anim = FuncAnimation(fig2, update, frames=frames, interval=45)
        anim.save(args.gif, writer=PillowWriter(fps=22))
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
