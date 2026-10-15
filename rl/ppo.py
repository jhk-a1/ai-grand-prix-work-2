"""PPO over thousands of parallel quadrotors.

Deliberately small and readable -- the point is to watch the thing learn, not
to win a benchmark. Everything is on-policy and in one file: rollout, GAE,
clipped surrogate, and a progress line you can actually read.

    python -m apex.rl.ppo --envs 2048 --iters 400
    python -m apex.rl.ppo --envs 4096 --iters 1000 --save models/pilot.pt
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from vec_quad import VecQuad


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        def trunk():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh())
        self.pi_trunk, self.v_trunk = trunk(), trunk()
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        # Start near-deterministic and fairly quiet: a loud initial policy on a
        # quadrotor just tumbles and never sees a gate.
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.9))
        nn.init.orthogonal_(self.mu.weight, 0.01); nn.init.zeros_(self.mu.bias)

    def dist(self, obs):
        mu = self.mu(self.pi_trunk(obs))
        return torch.distributions.Normal(mu, self.log_std.exp())

    def value(self, obs):
        return self.v(self.v_trunk(obs)).squeeze(-1)


def train(args) -> None:
    dev = torch.device(args.device)
    torch.set_num_threads(args.threads)
    env = VecQuad(n_envs=args.envs, device=args.device, seed=args.seed,
                  randomize=not args.no_randomize, course=args.course,
                  max_steps=args.max_steps)
    net = ActorCritic(env.obs_dim, env.act_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs = env.reset()
    T, N = args.horizon, args.envs
    print(f"[ppo] {N} parallel drones, {env.n_gates}-gate course, "
          f"horizon {T} -> {N*T:,} transitions/iter, device={args.device}")
    print(f"[ppo] domain randomisation: "
          f"{'ON (T/W 2-5, tau 0.03-0.12, drag 0-0.03)' if not args.no_randomize else 'OFF'}\n")
    print(f"{'iter':>5}{'reward':>10}{'gates/ep':>10}{'best':>7}"
          f"{'crash%':>8}{'steps/s':>10}{'elapsed':>9}")

    ep_gates = torch.zeros(N, device=dev)      # gates passed in current life
    gate_hist, best_gates = [], 0.0
    t0 = time.time()

    for it in range(args.iters):
        O = torch.zeros(T, N, env.obs_dim, device=dev)
        A = torch.zeros(T, N, env.act_dim, device=dev)
        LP = torch.zeros(T, N, device=dev)
        R = torch.zeros(T, N, device=dev)
        D = torch.zeros(T, N, device=dev)
        V = torch.zeros(T, N, device=dev)
        n_crash = 0

        with torch.no_grad():
            for t in range(T):
                d = net.dist(obs)
                a = d.sample()
                O[t], A[t] = obs, a
                LP[t] = d.log_prob(a).sum(-1)
                V[t] = net.value(obs)
                obs, rew, done, info = env.step(a)
                R[t], D[t] = rew, done.float()
                ep_gates += info["through"].float()
                n_crash += int(info["crashed"].sum())
                if bool(done.any()):
                    gate_hist.append(float(ep_gates[done].mean()))
                    ep_gates[done] = 0.0
            last_v = net.value(obs)

        # GAE
        adv = torch.zeros_like(R)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + args.gamma * nv * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        b_o = O.reshape(-1, env.obs_dim); b_a = A.reshape(-1, env.act_dim)
        b_lp, b_adv, b_ret = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        idx_all = torch.randperm(b_o.shape[0], device=dev)
        mb = max(1, b_o.shape[0] // args.minibatches)

        for _ in range(args.epochs):
            for s in range(0, b_o.shape[0], mb):
                i = idx_all[s:s + mb]
                d = net.dist(b_o[i])
                lp = d.log_prob(b_a[i]).sum(-1)
                ratio = (lp - b_lp[i]).exp()
                l1 = ratio * b_adv[i]
                l2 = ratio.clamp(1 - args.clip, 1 + args.clip) * b_adv[i]
                pi_loss = -torch.min(l1, l2).mean()
                v_loss = (net.value(b_o[i]) - b_ret[i]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + args.vf * v_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if gate_hist:
            recent = sum(gate_hist[-50:]) / len(gate_hist[-50:])
            best_gates = max(best_gates, recent)
        else:
            recent = 0.0
        el = time.time() - t0
        if it % args.log_every == 0 or it == args.iters - 1:
            print(f"{it:>5}{float(R.mean()):>10.3f}{recent:>10.2f}"
                  f"{best_gates:>7.2f}{100*n_crash/(N*T):>8.2f}"
                  f"{int(N*T/(el/(it+1))):>10,}{el:>8.0f}s")

    if args.save:
        torch.save({"model": net.state_dict(), "obs_dim": env.obs_dim,
                    "act_dim": env.act_dim, "hidden": args.hidden}, args.save)
        print(f"\n[ppo] saved -> {args.save}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--course", type=str, default="demo",
                    choices=["demo", "aigp"])
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
