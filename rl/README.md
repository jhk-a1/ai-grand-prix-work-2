# Vectorised Quadrotor RL

Thousands of quadrotors flying a gate course in parallel, in **plain PyTorch —
no physics engine, no GPU required**.

## Why no simulator

Isaac Sim earns its RTX requirement on rendering and contact-rich physics:
articulated arms, grasping, cloth. A quadrotor racing through gates has none
of that. Its state is 13 floats, its dynamics are about ten lines of
arithmetic, and "did it pass the gate" is a signed plane crossing plus a
radius test. All of that batches perfectly, so N drones cost almost exactly
what one costs.

Measured here: **383,000 env-steps/s** with no policy attached, and
**~50,000 env-steps/s** during live PPO training (which includes the network
forward pass) — on an 8-core CPU.

## Domain randomisation

Every agent flies a different aircraft:

| parameter | range | why |
|---|---|---|
| thrust-to-weight | 2.0 – 5.0 | the real value is genuinely uncertain |
| rotor time constant | 0.03 – 0.12 s | never measured directly |
| drag | 0.000 – 0.030 | never measured at all |

The policy therefore cannot learn "command 0.30 throttle"; it has to learn to
sense the response and adjust. That is the property a hand-tuned path never
has, and it is why an imperfect dynamics model is survivable — you do not need
a *correct* model, you need a distribution that **contains** the truth.

## Files

| file | contents |
|---|---|
| `vec_quad.py` | batched dynamics, gate crossing, reward, termination |
| `ppo.py` | PPO with GAE, clipped surrogate, entropy bonus |
| `course_aigp.py` | real course: sketched topology fused with measured legs |
| `viz.py` | 3D trajectory plots and animations |

```bash
python ppo.py --course aigp --envs 2048 --iters 900 --gamma 0.995 \
    --max-steps 900 --save pilot.pt
python viz.py --course aigp --ckpt pilot.pt --gif flight.gif
```

## Tests

`pytest test_vec_quad.py -v` — 27 tests on physics (hover equilibrium, free
fall at g, rate tracking, quaternion normalisation), quaternion maths,
course geometry, gate-crossing logic including the wide-miss case, batching
semantics, determinism and throughput.

Several exist because of bugs that cost real time. The worst: an arena radius
of 120 m, sized for a small test course, silently terminated every agent at
gates 7–15 of a 243 m course **before it could move**. Gates/episode sat at
0.69 and it looked like the policy could not fly those gates. With the bound
corrected it reached 2.22 — a 3.2× improvement from a constant, with no
change to the algorithm.

A broken RL environment does not raise. It trains.
