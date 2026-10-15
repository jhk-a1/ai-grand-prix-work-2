# AI Grand Prix — Autonomous Drone Racing

Selected modules from an autonomy stack built for the **AI Grand Prix Virtual
Qualifier 2**, a autonomous drone racing competition run on the DCL simulator.
The full stack is ~10,000 lines of Python; this repository contains three
self-contained pieces with their tests, plus the measurements behind them.

![drones flying the course](media/flight.gif)

*22 reinforcement-learning agents flying the reconstructed course in parallel.
Orange squares are gates, white dots are drones, trails show recent path.*

---

## The system

```
                    ┌──────────────┐
  MAVLink 2 / UDP ──►   ESKF       ├──► cascaded controller ──► CTBR @ 86 Hz
   IMU @ 200 Hz     │  15 states   │    pos → att → rates     thrust + body rates
                    └──────▲───────┘
                           │ position / yaw fixes
                    ┌──────┴───────┐
  UDP video 30 Hz ──►  detector    ├──► PnP ──► gate pose, body frame
   640×360          │ ring or CNN  │   IPPE_SQUARE on a known 1.5 m opening
                    └──────────────┘
```

The drone is commanded in **CTBR** — collective thrust plus body rates — which
is the standard interface for agile quadrotor flight and the same interface an
RL policy emits, so a learned controller drops into the same socket as a
classical one.

The hard constraint shaping everything: **the simulator publishes no absolute
position.** Verified by logging every MAVLink message it sends. Position comes
from the gates themselves or not at all.

---

## Results

| measurement | value | where |
|---|---|---|
| Control loop rate | **86–90 Hz** sustained | competition sim |
| Ring detector | **1.5 ms/frame** (0.65 ms downscaled) | measured in tests |
| PnP reprojection error | **0.33 px** median, 94.8% under 2 px | competition sim, 1,254 observations |
| CNN inference, CPU → GPU | **437 ms → 21.6 ms (20×)** | ONNX Runtime, DirectML on a GTX 1650 |
| Gate sighting interval | **0.035 s** median, 98.4% under 0.5 s | competition sim, 6 runs |
| Course reached | **6 of 15 gates** | competition sim |
| RL environment throughput | **383k steps/s** idle, **50k** training | 8-core CPU, no GPU |
| RL per-gate success | **94–100% on 14 of 15 gates** | **custom sim** |
| RL gates per episode | 0.00 → **2.22** | **custom sim** |

**What was measured where.** Everything above the divider ran against the
actual competition simulator. The reinforcement-learning numbers ran in the
custom environment in `rl/` — a dynamics model fitted to logged flights, on a
course reconstructed from a hand-drawn map fused with measured leg lengths and
altitudes. The RL agent receives gate positions directly, not through the
camera. It has never flown the competition simulator.

The scripted stack reached **6 of 15 gates**. No lap was completed.

---

## Modules

### [`estimation/`](estimation) — Error-State Kalman Filter
15-state ESKF with IMU propagation and vision updates, Mahalanobis outlier
gating, and process noise that inflates while vision is absent so the filter's
confidence tracks its real ignorance.
**21 tests.**

### [`perception/`](perception) — Ring Gate Detector
Gate detection by contour *topology* rather than colour. Gates, the wall
behind them, and the markings printed on them are all the same orange; what
separates a gate is that it is a ring with a large hole. Measured across 772
contours: openings 0.340 hole/parent ratio, markings 0.015.
**15 tests.**

### [`rl/`](rl) — Vectorised Quadrotor RL
2,048 quadrotors in parallel in plain PyTorch — no physics engine, no GPU.
PPO with GAE and domain randomisation across a 2.5× thrust-to-weight range,
so the policy must sense and adapt rather than memorise a throttle value.
**27 tests.**

![course](media/course.png)

*Course reconstruction. Filled green gates were measured from 50 logged runs;
hollow red are estimated from a hand-drawn map — the drone never flew past
gate 6, so everything beyond it is inference.*

---

## Two findings worth the space

**Timing jitter sets a speed limit.** A gate crossing is a level-set crossing,
so by the implicit function theorem the sensitivity of crossing time to any
trajectory parameter gives

```
δx = v · δt        position error = speed × timing jitter
```

Loop jitter is pinned near 0.07 s, so positional scatter scales linearly with
speed. For a 1.5 m gate that implies a ceiling of `v_max ≈ 10 m/s` for
open-loop timed flight. Checked against 50 runs: drones that **failed** a gate
crossed it **1.27 m/s faster** than those that passed. The path was flying
gates 4 and 5 at 15 and 21 m/s — well past the limit, which is why that
section behaved like a coin flip regardless of tuning.

**Three failures that looked like bad ideas were broken plumbing.** A
`SetPriorityClass` call silently failing on a truncated handle; a vision
hand-off running with its camera handle unset, so it took the "no gate
visible" branch every cycle and flew dead straight; an arena bound sized for a
different course terminating every agent before it could move. Each presented
as *"this approach does not work."* Each was *"this code never ran."*

---

## Running it

```bash
pip install -r requirements.txt

cd estimation && pytest test_eskf.py -v
cd ../perception && pytest test_ring_gate.py -v
cd ../rl && pytest test_vec_quad.py -v -s

# train a policy from scratch (~20 min on CPU)
cd rl && python ppo.py --course aigp --envs 2048 --iters 900 \
    --gamma 0.995 --max-steps 900 --save pilot.pt
python viz.py --course aigp --ckpt pilot.pt --gif flight.gif
```

Each directory is standalone — no cross-imports, no package install.

---

## Not included

The DCL simulator, the competition technical specification, and the raw flight
logs are licensed material and are not mine to redistribute. Nothing here
depends on them: the three modules run on synthetic inputs and the tests are
self-contained.
