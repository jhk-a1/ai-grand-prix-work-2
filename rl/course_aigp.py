"""The real AIGP course: sketched topology fused with measured geometry.

Two sources, each used only for what it is good for.

  * The operator's hand-drawn map gives the ORDER of the gates and the
    DIRECTION of each leg -- which way the track turns. It is explicitly not
    to scale: checking its leg lengths against the sim shows the implied
    metres-per-pixel varying 7.2x from leg to leg, so any distance read off it
    is meaningless.

  * The simulator logs give the MAGNITUDES. Altitude and speed are recorded at
    every gate crossing across 50 runs, and a leg length follows from the mean
    speed across it times the elapsed time. Those are real measurements with
    real error bars.

Measured, gates 1-6 (n=50 runs, sd in brackets):

    gate   alt m        speed m/s      leg -> next
    1      3.54 (0.29)   4.75 (0.37)   10.8 m
    2      8.30 (0.31)  11.68 (0.30)    6.6 m
    3      9.70 (0.32)   6.94 (0.53)   14.1 m
    4     10.10 (0.77)  15.24 (2.51)   41.3 m
    5      7.99 (1.05)  21.30 (1.82)   59.3 m
    6      6.00 (0.86)  24.41 (0.56)   -

Past gate 6 nothing has ever been measured -- the drone has not been there.
Those legs are scaled by the median measured factor and their altitudes come
from the sketch arrows, and they are flagged as estimates in `measured`.
"""

from __future__ import annotations

import numpy as np
import torch

# Plan-view pixel positions from the sketch, in draw order S -> E.
_SKETCH_PX = [
    (165, 470), (205, 480), (270, 465), (320, 290), (375, 215),
    (480, 200), (545, 305), (580, 400), (625, 480), (715, 530),
    (805, 410), (825, 295), (830, 150), (905, 105), (1000, 275),
]

# Measured leg lengths, metres. Index i is the leg from gate i to i+1.
_MEASURED_LEG_M = {0: 10.8, 1: 6.6, 2: 14.1, 3: 41.3, 4: 59.3}

# Measured altitude at each gate, metres. Sim gate g is sketch index g-1.
_MEASURED_ALT_M = {0: 3.54, 1: 8.30, 2: 9.70, 3: 10.10, 4: 7.99, 5: 6.00}

# Altitude past gate 6 -- read off the red arrows only, never measured.
_SKETCH_ALT_M = {6: 8.0, 7: 11.0, 8: 12.0, 9: 10.0, 10: 12.0,
                 11: 14.0, 12: 11.0, 13: 8.0, 14: 6.0}


def build(device="cpu"):
    """Return (gates [G,3] NED, normals [G,3], measured [G] bool)."""
    px = np.array(_SKETCH_PX, float)
    d_px = np.diff(px, axis=0)
    leg_px = np.linalg.norm(d_px, axis=1)
    unit = d_px / leg_px[:, None]                     # plan-view direction

    scale = np.median([_MEASURED_LEG_M[i] / leg_px[i] for i in _MEASURED_LEG_M])

    pos, alt, meas = [np.zeros(2)], [], []
    for i in range(len(px)):
        a = _MEASURED_ALT_M.get(i)
        meas.append(a is not None)
        alt.append(a if a is not None else _SKETCH_ALT_M.get(i, 8.0))
    for i, u in enumerate(unit):
        L = _MEASURED_LEG_M.get(i, leg_px[i] * scale)
        pos.append(pos[-1] + u * L)
    P = np.array(pos)

    # Image axes -> NED. Image x is east, image y runs DOWN the page so it
    # negates into north. z is DOWN, hence -altitude.
    g = np.stack([-P[:, 1], P[:, 0], -np.array(alt)], axis=1)
    g = torch.tensor(g, dtype=torch.float32, device=device)

    prev = torch.cat([g[:1] - (g[1:2] - g[:1]), g[:-1]], 0)
    nxt = torch.cat([g[1:], g[-1:] + (g[-1] - g[-2])[None]], 0)
    n = nxt - prev
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return g, n, torch.tensor(meas, device=device)


def aigp_course(device):
    g, n, _ = build(device)
    return g, n


def summary() -> str:
    g, _, m = build("cpu")
    d = (g[1:] - g[:-1]).norm(dim=-1)
    return (f"{g.shape[0]} gates, {float(d.sum()):.0f} m of track "
            f"({int(m.sum())} gates measured, {int((~m).sum())} estimated), "
            f"legs {float(d.min()):.0f}-{float(d.max()):.0f} m, "
            f"altitude {float(-g[:,2].max()):.1f}-{float(-g[:,2].min()):.1f} m")


if __name__ == "__main__":
    print(summary())
