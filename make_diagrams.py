"""Generate architecture and results figures for the public write-up.

These describe the FULL competition stack, not only the modules whose source
is published. Palette follows the validated categorical set (blue/orange/aqua/
yellow/magenta/violet/red) on a light surface.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# validated categorical palette, light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
VIOLET, RED = "#4a3aa7", "#e34948"
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})


def box(ax, x, y, w, h, label, sub=None, color=BLUE, fill=None, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.8, edgecolor=color,
                                facecolor=fill or SURF, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label,
            ha="center", va="center", fontsize=fs, fontweight="bold",
            color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=fs - 2.6, color=MUTED, zorder=3)


def arrow(ax, p0, p1, color=MUTED, style="-|>", lw=1.6, rad=0.0, label=None,
          lx=0.0, ly=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=7.6, color=MUTED,
                zorder=4, bbox=dict(fc=SURF, ec="none", pad=1.4))


# ===========================================================================
# 1. FULL SYSTEM ARCHITECTURE
# ===========================================================================
# ===========================================================================
# 2. CONTROL LOOP TIMING
# ===========================================================================
def control_loop(path="media/control_loop.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: who runs at what rate -----------------------------------
    ax = axes[0]; ax.set_xlim(0, 100); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.set_title("Rates in the loop", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left")

    rows = [
        ("Control loop  86–90 Hz", 11.6, BLUE, 3.4),
        ("IMU            200 Hz", 5.0, AQUA, 2.5),
        ("Camera          30 Hz", 33.3, ORANGE, 1.6),
        ("Gate detect   1.5 ms", 1.5, ORANGE, 0.7),
    ]
    for label, period, col, y in rows:
        n = int(100 / period) + 1
        for k in range(n):
            x = k * period
            if x > 100:
                break
            ax.add_patch(plt.Rectangle((x, y), min(period * 0.82, 100 - x),
                                       0.42, facecolor=col, alpha=0.85,
                                       edgecolor=SURF, linewidth=1.4))
        ax.text(-1, y + 0.21, label, ha="right", va="center", fontsize=9.3,
                color=INK2, family="DejaVu Sans Mono")
    ax.plot([0, 100], [0.3, 0.3], color=GRID, lw=1.2)
    for t in range(0, 101, 20):
        ax.text(t, 0.12, f"{t}", ha="center", fontsize=8, color=MUTED)
    ax.text(50, -0.12, "milliseconds", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(-30, 102)
    ax.text(50, 4.15,
            "Detection costs 1.5 ms of a 11.6 ms budget — it fits between "
            "control cycles",
            ha="center", fontsize=8.6, color=MUTED, style="italic")

    # --- right: the scheduling bug -------------------------------------
    ax = axes[1]
    segs = ["hold80_l2", "hold80_level", "post_g5_lift", "cruise", "climb"]
    before = [40, 44, 45, 47, 43]
    after = [88, 87, 87, 87, 87]
    x = np.arange(len(segs)); w = 0.38
    ax.bar(x - w/2, before, w, color=RED, label="before fix", zorder=3)
    ax.bar(x + w/2, after, w, color=AQUA, label="after fix", zorder=3)
    ax.axhline(86, color=INK2, ls="--", lw=1.3, zorder=2)
    ax.text(len(segs) - 0.52, 78, "86 Hz target", fontsize=8.4, color=INK2,
            ha="right")
    ax.set_xticks(x); ax.set_xticklabels(segs, fontsize=8.6, rotation=18,
                                         ha="right")
    ax.set_ylabel("command rate (Hz)", fontsize=9.5)
    ax.set_ylim(0, 122)
    ax.set_title("Windows timer + priority fix", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.30,
            "Default Windows timer resolution is 15.6 ms — wider than the "
            "entire 11.6 ms control period.\nThe process was free-riding on "
            "another app holding it at 1 ms, and ran at normal priority.",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
# 3. THE SPEED LIMIT  (delta x = v * delta t)
# ===========================================================================
def speed_limit(path="media/speed_limit.png"):
    gates = [1, 2, 3, 4, 5, 6]
    speed = [4.75, 11.68, 6.94, 15.24, 21.30, 24.41]
    jitter = [0.072, 0.073, 0.071, 0.077, 0.244, 0.189]
    scatter = [s * j for s, j in zip(speed, jitter)]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))

    ax = axes[0]
    v = np.linspace(0, 27, 200)
    ax.fill_between(v, 0, 0.75, color=AQUA, alpha=0.12, zorder=0)
    ax.plot(v, v * 0.07, color=BLUE, lw=2.2, zorder=3,
            label="predicted  δx = v × 0.07 s")
    ax.axhline(0.75, color=INK2, ls="--", lw=1.4, zorder=2)
    ax.text(26.6, 0.86, "gate half-width 0.75 m", fontsize=8.8, color=INK2,
            ha="right")
    ax.scatter(speed, scatter, s=95, color=ORANGE, zorder=4, edgecolors=SURF,
               linewidths=1.6, label="measured, 50 runs")
    for g, s, sc in zip(gates, speed, scatter):
        ax.annotate(f"gate {g}", (s, sc), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.2, color=INK2)
    ax.axvline(10.7, color=RED, ls=":", lw=1.8, zorder=2)
    ax.text(11.2, 4.6, "v_max ≈ 10.7 m/s", fontsize=9, color=RED,
            fontweight="bold")
    ax.set_xlabel("crossing speed (m/s)", fontsize=9.8)
    ax.set_ylabel("along-track position scatter (m)", fontsize=9.8)
    ax.set_title("Timing jitter sets a speed limit", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=GRID, lw=0.7)
    ax.set_xlim(0, 27); ax.set_ylim(0, 6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    note = ("The line assumes the 0.07 s jitter measured on gates 1-4. Gates "
            "5 and 6 sit above it because\ntheir legs generate extra jitter "
            "of their own (0.244 s and 0.189 s), compounding the speed term.")
    ax.text(0.0, -0.165, note, transform=ax.transAxes, fontsize=8.4,
            color=MUTED, va="top")

    ax = axes[1]
    labels = ["passed\nthe gate", "failed\nthe gate"]
    vals = [14.55, 15.82]
    bars = ax.bar(labels, vals, width=0.5, color=[AQUA, RED], zorder=3)
    for b, v_ in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v_ + 0.25, f"{v_:.2f} m/s",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(13.5, 16.6)
    ax.set_ylabel("speed at gate 4 (m/s)", fontsize=9.8)
    ax.set_title("Faster crossings failed more often", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.17,
            "n = 44 runs. The prediction is directional: at these speeds the "
            "loop's own jitter is\nwider than the gate, so the outcome is set "
            "before any steering correction can act.",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
# 4. PERCEPTION: why topology beats colour  + latency
# ===========================================================================
def perception(path="media/perception.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    cats = ["gate\nopening", "printed\nmarkings"]
    vals = [0.340, 0.015]
    bars = ax.bar(cats, vals, width=0.5, color=[AQUA, RED], zorder=3)
    ax.axhline(0.20, color=INK2, ls="--", lw=1.4, zorder=4)
    ax.text(1.42, 0.215, "threshold 0.20", fontsize=8.6, color=INK2,
            ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel("hole / parent area ratio", fontsize=9.6)
    ax.set_title("Topology separates them", fontsize=12, fontweight="bold",
                 color=INK, loc="left")
    ax.set_ylim(0, 0.40)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.18, "measured across 772 contours",
            transform=ax.transAxes, fontsize=8.4, color=MUTED)

    ax = axes[1]
    names = ["ring\ndetector", "CNN\nGPU", "CNN\nCPU"]
    ms = [1.5, 21.6, 437.0]
    cols = [AQUA, BLUE, RED]
    bars = ax.barh(names[::-1], ms[::-1], height=0.52, color=cols[::-1],
                   zorder=3)
    ax.set_xscale("log")
    ax.axvline(33.3, color=INK2, ls="--", lw=1.4, zorder=4)
    ax.text(36, 2.38, "30 fps camera\nbudget 33 ms", fontsize=8.2, color=INK2)
    for b, v in zip(bars, ms[::-1]):
        ax.text(v * 1.18, b.get_y() + b.get_height()/2, f"{v:g} ms",
                va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("inference time per frame (log scale)", fontsize=9.6)
    ax.set_title("Detector cost", fontsize=12, fontweight="bold", color=INK,
                 loc="left")
    ax.set_xlim(0.8, 1600)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
# 5. RL TRAINING + PER-GATE RESULT
# ===========================================================================
def rl_results(path="media/rl_results.png"):
    it = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
          650, 700, 750, 800, 850, 899]
    gates = [0.00, 0.15, 0.13, 0.29, 0.52, 0.92, 1.05, 1.23, 1.34, 1.37,
             1.77, 1.46, 1.34, 1.51, 1.55, 1.71, 1.66, 1.90, 2.22]
    broken = [0.00, 0.02, 0.06, 0.19, 0.32, 0.22, 0.34, 0.29, 0.48, 0.44,
              0.43, 0.49, 0.47, 0.41, 0.52, 0.49, 0.58, 0.53, 0.57]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    ax.plot(it, gates, color=AQUA, lw=2.4, zorder=4,
            label="arena bound correct")
    ax.plot(it, broken, color=RED, lw=2.0, zorder=3, ls="--",
            label="arena bound too small")
    ax.scatter([899], [2.22], s=80, color=AQUA, zorder=5, edgecolors=SURF,
               linewidths=1.6)
    ax.annotate("2.22", (899, 2.22), textcoords="offset points",
                xytext=(-12, 10), fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("training iteration", fontsize=9.8)
    ax.set_ylabel("gates per episode", fontsize=9.8)
    ax.set_title("Same algorithm, one constant different", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.18,
            "A 120 m arena radius on a 243 m course terminated every agent at "
            "gates 7–15 before it\ncould move. 3.2× improvement from fixing a "
            "bound, with no change to the learning.",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="top")

    ax = axes[1]
    g = list(range(1, 16))
    rate = [100, 99, 100, 100, 97, 94, 100, 100, 99, 100, 100, 99, 100, 95, 7]
    cols = [AQUA if r >= 90 else RED for r in rate]
    ax.bar(g, rate, width=0.68, color=cols, zorder=3)
    ax.axhline(90, color=INK2, ls="--", lw=1.2, zorder=4)
    ax.set_xticks(g)
    ax.set_xlabel("gate", fontsize=9.8)
    ax.set_ylabel("success rate (%)", fontsize=9.8)
    ax.set_ylim(0, 145)
    ax.set_title("Per-gate success, trained policy", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("gate 15: the course\nreconstruction puts a 91°\nturn here — a "
                "mapping error,\nnot a flying one",
                xy=(15, 7), xytext=(11.3, 36), fontsize=8.2, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    ax.text(0.0, -0.18,
            "Spawned mid-leg at each gate, 100 agents per gate. Custom "
            "environment, not the competition simulator.",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os
    os.makedirs("media", exist_ok=True)
    control_loop()
    speed_limit()
    perception()
    rl_results()
