"""Tests for the ring gate detector.

The detector's defining property is topological rather than chromatic. Gates
are orange; so is the wall behind them, and so are the painted markings on the
gate frame itself. Colour alone cannot separate those, and a colour-only
detector in this stack repeatedly locked onto the white checker markings --
which would have flown the drone into the gate it thought it had found.

What distinguishes a real gate is that it is a RING: a closed contour with a
hole, where the hole is a large fraction of its parent's area. Markings have
no hole; the solid wall has no hole; a gate opening measured across 772
contours had a median hole/parent ratio of 0.340 against 0.015 for markings.

    pytest test_ring_gate.py -v
"""

import time

import cv2
import numpy as np
import pytest

from ring_gate import RingGateDetector
from types_ import PerceptionConfig

W, H = 640, 360
GATE_BGR = (11, 61, 249)          # BGR of the measured gate colour RGB(249,61,11)


def blank():
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (60, 50, 45)
    return img


def draw_gate(img, cx, cy, outer, thickness=18, colour=GATE_BGR):
    """A square ring: filled outer square, hole punched in the middle."""
    h = outer // 2
    cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), colour, -1)
    i = h - thickness
    cv2.rectangle(img, (cx - i, cy - i), (cx + i, cy + i), (60, 50, 45), -1)
    return img


def detector(**kw):
    return RingGateDetector(PerceptionConfig(**kw), only_closest=False)


# ------------------------------------------------------------ finds gates
def test_finds_a_single_gate():
    d = detector()
    got = d.detect(draw_gate(blank(), 320, 180, 160), 0.0)
    assert len(got) >= 1


def test_reports_the_opening_centre_not_the_frame_centre():
    d = detector()
    got = d.detect(draw_gate(blank(), 400, 150, 170), 0.0)
    assert got, "no detection"
    c = np.asarray(got[0].corners_px, float).reshape(-1, 2).mean(axis=0)
    assert c[0] == pytest.approx(400, abs=18)
    assert c[1] == pytest.approx(150, abs=18)


def test_finds_several_gates_at_once():
    img = blank()
    draw_gate(img, 150, 180, 110)
    draw_gate(img, 480, 180, 110)
    assert len(detector().detect(img, 0.0)) >= 2


def test_corners_are_four_points_in_image_bounds():
    got = detector().detect(draw_gate(blank(), 320, 180, 160), 0.0)
    q = np.asarray(got[0].corners_px, float).reshape(-1, 2)
    assert q.shape == (4, 2)
    assert (q[:, 0] >= -5).all() and (q[:, 0] <= W + 5).all()
    assert (q[:, 1] >= -5).all() and (q[:, 1] <= H + 5).all()


# ------------------------------------------------ rejects what is not a gate
def test_empty_scene_yields_nothing():
    assert detector().detect(blank(), 0.0) == []


def test_solid_orange_wall_is_rejected():
    """A filled square is the right colour but has no hole -- not a gate."""
    img = blank()
    cv2.rectangle(img, (220, 80), (420, 280), GATE_BGR, -1)
    assert detector().detect(img, 0.0) == []


def test_orange_markings_on_the_frame_are_rejected():
    """Small solid blobs: the exact false positive that crashed the drone."""
    img = blank()
    for x in range(180, 460, 34):
        cv2.rectangle(img, (x, 170), (x + 16, 190), GATE_BGR, -1)
    assert detector().detect(img, 0.0) == []


def test_a_ring_of_the_wrong_colour_is_rejected():
    img = draw_gate(blank(), 320, 180, 160, colour=(230, 120, 20))   # blue-ish
    assert detector().detect(img, 0.0) == []


def test_a_hole_too_small_for_its_frame_is_rejected():
    """Hole/parent ratio is the discriminator; starve it and detection stops."""
    img = draw_gate(blank(), 320, 180, 170, thickness=72)   # tiny opening
    assert detector().detect(img, 0.0) == []


# ---------------------------------------------------------- only_closest
def test_only_closest_returns_the_largest_gate():
    img = blank()
    draw_gate(img, 140, 180, 70)          # far
    draw_gate(img, 450, 180, 200)         # near
    d = RingGateDetector(PerceptionConfig(), only_closest=True)
    got = d.detect(img, 0.0)
    assert len(got) == 1
    c = np.asarray(got[0].corners_px, float).reshape(-1, 2).mean(axis=0)
    assert c[0] > 300, "returned the distant gate instead of the near one"


# ------------------------------------------------------------- robustness
def test_gate_clipped_by_the_frame_edge_does_not_crash():
    for cx in (0, 30, W - 30, W):
        detector().detect(draw_gate(blank(), cx, 180, 150), 0.0)


def test_noise_does_not_crash_or_hallucinate_many_gates():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    assert len(detector().detect(img, 0.0)) < 6


def test_handles_odd_frame_sizes():
    for w, h in ((320, 180), (800, 450), (97, 61)):
        img = np.zeros((h, w, 3), np.uint8); img[:, :] = (60, 50, 45)
        detector().detect(img, 0.0)


def test_all_black_and_all_white_are_safe():
    for fill in (0, 255):
        detector().detect(np.full((H, W, 3), fill, np.uint8), 0.0)


# ------------------------------------------------------------------ speed
def test_runs_fast_enough_for_the_control_loop():
    """Budget: the camera delivers ~30 fps, so a frame must cost << 33 ms.

    Measured on the real stack at 0.65-2 ms. A generous 15 ms ceiling here
    keeps the test meaningful on slow CI without becoming decorative.
    """
    d = detector()
    img = draw_gate(blank(), 320, 180, 160)
    d.detect(img, 0.0)                                  # warm up
    t0 = time.perf_counter()
    for _ in range(50):
        d.detect(img, 0.0)
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"\n  detector: {ms:.2f} ms/frame")
    assert ms < 15.0, f"{ms:.1f} ms/frame is too slow for a 30 fps camera"
