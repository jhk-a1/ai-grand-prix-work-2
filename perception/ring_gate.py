"""Gate detector keyed on the measured colour AND on the gate being a ring.

Needs no training and is deterministic. Two measured facts do the work:

1.  **The gates are emissive.** 3.26 million saturated pixels sampled from real
    drone frames give hue 1-13, saturation 180-255, value 222-255 — RGB
    (249, 61, 11). The high value is the discriminator: in a dark hangar
    almost nothing else is simultaneously that saturated *and* that bright.

2.  **A gate is a hollow ring; a wall is not.** The big orange banner that the
    CNN fired on at 0.47 is the same colour as a gate, so colour alone cannot
    separate them — but the banner is solid and a gate has an opening. Keeping
    only colour blobs that *enclose a hole* rejects it on topology. The hole is
    also exactly the aperture the drone must fly through, so it is what we
    steer at.

The white "AI-GP" lettering breaks the orange ring into arcs, so the mask is
closed morphologically before contouring — that welds the arcs back into a
ring without merging separate gates.

Speed matters: this runs in the same process as an 87 Hz control loop. Work is
done at half resolution and only the closest gate is fully evaluated, which
took it from 24.6 ms to a few ms per frame. Coordinates are returned in the
caller's full-resolution frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from types_ import GateDetection, PerceptionConfig

# Measured on real drone frames; slightly wider than p5..p95 so a dimmer,
# more distant gate still registers.
_HUE_HI = 15                      # OpenCV hue, plus the wrap band below
_HUE_WRAP_LO = 172
_SAT_MIN = 150
_VAL_MIN = 180

_PROC_W, _PROC_H = 320, 180       # half resolution: 4x fewer pixels
_CLOSE_K = 5                      # welds lettering-broken arcs AND swallows
                                  # the small white checker marks (at half res)
_MIN_RING_AREA = 15.0             # px^2 at half res == 60 at full
_MAX_ASPECT = 3.2
_MIN_FILL = 0.45                  # hole area / its own bounding quad area

# The decisive test. The white checker marks and lettering are non-orange
# regions INSIDE the frame, so contour hierarchy sees them as holes with an
# orange parent -- the same signature as a real opening. Steering at one would
# fly the drone into the frame.
#
# They separate cleanly on the ratio of hole area to the parent contour's
# area, measured over 772 holes in real frames:
#     real openings   median 0.340
#     checker marks   median 0.015
# Geometrically a frame of thickness t around an opening of side s gives
# s^2/(s+2t)^2, i.e. 0.25..0.7 for any sane gate. A mark inside the bar cannot
# reach that. The ratio is scale-invariant, so it holds at any distance.
_MIN_HOLE_RATIO = 0.20


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL by angle about the centroid."""
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


class RingGateDetector:
    """Finds gate openings by colour + hole topology. No model file needed."""

    def __init__(self, cfg: PerceptionConfig, only_closest: bool = True):
        self.cfg = cfg
        self.only_closest = only_closest
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (_CLOSE_K, _CLOSE_K))
        self._lo1 = np.array([0, _SAT_MIN, _VAL_MIN], np.uint8)
        self._hi1 = np.array([_HUE_HI, 255, 255], np.uint8)
        self._lo2 = np.array([_HUE_WRAP_LO, _SAT_MIN, _VAL_MIN], np.uint8)
        self._hi2 = np.array([179, 255, 255], np.uint8)
        self._small = np.empty((_PROC_H, _PROC_W, 3), np.uint8)

    def _mask(self, small_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self._lo1, self._hi1)
        # hue wraps at 180, so the red end needs a second band
        cv2.bitwise_or(m, cv2.inRange(hsv, self._lo2, self._hi2), dst=m)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k)

    def detect(self, frame_bgr: np.ndarray,
               t_frame: float) -> list[GateDetection]:
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            return []
        h0, w0 = frame_bgr.shape[:2]
        if h0 < 4 or w0 < 4:
            return []

        cv2.resize(frame_bgr, (_PROC_W, _PROC_H), dst=self._small,
                   interpolation=cv2.INTER_AREA)
        mask = self._mask(self._small)

        # RETR_CCOMP gives a two-level hierarchy: outer blobs and their holes.
        # A hole whose parent is orange IS a gate opening.
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or not cnts:
            return []
        hier = hier[0]

        # Rank holes by area first and only fully evaluate the largest, which
        # is the nearest gate. Everything else is wasted work in a control
        # loop that has 11 ms per cycle.
        # The marking test happens HERE, during selection, not after. If it
        # ran later, a marking that happened to be the largest hole would be
        # picked, then rejected, and the real gate never considered.
        cands = []
        for i, cnt in enumerate(cnts):
            if hier[i][3] < 0:          # no parent -> outer blob, not a hole
                continue
            a = float(cv2.contourArea(cnt))
            if not (_MIN_RING_AREA <= a <= 0.6 * _PROC_W * _PROC_H):
                continue
            pa = float(cv2.contourArea(cnts[hier[i][3]]))
            if pa <= 0 or a / pa < _MIN_HOLE_RATIO:
                continue                # a marking on the frame, not an opening
            cands.append((a, i, cnt, pa))
        if not cands:
            return []
        cands.sort(key=lambda c: -c[0])
        if self.only_closest:
            cands = cands[:1]
        else:
            cands = cands[:4]

        sx = w0 / _PROC_W
        sy = h0 / _PROC_H
        out: list[GateDetection] = []
        for area, i, cnt, parent_area in cands:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            quad = (approx.reshape(-1, 2).astype(np.float64)
                    if len(approx) == 4
                    else cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64))
            if quad.shape != (4, 2):
                continue
            quad = _order_corners(quad)

            sides = [float(np.linalg.norm(quad[j] - quad[(j + 1) % 4]))
                     for j in range(4)]
            lo = min(sides)
            if lo < 1e-6 or max(sides) / lo > _MAX_ASPECT:
                continue
            quad_area = 0.5 * abs(
                np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) -
                np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))
            if quad_area <= 0:
                continue
            fill = area / quad_area
            if fill < _MIN_FILL:
                continue

            # Confidence from contour areas only -- no per-detection full-frame
            # masks, which is what made the first version slow. The parent blob
            # is orange by construction; what we score is whether the ring
            # around the hole is a sane thickness for a gate frame.
            ring = max(parent_area - area, 0.0)
            ratio = ring / max(area, 1e-6)
            shape_ok = float(np.clip(ratio / 0.6, 0.0, 1.0)) if ratio < 0.6 \
                else float(np.clip(2.5 / max(ratio, 1e-6), 0.0, 1.0))
            conf = float(np.clip(0.5 * fill + 0.5 * shape_ok, 0.0, 1.0))

            quad[:, 0] *= sx
            quad[:, 1] *= sy
            out.append(GateDetection(corners_px=quad, confidence=conf,
                                     t_frame=t_frame))
        return out

    # Present so callers can treat this like the colour detector.
    def autocalibrate(self, frame_bgr: np.ndarray) -> bool:
        return True
