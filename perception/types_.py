"""Types the gate detector needs, extracted so this directory runs on its own.

Verbatim subset of apex/common.py and apex/config.py from the full stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateDetection:
    """One detected gate in an image."""
    corners_px: np.ndarray   # 4x2, inner-square corners TL,TR,BR,BL
    confidence: float
    t_frame: float
    reproj_error_px: float = float("nan")    # filled by PnP
    p_gate_body: np.ndarray | None = None    # gate centre in body FRD [m]
    normal_body: np.ndarray | None = None    # through-direction in body FRD
    range_m: float = float("nan")


@dataclass
class PerceptionConfig:
    """Detector and PnP acceptance thresholds.

    The PnP gates are not decoration. A gate seen edge-on or from 30 m away
    produces a geometrically valid but useless pose, and feeding those to the
    estimator is worse than feeding it nothing.
    """
    min_quad_area_px: float = 120.0
    max_quad_area_frac: float = 0.65
    subpix_refine: bool = True
    max_reproj_error_px: float = 4.0
    min_range_m: float = 1.2           # PnP degenerates when too close
    max_range_m: float = 35.0
    max_view_angle_deg: float = 65.0   # too oblique -> reject
