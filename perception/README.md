# Ring Gate Detector

Monocular gate detection at **1.5 ms/frame** (measured in `test_ring_gate.py`;
0.65 ms on downscaled input in the full stack), with no learned model.

## The idea

Gates are orange. So is the wall behind them, and so are the markings printed
on the gate frame itself. **Colour cannot separate those**, and a colour-only
detector in this stack repeatedly locked onto the white checker markings — a
failure that would fly the drone into the gate it thought it had found.

What makes a gate a gate is that it is a **ring**: a closed contour containing
a hole, where the hole is a large fraction of its parent's area. That is a
topological property, not a chromatic one, and `cv2.RETR_CCOMP` gives the
parent/child hierarchy needed to test it directly.

Measured across 772 contours in real footage:

| | median hole/parent area ratio |
|---|---|
| real gate openings | **0.340** |
| printed markings | **0.015** |

A threshold of 0.20 separates them cleanly. The solid wall is rejected for
having no hole at all.

The test is applied **during** candidate selection rather than as a filter
afterwards — otherwise a marking can win the "largest contour" race and
suppress the real gate before the check ever runs.

## Tests

`pytest test_ring_gate.py -v` — 15 tests. Beyond the happy path they assert
the three false positives that actually occurred: a solid orange wall, printed
markings on the frame, and a ring whose hole is too small for its frame. Plus
frame-edge clipping, random noise, odd image sizes, all-black/all-white input,
and a timing budget tied to the 30 fps camera.
