"""RGB-only segmentation and depth backprojection; no simulator object IDs."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray  # OpenGL depth buffer in [0, 1], NOT metres
    view: np.ndarray
    projection: np.ndarray


@dataclass
class Detection:
    center: np.ndarray
    pixels: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def backproject(observation: Observation, rows, cols) -> np.ndarray:
    """Invert the actual OpenGL projection, including non-linear depth."""
    h, w = observation.depth.shape
    clip = np.column_stack((
        2.0 * (np.asarray(cols) + 0.5) / w - 1.0,
        1.0 - 2.0 * (np.asarray(rows) + 0.5) / h,
        2.0 * observation.depth[rows, cols] - 1.0,
        np.ones(len(rows)),
    ))
    inverse = np.linalg.inv(observation.projection @ observation.view)
    points = (inverse @ clip.T).T
    return points[:, :3] / points[:, 3:4]


def detect_cube(observation: Observation, side: float = 0.04) -> Detection | None:
    """Known red 4-cm cube baseline; assumes an approximately upright cube.

    The 3D bounds of visible pixels estimate horizontal position. The highest
    surface estimates the cube top. Side length is a declared object prior.
    """
    hsv = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168))
    mask &= (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 45)
    mask &= observation.depth < 0.999
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        return None
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[label, cv2.CC_STAT_AREA] < 12:
        return None
    mask = labels == label
    rows, cols = np.nonzero(mask)
    points = backproject(observation, rows, cols)
    lo, hi = np.percentile(points, [3, 97], axis=0)
    center = (lo + hi) / 2
    center[2] = hi[2] - side / 2
    return Detection(center, len(rows), tuple(int(x) for x in stats[label, :4]), mask)
