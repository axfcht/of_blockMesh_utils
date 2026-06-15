"""Rotation helpers for 3-D point geometry.

Provides:
- normalize: return the unit vector of a direction vector
- rotate_point: rotate a point around an arbitrary axis using Rodrigues' formula
"""

import math
from collections.abc import Sequence

import numpy as np


def normalize(v: Sequence[float]) -> np.ndarray:
    """Return a unit vector in the direction of *v*.

    Raises
    ------
    ValueError
        If *v* is the zero vector.
    """
    arr = np.asarray(v, dtype=float)
    length = float(np.linalg.norm(arr))
    if length < 1e-15:
        raise ValueError(
            f"axis_dir must not be the zero vector; got {list(v)}"
        )
    return arr / length


def rotate_point(
    p: Sequence[float],
    axis_point: Sequence[float],
    axis_unit: np.ndarray,
    angle_rad: float,
) -> list[float]:
    """Rotate point *p* around the axis defined by *axis_point* and *axis_unit*.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    p:
        The point to rotate (x, y, z).
    axis_point:
        A point on the rotation axis.
    axis_unit:
        **Unit** direction vector of the rotation axis (already normalised).
    angle_rad:
        Rotation angle in radians (right-hand rule).

    Returns
    -------
    list[float]
        The rotated point as a plain Python list ``[x, y, z]``.
    """
    p_arr = np.asarray(p, dtype=float)
    ap = np.asarray(axis_point, dtype=float)
    k = axis_unit  # already unit vector

    v = p_arr - ap
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    v_rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)
    return (ap + v_rot).tolist()
