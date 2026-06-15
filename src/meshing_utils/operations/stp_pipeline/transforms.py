"""Geometric transforms applied to the freshly loaded STEP data."""

from __future__ import annotations

from meshing_utils.geometry.hex_topology import HexCandidate, PointPool


def apply_origin_shift(
    candidates: list[HexCandidate],
    pool: PointPool,
    origin: tuple[float, float, float],
) -> None:
    """Shift all coordinates in *pool* by subtracting *origin* in-place."""
    ox, oy, oz = origin
    for i in range(len(pool)):
        x, y, z = pool.coord_at(i)
        pool._coords[i] = (x - ox, y - oy, z - oz)
