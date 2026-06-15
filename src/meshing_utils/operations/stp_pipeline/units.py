"""STEP ``LENGTH_UNIT`` -> blockMeshDict ``convertToMeters`` mapping."""

from __future__ import annotations

_STEP_UNIT_TO_METERS: dict[str, float] = {
    "M": 1.0,
    "METRE": 1.0,
    "METER": 1.0,
    "DM": 0.1,
    "CM": 0.01,
    "MM": 0.001,
    "MILLIMETRE": 0.001,
    "MILLIMETER": 0.001,
    "UM": 1.0e-6,
    "MICROMETRE": 1.0e-6,
    "MICROMETER": 1.0e-6,
    "IN": 0.0254,
    "INCH": 0.0254,
    "FT": 0.3048,
    "FOOT": 0.3048,
}


def step_unit_to_meters(unit: str | None) -> float | None:
    """Map a STEP ``LENGTH_UNIT`` string to a ``convertToMeters`` factor.

    Returns ``None`` when the unit is unknown or empty, so callers can fall
    back to a default or preserve a previously known value.
    """
    if not unit:
        return None
    return _STEP_UNIT_TO_METERS.get(unit.strip().upper())
