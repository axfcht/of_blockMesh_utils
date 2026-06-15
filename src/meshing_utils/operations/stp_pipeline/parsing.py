"""String parsers for CLI tuple arguments: ``parse_origin`` and ``parse_fractions``."""

from __future__ import annotations

import re

_ORIGIN_RE = re.compile(
    r"^\(\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)$"
)

_FRACTIONS_RE = re.compile(
    r"^\(\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)$"
)


def parse_origin(s: str) -> tuple[float, float, float]:
    """Parse an origin string of the form ``(x y z)``.

    Raises ``ValueError`` if the string does not match the expected pattern.
    """
    m = _ORIGIN_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Invalid origin format {s!r}. "
            "Expected '(x y z)' with numeric components."
        )
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def parse_fractions(s: str) -> tuple[float, float, float]:
    """Parse a fractions string of the form ``(fx fy fz)``.

    All components must be strictly positive (> 0). Raises ``ValueError`` on
    malformed input or non-positive components.
    """
    m = _FRACTIONS_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Invalid fractions format {s!r}. "
            "Expected '(fx fy fz)' with positive numeric components."
        )
    fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
    for name, val in (("fx", fx), ("fy", fy), ("fz", fz)):
        if val <= 0.0:
            raise ValueError(
                f"Fraction {name}={val} must be strictly positive (> 0)."
            )
    return (fx, fy, fz)
