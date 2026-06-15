"""Core logic for scaling a BlockMeshDict geometry by per-axis factors.

This module provides pure, side-effect-free helper functions and a
:func:`scale` entry point that returns a new
:class:`~meshing_utils.block_mesh_dict.BlockMeshDict` whose vertex
and edge coordinates have been multiplied by user-supplied scale factors.

The following elements are scaled:
- ``vertices``: all three coordinate components multiplied by (fx, fy, fz).
- ``edges``: all points in ``e.points`` scaled component-wise by (fx, fy, fz).
  For ``arc`` edges with non-uniform factors a **single** warning is emitted
  per :func:`scale` call (arc midpoints are no longer geometrically correct
  after non-uniform scaling).

The following elements are intentionally left unchanged:
- ``convertToMeters`` — the scale factors apply *on top of* the existing
  conversion; callers are responsible for adjusting ``convertToMeters`` if
  that is their intent.
- ``blocks`` — vertex references, cell counts, and grading are topology, not
  geometry.
- ``boundary`` patches — face vertex references are topology.
- ``geometry`` section — raw text; no transformation is applied.

The module never writes to disk; I/O is handled by the calling CLI tool.
"""

import copy
import logging
import math
from collections.abc import Sequence

from meshing_utils.foam.dict_file import BlockMeshDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_factors(factors: Sequence[float]) -> tuple[float, float, float]:
    """Validate and return the three scale factors as a tuple.

    Parameters
    ----------
    factors:
        A sequence of exactly three floats representing (fx, fy, fz).

    Returns
    -------
    tuple[float, float, float]
        The validated factors.

    Raises
    ------
    ValueError
        If the sequence does not contain exactly three values, if any value
        is not finite (NaN or infinity), or if any value is <= 0.
    """
    factors = list(factors)
    if len(factors) != 3:
        raise ValueError(
            f"Exactly 3 scale factors required (fx, fy, fz); got {len(factors)}"
        )
    for i, f in enumerate(factors):
        if not math.isfinite(f):
            raise ValueError(
                f"Scale factor[{i}] must be a finite number; got {f!r}"
            )
        if f <= 0.0:
            raise ValueError(
                f"Scale factor[{i}] must be strictly positive (> 0); got {f!r}"
            )
    return (factors[0], factors[1], factors[2])


def is_uniform(fx: float, fy: float, fz: float, *, rtol: float = 1e-9) -> bool:
    """Return ``True`` when all three factors are equal within *rtol*."""
    return abs(fx - fy) <= rtol * max(abs(fx), abs(fy), 1.0) and \
           abs(fx - fz) <= rtol * max(abs(fx), abs(fz), 1.0)


# ---------------------------------------------------------------------------
# Coordinate scaling helper
# ---------------------------------------------------------------------------

def _scale_point(point: list[float], fx: float, fy: float, fz: float) -> list[float]:
    """Return a new list with the point's coordinates multiplied by (fx, fy, fz)."""
    return [point[0] * fx, point[1] * fy, point[2] * fz]


# ---------------------------------------------------------------------------
# Main scale function
# ---------------------------------------------------------------------------

def scale(source: BlockMeshDict, fx: float, fy: float, fz: float) -> BlockMeshDict:
    """Scale *source* by the given per-axis factors and return the result.

    The original mesh is left **unchanged** (a deep copy is made internally).
    Only ``vertices`` and ``edges`` coordinates are modified; all other
    sections are copied verbatim.

    Parameters
    ----------
    source:
        The source mesh.  May contain zero blocks (a warning is logged in
        that case but no error is raised).
    fx, fy, fz:
        Scale factors for the x, y, and z axes respectively.  Must each be
        strictly positive finite floats.

    Returns
    -------
    BlockMeshDict
        The scaled mesh (a deep copy of *source* with coordinates updated).

    Raises
    ------
    ValueError
        If any factor is <= 0 or not finite.  (Callers should validate with
        :func:`validate_factors` before calling this function, but the check
        is repeated here for safety.)
    """
    # Re-validate inside the function for safety
    validate_factors([fx, fy, fz])

    if len(source.blocks) == 0:
        logger.warning(
            "Source mesh contains no blocks; scaling vertices and edges only."
        )

    # Deep-copy source so the original is not mutated
    result: BlockMeshDict = copy.deepcopy(source)

    # --- Scale vertices ---
    for vertex in result.vertices:
        vertex.coords = _scale_point(vertex.coords, fx, fy, fz)

    # --- Scale edge control points ---
    non_uniform = not is_uniform(fx, fy, fz)
    arc_warning_issued = False

    for edge in result.edges:
        if edge.type == "arc" and non_uniform and not arc_warning_issued:
            logger.warning(
                "Non-uniform scaling of 'arc' edges: the arc midpoint is "
                "scaled component-wise, which may produce geometrically "
                "incorrect arcs.  Consider using uniform scaling for arc edges."
            )
            arc_warning_issued = True
        edge.points = [_scale_point(pt, fx, fy, fz) for pt in edge.points]

    logger.info(
        "scale complete: %d vertices and %d edges scaled by (fx=%.6g, fy=%.6g, fz=%.6g)",
        len(result.vertices),
        len(result.edges),
        fx,
        fy,
        fz,
    )

    return result
