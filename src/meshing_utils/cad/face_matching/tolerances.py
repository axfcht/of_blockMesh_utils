"""Surface-type queries and angular tolerance helpers.

Wraps the OCC ``BRepAdaptor_Surface`` surface-type enum into a readable
string, derives the adaptive normal tolerance per surface kind, and
provides the bidirectional normal-consistency check used during face
matching.
"""

from __future__ import annotations

import math

# Mapping from OCC GeomAbs surface-type enum values to readable strings.
# The integer values correspond to GeomAbs_SurfaceType members:
#   Plane=0, Cylinder=1, Cone=2, Sphere=3, Torus=4,
#   BezierSurface=5, BSplineSurface=6, SurfaceOfRevolution=7,
#   SurfaceOfExtrusion=8, OffsetSurface=9, OtherSurface=10
_SURFACE_TYPE_MAP: dict[int, str] = {
    0: "plane",
    1: "cylinder",
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "bezier",
    6: "bspline",
    7: "revolution",
    8: "extrusion",
    9: "offset",
    10: "other",
}


def surface_type_of(face) -> str:
    """Return a string describing the underlying surface type of *face*.

    One of ``plane``, ``cylinder``, ``cone``, ``sphere``, ``torus``,
    ``bezier``, ``bspline``, ``revolution``, ``extrusion``, ``offset``,
    or ``other`` (the fallback when the type cannot be determined).
    """
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(face)
        surface_type_enum = adaptor.GetType()
        try:
            type_int = int(surface_type_enum)
        except (TypeError, ValueError):
            type_int = int(surface_type_enum.value) if hasattr(surface_type_enum, "value") else -1
        return _SURFACE_TYPE_MAP.get(type_int, "other")
    except Exception:
        return "other"


def effective_normal_tolerance(
    surface_type: str,
    planar_tol_deg: float,
    curved_tol_deg: float,
) -> float:
    """Return the appropriate normal-consistency tolerance for *surface_type*.

    Planar surfaces get *planar_tol_deg*; cylinders and cones get twice
    that (more curvature variation); all other surface types use
    *curved_tol_deg*.
    """
    if surface_type == "plane":
        return planar_tol_deg
    if surface_type in ("cylinder", "cone"):
        return 2.0 * planar_tol_deg
    return curved_tol_deg


def normals_consistent(
    n1: tuple[float, float, float],
    n2: tuple[float, float, float],
    angle_tol_deg: float = 5.0,
) -> bool:
    """Return ``True`` when *n1* and *n2* are approximately parallel or anti-parallel.

    Bidirectional: the angle between *n1* and *n2* must be within
    *angle_tol_deg* of either 0° or 180°. This accommodates the fact that
    OpenFOAM and STEP may orient corresponding normals opposite to each
    other.
    """
    l1 = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)
    l2 = math.sqrt(n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2)
    if l1 < 1e-15 or l2 < 1e-15:
        return False

    dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2)
    dot = max(-1.0, min(1.0, dot))
    angle_deg = math.degrees(math.acos(dot))

    tol = abs(angle_tol_deg)
    return angle_deg <= tol or angle_deg >= (180.0 - tol)
