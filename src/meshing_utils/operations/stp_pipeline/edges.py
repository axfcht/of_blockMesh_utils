"""OCC edge classification: line / arc / bspline -> :class:`CurveInfo`."""

from __future__ import annotations

import logging
import math

from meshing_utils.geometry.hex_topology import CurveInfo

logger = logging.getLogger(__name__)


def classify_edge(topo_edge, n_samples: int = 20) -> CurveInfo:
    """Classify a topological edge and return a ``CurveInfo``.

    ``kind`` is ``"line"``, ``"arc"``, or ``"bspline"``.
    """
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import (
            GeomAbs_BezierCurve,
            GeomAbs_BSplineCurve,
            GeomAbs_Circle,
            GeomAbs_Line,
        )
    except ImportError as exc:
        raise ImportError(
            "OCP (PythonOCC-core) is required for classify_edge."
        ) from exc

    adaptor = BRepAdaptor_Curve(topo_edge)
    curve_type = adaptor.GetType()
    u_first = adaptor.FirstParameter()
    u_last = adaptor.LastParameter()

    if curve_type == GeomAbs_Line:
        return CurveInfo(kind="line", support_points=[])

    if curve_type == GeomAbs_Circle:
        arc_angle = abs(u_last - u_first)
        if arc_angle <= math.pi / 2:
            u_mid = (u_first + u_last) / 2.0
            pnt = adaptor.Value(u_mid)
            midpoint = (pnt.X(), pnt.Y(), pnt.Z())
            return CurveInfo(
                kind="arc",
                support_points=[],
                arc_midpoint=midpoint,
                arc_angle=arc_angle,
            )
        else:
            logger.warning(
                "Arc edge has angle %.3f rad > pi/2; falling back to bspline sampling.",
                arc_angle,
            )
            inner = _sample_inner_points(adaptor, u_first, u_last, n_samples)
            return CurveInfo(kind="bspline", support_points=inner, arc_angle=arc_angle)

    if curve_type in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        inner = _sample_inner_points(adaptor, u_first, u_last, n_samples)
        return CurveInfo(kind="bspline", support_points=inner)

    logger.warning(
        "Unknown curve type %s; falling back to bspline sampling.", curve_type
    )
    inner = _sample_inner_points(adaptor, u_first, u_last, n_samples)
    return CurveInfo(kind="bspline", support_points=inner)


def _sample_inner_points(
    adaptor,
    u_first: float,
    u_last: float,
    n_samples: int,
) -> list[tuple[float, float, float]]:
    """Sample *n_samples* inner points (excluding endpoints) along *adaptor*."""
    points = []
    total = n_samples + 2
    for i in range(1, total - 1):
        u = u_first + (u_last - u_first) * i / (total - 1)
        pnt = adaptor.Value(u)
        points.append((pnt.X(), pnt.Y(), pnt.Z()))
    return points
