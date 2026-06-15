"""Point-in-solid containment tests and block centroid helpers.

OCC (PythonOCC-core) is imported lazily so that the module is always
importable even when OCC is not installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Classification state constants
# ---------------------------------------------------------------------------

STATE_IN = "IN"
STATE_ON = "ON"
STATE_OUT = "OUT"


# ---------------------------------------------------------------------------
# Inset-sampling constants
# ---------------------------------------------------------------------------

DEFAULT_INSET_FACTOR = 0.5

# Minimum combined vertex_in + vertex_on count required to activate Stage A
# (vertex-dominant decision).  5 is the smallest strict majority of 8 vertices;
# 4/8 would be a 50/50 tie and thus ambiguous.
VERTEX_DOMINANT_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Block geometry helpers
# ---------------------------------------------------------------------------

def compute_block_centroid(
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the arithmetic mean of 8 vertex coordinates.

    Parameters
    ----------
    coords:
        Exactly 8 (x, y, z) tuples — the corners of a hex block.

    Returns
    -------
    tuple[float, float, float]
        The centroid point.

    Raises
    ------
    AssertionError
        When ``coords`` does not contain exactly 8 entries.
    """
    assert len(coords) == 8, f"Expected 8 coords, got {len(coords)}"
    x = sum(c[0] for c in coords) / 8.0
    y = sum(c[1] for c in coords) / 8.0
    z = sum(c[2] for c in coords) / 8.0
    return (x, y, z)


def compute_inset_sample_points(
    coords: list[tuple[float, float, float]],
    inset_factor: float = DEFAULT_INSET_FACTOR,
) -> list[tuple[float, float, float]]:
    """Return 9 sample points: [centroid, V0_inset, V1_inset, ..., V7_inset].

    Each inset vertex sample is computed as::

        P_i = M + inset_factor * (V_i - M)

    where M is the block centroid.  With ``inset_factor = 0.5`` the sample
    lies exactly halfway between the centroid and the corresponding vertex,
    which is robust for typical hex blocks with sector angles up to ~90°.

    Parameters
    ----------
    coords:
        Exactly 8 (x, y, z) tuples — the corners of a hex block in the
        same order as expected by :func:`compute_block_centroid`.
    inset_factor:
        Interpolation factor in the open interval (0, 1).  Default: 0.5.

    Returns
    -------
    list of tuple[float, float, float]
        List of 9 points: index 0 is the centroid, indices 1-8 are the
        inset vertex samples corresponding to ``coords[0]``-``coords[7]``.

    Raises
    ------
    ValueError
        When ``coords`` does not contain exactly 8 entries, or when
        ``inset_factor`` is not in the open interval (0, 1).
    """
    if len(coords) != 8:
        raise ValueError(f"Expected 8 coords, got {len(coords)}")
    if not (0.0 < inset_factor < 1.0):
        raise ValueError(
            f"inset_factor must be in (0, 1), got {inset_factor}"
        )
    M = compute_block_centroid(coords)
    samples: list[tuple[float, float, float]] = [M]
    for V in coords:
        P = (
            M[0] + inset_factor * (V[0] - M[0]),
            M[1] + inset_factor * (V[1] - M[1]),
            M[2] + inset_factor * (V[2] - M[2]),
        )
        samples.append(P)
    return samples


def compute_block_sample_sets(
    coords: list[tuple[float, float, float]],
    inset_factor: float = DEFAULT_INSET_FACTOR,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return ``(interior_samples, vertex_samples)`` for a hex block.

    ``interior_samples`` is a list of 9 points: the block centroid followed by
    8 inset-vertex samples as returned by :func:`compute_inset_sample_points`.

    ``vertex_samples`` is a list of the 8 raw vertex coordinates in the same
    order as *coords*.

    Parameters
    ----------
    coords:
        Exactly 8 (x, y, z) tuples — the corners of a hex block.
    inset_factor:
        Inset factor forwarded to :func:`compute_inset_sample_points`; must
        be in the open interval (0, 1).  Default: 0.5.

    Returns
    -------
    tuple[list, list]
        ``(interior_samples, vertex_samples)`` where ``interior_samples`` has
        length 9 and ``vertex_samples`` has length 8.

    Raises
    ------
    ValueError
        When ``coords`` does not contain exactly 8 entries, or when
        ``inset_factor`` is not in (0, 1).
    """
    interior = compute_inset_sample_points(coords, inset_factor)
    vertices = list(coords)
    return (interior, vertices)


def _lookup_vertex(bmd, ref: str):
    """Return the Vertex for *ref* from *bmd*.

    First tries a named lookup (``bmd.vertices.get(ref)``); if that raises
    ``KeyError``, attempts an integer-index fallback.

    Parameters
    ----------
    bmd:
        A :class:`~meshing_utils.foam.dict_file.BlockMeshDict` instance.
    ref:
        Vertex reference string — either a name or a numeric index string.

    Raises
    ------
    KeyError
        When the vertex cannot be found by name or index.
    IndexError
        When *ref* is a valid integer but out of range.
    """
    try:
        return bmd.vertices.get(ref)
    except KeyError:
        pass
    # Numeric index fallback
    idx = int(ref)
    return bmd.vertices[idx]


def resolve_block_coords(
    block,
    bmd,
) -> list[tuple[float, float, float]]:
    """Resolve the 3-D coordinates of all vertices referenced by *block*.

    Parameters
    ----------
    block:
        A :class:`~meshing_utils.foam.elements.block.Block` instance.
    bmd:
        A :class:`~meshing_utils.foam.dict_file.BlockMeshDict` instance.

    Returns
    -------
    list of tuple[float, float, float]
        Ordered list of (x, y, z) tuples, one per vertex reference.

    Raises
    ------
    ValueError
        When the resolved coordinate list does not contain exactly 8 entries.
    """
    result: list[tuple[float, float, float]] = []
    for ref in block.vertices:
        vertex = _lookup_vertex(bmd, ref)
        result.append((vertex.coords[0], vertex.coords[1], vertex.coords[2]))
    if len(result) != 8:
        raise ValueError(
            f"Block {block.name!r} has {len(result)} vertices, expected 8"
        )
    return result


# ---------------------------------------------------------------------------
# OCC-based containment tests
# ---------------------------------------------------------------------------

def classify_point_in_solid(solid, point: tuple[float, float, float], tol: float) -> str:
    """Classify *point* relative to *solid* using OCC BRepClass3d.

    Parameters
    ----------
    solid:
        A ``TopoDS_Solid`` OCC shape.
    point:
        The (x, y, z) query point.
    tol:
        Classifier tolerance in model units.

    Returns
    -------
    str
        One of :data:`STATE_IN`, :data:`STATE_ON`, or :data:`STATE_OUT`.
    """
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    classifier = BRepClass3d_SolidClassifier(solid, gp_Pnt(*point), tol)
    state = classifier.State()
    if state == TopAbs_IN:
        return STATE_IN
    if state == TopAbs_ON:
        return STATE_ON
    return STATE_OUT


def point_inside_solid(solid, point: tuple[float, float, float], tol: float) -> bool:
    """Return ``True`` when *point* is inside or on the boundary of *solid*.

    Parameters
    ----------
    solid:
        A ``TopoDS_Solid`` OCC shape.
    point:
        The (x, y, z) query point.
    tol:
        Classifier tolerance in model units.
    """
    return classify_point_in_solid(solid, point, tol) in (STATE_IN, STATE_ON)


# ---------------------------------------------------------------------------
# Bounding-box diagonal helper
# ---------------------------------------------------------------------------

def bmd_bbox_diagonal(bmd) -> float:
    """Return the Euclidean length of the bounding-box diagonal over all vertices.

    Parameters
    ----------
    bmd:
        A :class:`~meshing_utils.foam.dict_file.BlockMeshDict` instance.

    Returns
    -------
    float
        Diagonal length, or ``0.0`` when *bmd* contains no vertices.
    """
    vertices = list(bmd.vertices)
    if not vertices:
        return 0.0

    xs = [v.coords[0] for v in vertices]
    ys = [v.coords[1] for v in vertices]
    zs = [v.coords[2] for v in vertices]

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ---------------------------------------------------------------------------
# AABB type alias
# ---------------------------------------------------------------------------

# An axis-aligned bounding box represented as (xmin, ymin, zmin, xmax, ymax, zmax).
AABB = tuple[float, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Solid AABB computation (OCC-based, lazy import)
# ---------------------------------------------------------------------------

def compute_solid_aabb(solid, tol: float) -> AABB:
    """Return a padded AABB for *solid* using OCC ``Bnd_Box``.

    The padding is ``max(tol, diag * 1e-9)`` where *diag* is the Euclidean
    length of the unpadded bounding-box diagonal.

    Parameters
    ----------
    solid:
        A ``TopoDS_Solid`` OCC shape.
    tol:
        Base tolerance used as a minimum padding value (model units).

    Returns
    -------
    AABB
        ``(xmin-pad, ymin-pad, zmin-pad, xmax+pad, ymax+pad, zmax+pad)``

    Raises
    ------
    ValueError
        When the OCC bounding box is void (e.g. empty shape).
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bbox = Bnd_Box()
    BRepBndLib.Add_s(solid, bbox)

    if bbox.IsVoid():
        raise ValueError("compute_solid_aabb: bounding box is void for the given solid")

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    diag = math.sqrt(dx * dx + dy * dy + dz * dz)
    pad = max(tol, diag * 1e-9)

    return (xmin - pad, ymin - pad, zmin - pad, xmax + pad, ymax + pad, zmax + pad)


# ---------------------------------------------------------------------------
# Point-cloud AABB computation (pure Python)
# ---------------------------------------------------------------------------

def compute_points_aabb(points: Iterable[tuple[float, float, float]]) -> AABB:
    """Return the axis-aligned bounding box of *points* without padding.

    Parameters
    ----------
    points:
        Iterable of (x, y, z) tuples.

    Returns
    -------
    AABB
        ``(xmin, ymin, zmin, xmax, ymax, zmax)``

    Raises
    ------
    ValueError
        When *points* is empty.
    """
    pts = list(points)
    if not pts:
        raise ValueError("compute_points_aabb: points iterable is empty")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# AABB overlap test (pure Python)
# ---------------------------------------------------------------------------

def aabbs_overlap(a: AABB, b: AABB) -> bool:
    """Return ``True`` when the two AABBs overlap (edge-inclusive).

    Uses the Separating Axis Theorem along each coordinate axis.

    Parameters
    ----------
    a:
        First AABB as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
    b:
        Second AABB as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    Returns
    -------
    bool
        ``True`` when the AABBs share at least one point (touching counts
        as overlapping).
    """
    # Separating axis test: if separated along any axis → no overlap
    if a[3] < b[0] or b[3] < a[0]:
        return False
    if a[4] < b[1] or b[4] < a[1]:
        return False
    return not (a[5] < b[2] or b[5] < a[2])


# ---------------------------------------------------------------------------
# Reusable solid classifier (OCC-based, lazy import)
# ---------------------------------------------------------------------------

def make_solid_classifier(solid):
    """Return a ``BRepClass3d_SolidClassifier`` initialised with *solid*.

    The returned classifier can be reused for multiple point queries via
    :func:`classify_point_with_classifier` without re-initialising the
    internal OCC data structures per point.

    Parameters
    ----------
    solid:
        A ``TopoDS_Solid`` OCC shape.

    Returns
    -------
    BRepClass3d_SolidClassifier
        A classifier instance bound to *solid*.
    """
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier

    return BRepClass3d_SolidClassifier(solid)


def classify_point_with_classifier(
    classifier,
    point: tuple[float, float, float],
    tol: float,
) -> str:
    """Classify *point* using a pre-built ``BRepClass3d_SolidClassifier``.

    Unlike :func:`classify_point_in_solid`, this function reuses an existing
    classifier object, avoiding the overhead of re-initialising OCC internal
    data structures for every query point.

    Parameters
    ----------
    classifier:
        A ``BRepClass3d_SolidClassifier`` instance as returned by
        :func:`make_solid_classifier`.
    point:
        The (x, y, z) query point.
    tol:
        Classifier tolerance in model units.

    Returns
    -------
    str
        One of :data:`STATE_IN`, :data:`STATE_ON`, or :data:`STATE_OUT`.
    """
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    classifier.Perform(gp_Pnt(*point), tol)
    state = classifier.State()
    if state == TopAbs_IN:
        return STATE_IN
    if state == TopAbs_ON:
        return STATE_ON
    return STATE_OUT
