"""Core geometry logic for revolving a BlockMeshDict around an arbitrary axis.

This module provides pure, side-effect-free helper functions and a
:func:`revolve` entry point that builds a new :class:`~meshing_utils.block_mesh_dict.BlockMeshDict`
from a source mesh by replicating all elements (vertices, edges, blocks, and
boundary faces) at uniformly spaced angular steps around a user-defined rotation
axis.

All geometry is computed with Rodrigues' rotation formula.  The module never
writes to disk; I/O is handled by the calling CLI tool.
"""

import copy
import logging
import math
from dataclasses import dataclass

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Block, Edge, Face, Patch
from meshing_utils.geometry.rotations import normalize, rotate_point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RevolveConfig:
    """Parameters for a revolve operation.

    Attributes
    ----------
    axis_point:
        A point on the rotation axis (x, y, z).
    axis_dir:
        Direction vector of the rotation axis.  Need not be normalised; it is
        normalised internally.  Must not be the zero vector.
    count:
        Total number of copies **including** the original.  Must be >= 2.
    angle:
        Total rotation angle in **degrees**.  Positive = right-hand rule around
        *axis_dir*.  Must satisfy ``0 < |angle| <= 360``.
        Default: 360.0.
    tol:
        Snap-to-grid tolerance for vertex deduplication.  When ``None`` the
        tolerance is derived automatically from the bounding-box diagonal of the
        source mesh: ``max(1e-9, 1e-7 * bbox_diagonal)``.
    unique_patches:
        Controls per-copy patch uniquification.

        * ``None`` (default) — feature disabled; all rotational copies are
          accumulated into the original patch, preserving existing behaviour.
        * ``[]`` (empty list) — uniquify **all** boundary patches: for each
          copy at step *k* (k = 1 … count-1) a new patch named
          ``<original>_<k>`` is created that contains only the faces for that
          specific rotational copy.
        * ``["name", ...]`` — uniquify only the named patches; all other
          patches accumulate as before.
    """

    axis_point: tuple[float, float, float]
    axis_dir: tuple[float, float, float]
    count: int
    angle: float = 360.0
    tol: float | None = None
    unique_patches: list[str] | None = None

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError(f"count must be >= 2, got {self.count}")
        if not (0.0 < abs(self.angle) <= 360.0):
            raise ValueError(f"angle must satisfy 0 < |angle| <= 360, got {self.angle}")
        normalize(self.axis_dir)  # raises ValueError if axis_dir is the zero vector


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def plan_angles(total_angle_deg: float, count: int) -> list[float]:
    """Return the list of rotation angles for copies k = 1 … count-1.

    For a **full circle** (``|total_angle_deg| == 360``), the step is
    ``total_angle_deg / count`` so that the last copy does NOT land on 360°
    (which would coincide with the original).  A full circle with count=4
    yields ``[90, 180, 270]``.

    For a **partial sweep**, the step is ``total_angle_deg / (count - 1)``
    so that the last copy lands exactly on *total_angle_deg*.  A half-circle
    (180°) with count=4 yields ``[60, 120, 180]``.

    Parameters
    ----------
    total_angle_deg:
        Total rotation angle in degrees.
    count:
        Total number of instances including the original (>= 2).

    Returns
    -------
    list[float]
        Rotation angles in degrees for each additional copy.
    """
    if abs(abs(total_angle_deg) - 360.0) <= 1e-9:
        step = total_angle_deg / count          # full circle: last copy is NOT at 360
    else:
        step = total_angle_deg / (count - 1)    # partial: last copy lands on total_angle_deg
    return [k * step for k in range(1, count)]


def bbox_diagonal(vertices) -> float:
    """Return the bounding-box diagonal length of *vertices*.

    Parameters
    ----------
    vertices:
        An iterable of :class:`~meshing_utils.foam.elements.Vertex` objects.

    Returns
    -------
    float
        Euclidean length of the bounding-box diagonal, or 1.0 for empty input.
    """
    coords = [v.coords for v in vertices]
    if not coords:
        return 1.0
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    diag = math.sqrt(
        (max(xs) - min(xs)) ** 2
        + (max(ys) - min(ys)) ** 2
        + (max(zs) - min(zs)) ** 2
    )
    return diag if diag > 0.0 else 1.0


def compute_default_tol(vertices) -> float:
    """Return the default snap tolerance derived from the vertex bounding box.

    Formula: ``max(1e-9, 1e-7 * bbox_diagonal)``.
    """
    return max(1e-9, 1e-7 * bbox_diagonal(vertices))


def parse_axis(s: str) -> tuple[float, float, float]:
    """Parse a ``"(x y z)"`` or ``"x y z"`` string into a 3-tuple of floats.

    This helper allows callers to pass axis point / direction either as a
    parenthesised OpenFOAM-style string or as bare space-separated numbers.

    Parameters
    ----------
    s:
        Input string, e.g. ``"(0 0 1)"`` or ``"0 0 1"``.

    Returns
    -------
    tuple[float, float, float]

    Raises
    ------
    ValueError
        If the string cannot be parsed into exactly three floats.
    """
    cleaned = s.strip().lstrip("(").rstrip(")")
    parts = cleaned.split()
    if len(parts) != 3:
        raise ValueError(
            f"Expected exactly 3 numbers for axis, got {len(parts)!r} in {s!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Cannot parse axis from string {s!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal face-removal helper
# ---------------------------------------------------------------------------

def strip_internal_sector_faces(
    result: BlockMeshDict,
    exclude_patches: set | None = None,
) -> int:
    """Remove boundary faces that are shared by two or more blocks.

    A face is considered *internal* (i.e., lying on the interface between two
    adjacent sector copies) when its vertex set is a subset of the vertex sets
    of at least **two** different blocks.  Such faces must be removed to avoid
    non-conformal interior boundaries in the revolve result.

    Parameters
    ----------
    result:
        The :class:`~meshing_utils.block_mesh_dict.BlockMeshDict` to
        mutate in-place.
    exclude_patches:
        Optional set of patch names to skip entirely.  Patches listed here are
        left untouched regardless of whether their faces are shared.  Primarily
        used to protect newly created unique-patch copies from being stripped.
        When ``None`` or empty, no patches are excluded (default behaviour).

    Returns
    -------
    int
        Total number of faces removed across all patches.
    """
    excluded = exclude_patches or set()

    block_vsets: list[tuple[str, frozenset]] = [
        (b.name, frozenset(b.vertices)) for b in result.blocks
    ]

    total_removed = 0
    for patch in result.boundary:
        if patch.name in excluded:
            continue
        kept: list[Face] = []
        removed_count = 0
        for face in patch.faces:
            fset = frozenset(face.vertices)
            containing = sum(1 for _, bset in block_vsets if fset <= bset)
            if containing >= 2:
                removed_count += 1
            else:
                kept.append(face)
        if removed_count > 0:
            logger.debug(
                "Patch '%s': removed %d internal sector face(s)",
                patch.name,
                removed_count,
            )
        patch.faces[:] = kept
        total_removed += removed_count

    return total_removed


# ---------------------------------------------------------------------------
# Main revolve function
# ---------------------------------------------------------------------------

def revolve(source: BlockMeshDict, cfg: RevolveConfig) -> BlockMeshDict:
    """Revolve *source* around the axis defined in *cfg* and return the result.

    The original mesh is left **unchanged** (a deep copy is made internally).
    The returned :class:`~meshing_utils.block_mesh_dict.BlockMeshDict`
    contains all original elements plus (count-1) rotated copies.

    Parameters
    ----------
    source:
        The source mesh.  Must contain at least one block.
    cfg:
        Revolve configuration.  See :class:`RevolveConfig`.

    Returns
    -------
    BlockMeshDict
        The revolved mesh.

    Raises
    ------
    ValueError
        On invalid configuration (count < 2, bad angle, zero axis, empty
        blocks).
    """
    # ------------------------------------------------------------------
    # 1. Validation
    # ------------------------------------------------------------------
    if len(source.blocks) == 0:
        raise ValueError("source mesh contains no blocks; nothing to revolve")

    axis_unit = normalize(cfg.axis_dir)  # raises ValueError if zero

    if source.geometry_body and source.geometry_body.strip():
        logger.warning(
            "Source mesh contains a geometry section which will NOT be "
            "rotated.  The geometry section is copied verbatim."
        )

    # ------------------------------------------------------------------
    # 2. Deep-copy source → result; determine tolerance
    # ------------------------------------------------------------------
    result: BlockMeshDict = copy.deepcopy(source)
    tol = cfg.tol if cfg.tol is not None else compute_default_tol(source.vertices)

    # Initialise the snap-to-grid for the result with all existing vertices
    # by doing a dummy lookup that triggers a grid rebuild.
    if len(result.vertices) > 0:
        first = result.vertices[0]
        result.find_or_add_vertex(
            (first.coords[0], first.coords[1], first.coords[2]), tol
        )

    # ------------------------------------------------------------------
    # 3. Snapshot of source elements (BEFORE any mutation of result)
    # ------------------------------------------------------------------
    src_vertex_names: list[str] = [v.name for v in source.vertices]
    src_vertex_name_set: frozenset = frozenset(src_vertex_names)
    src_edges: list[Edge] = list(source.edges)
    src_blocks: list[Block] = list(source.blocks)

    # Only include patch faces whose ALL vertices are source vertices
    src_patch_faces: list[tuple[str, Face]] = []
    for patch in source.boundary:
        for face in patch.faces:
            if all(n in src_vertex_name_set for n in face.vertices):
                src_patch_faces.append((patch.name, face))

    # ------------------------------------------------------------------
    # 4. Generate copies for k = 1 … count-1
    # ------------------------------------------------------------------
    angles_deg: list[float] = plan_angles(cfg.angle, cfg.count)

    # Resolve unique_set from cfg.unique_patches (tri-state semantics)
    if cfg.unique_patches is None:
        unique_set: set[str] = set()                         # feature off
    elif len(cfg.unique_patches) == 0:
        unique_set = {p.name for p in source.boundary}      # all patches
    else:
        requested = set(cfg.unique_patches)
        available = {p.name for p in source.boundary}
        missing = requested - available
        if missing:
            raise ValueError(
                f"unique_patches contains unknown patches: {sorted(missing)}. "
                f"Available patches: {sorted(available)}"
            )
        unique_set = requested

    # Naming-collision check: ensure <orig>_<k> does not already exist
    if unique_set:
        existing_names = {p.name for p in source.boundary}
        for orig in unique_set:
            for k in range(1, cfg.count):
                candidate = f"{orig}_{k}"
                if candidate in existing_names:
                    raise ValueError(
                        f"Patch name collision: '{candidate}' already exists in "
                        f"source boundary; cannot uniquify '{orig}'."
                    )

    # Track patch names created by the unique feature so they can be
    # excluded from internal-face stripping later.
    created_unique_names: set[str] = set()

    for k_index, angle_deg in enumerate(angles_deg, start=1):
        angle_rad = angle_deg * math.pi / 180.0

        # 4a. Vertex mapping: source name -> rotated name in result
        name_map: dict[str, str] = {}
        for vname in src_vertex_names:
            v = source.vertices.get(vname)
            p_rot = rotate_point(
                v.coords, cfg.axis_point, axis_unit, angle_rad
            )
            name_map[vname] = result.find_or_add_vertex(
                (p_rot[0], p_rot[1], p_rot[2]), tol
            )

        # 4b. Edges
        for e in src_edges:
            a = name_map[e.v_start]
            b = name_map[e.v_end]
            if a == b:
                # Both endpoints collapsed onto the axis — degenerate edge
                continue
            if result.find_edge(a, b) is not None:
                # Edge already present (e.g. deduplication at 360 seam)
                continue
            new_points = [
                rotate_point(pt, cfg.axis_point, axis_unit, angle_rad)
                for pt in e.points
            ]
            result.edges.add(
                Edge(e.type, a, b, points=new_points)
            )

        # 4c. Blocks
        for blk in src_blocks:
            new_verts = [name_map[n] for n in blk.vertices]
            if len(set(new_verts)) != 8:
                # Degenerate block (vertices collapsed onto rotation axis)
                logger.debug(
                    "Skipping degenerate block (collapsed vertices) for "
                    "angle %.4f deg", angle_deg
                )
                continue
            if result.has_block_with_vertex_set(new_verts):
                # Already exists (e.g. deduplication at 360 seam)
                continue
            new_block = Block(
                name_or_string=f"block{result.next_block_index()}",
                vertices=new_verts,
                cells=list(blk.cells),
                type=blk.type,
                grading_type=blk.grading_type,
                grading_def=list(blk.grading_def),
                zone=blk.zone,
            )
            result.blocks.add(new_block)

        # 4d. Patch faces
        for (patch_name, face) in src_patch_faces:
            new_face_verts = [name_map[n] for n in face.vertices]
            if len(set(new_face_verts)) != len(new_face_verts):
                # Degenerate face (duplicate vertices)
                continue

            if patch_name in unique_set:
                # Unique-patch mode: route this copy's faces to a dedicated patch
                target_name = f"{patch_name}_{k_index}"
                if target_name not in result.boundary:
                    src_patch = source.boundary.get(patch_name)
                    new_patch = Patch(
                        name_or_string=target_name,
                        type=src_patch.type,
                        faces=[],
                        marker=src_patch.marker,
                    )
                    result.boundary.add(new_patch)
                    created_unique_names.add(target_name)
                target = result.boundary.get(target_name)
            else:
                target = result.boundary.get(patch_name)

            existing_sets = [frozenset(f.vertices) for f in target.faces]
            if frozenset(new_face_verts) in existing_sets:
                continue
            target.faces.append(Face(vertices_or_string=new_face_verts))

    # ------------------------------------------------------------------
    # 5. Remove internal sector faces
    #    Newly created unique patches are excluded from stripping so that
    #    their per-copy faces are always preserved.
    # ------------------------------------------------------------------
    total_removed = strip_internal_sector_faces(result, exclude_patches=created_unique_names)
    if total_removed > 0:
        logger.info(
            "Removed %d internal sector face(s) from boundary patches",
            total_removed,
        )

    # ------------------------------------------------------------------
    # 6. Sanity check
    # ------------------------------------------------------------------
    n_src = len(source.blocks)
    n_result = len(result.blocks)
    if n_result == n_src:
        logger.error(
            "revolve produced no new blocks (result has %d block(s), same as "
            "source).  Check axis direction, angle, count, and tolerance.",
            n_result,
        )
    else:
        logger.info(
            "revolve complete: %d source block(s) → %d total block(s) "
            "(%d new), %d total vertices",
            n_src,
            n_result,
            n_result - n_src,
            len(result.vertices),
        )

    return result
