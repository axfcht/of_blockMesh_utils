"""Algorithm: match block faces against STEP solid surfaces and create patches.

This module is a pure-operations layer with no argparse or sys.exit.
The CLI (``meshing_utils.cli.extract_patches``) calls :func:`extract_patches`
and maps any :class:`~meshing_utils.exceptions.MeshingUtilsError` to
``sys.exit(1)`` for strict-mode errors.
"""

from __future__ import annotations

import logging
import math

from meshing_utils.cad.face_matching import (
    BlockFace,
    compute_outward_normal,
    effective_normal_tolerance,
    extract_block_faces,
    find_dominant_face,
    local_surface_normal,
    nearest_face_within_tol,
    normals_consistent,
    surface_type_of,
)
from meshing_utils.cad.step_names import NamedSolid
from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Face, Patch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_centroid(
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the arithmetic centroid of a list of points."""
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


# ---------------------------------------------------------------------------
# Public algorithm
# ---------------------------------------------------------------------------

def extract_patches(
    bmd: BlockMeshDict,
    named_solids: list[NamedSolid],
    *,
    tol: float,
    normal_angle_tol: float,
    curved_normal_angle_tol: float,
    default_patch_type: str,
    strict: bool,
) -> int:
    """Match block faces against solid surfaces and append matched patches.

    Mutates *bmd* in place by adding new :class:`~meshing_utils.foam.elements.Patch`
    objects to ``bmd.boundary``.

    Parameters
    ----------
    bmd:
        Loaded :class:`~meshing_utils.foam.dict_file.BlockMeshDict` to mutate.
    named_solids:
        Solids with their resolved names, as returned by
        :func:`~meshing_utils.cad.step_loader.load_solids_with_names`.
    tol:
        Distance tolerance in model units.
    normal_angle_tol:
        Angular tolerance in degrees for the normal consistency check on planar
        surfaces.  Cylinder and cone surfaces use ``2 * normal_angle_tol``.
    curved_normal_angle_tol:
        Angular tolerance in degrees for normal consistency on strongly curved
        surfaces.
    default_patch_type:
        OpenFOAM patch type string assigned to newly created patches.
    strict:
        When ``True``, multiple solid matches for a single block face raise
        :class:`~meshing_utils.exceptions.MeshingUtilsError` instead of only
        logging a warning.

    Returns
    -------
    int
        Number of patches added to *bmd*.

    Raises
    ------
    MeshingUtilsError
        In strict mode when a block face matches more than one solid.
    """
    solid_label_pairs = [(ns.solid, ns.name) for ns in named_solids]

    # --- Build block faces ---
    all_block_faces: list[BlockFace] = []
    for block in bmd.blocks:
        if len(block.vertices) != 8:
            logger.warning(
                "Block '%s' has %d vertices (expected 8) — skipped.",
                block.name,
                len(block.vertices),
            )
            continue
        faces = extract_block_faces(block, bmd)
        all_block_faces.extend(faces)

    logger.info("Extracted %d block faces from %d blocks", len(all_block_faces), len(bmd.blocks))

    # --- Match block faces to STP solids ---
    face_matches: dict[int, tuple[int, object, str | None]] = {}

    for bf_idx, block_face in enumerate(all_block_faces):
        matched: list[tuple[int, object, str | None]] = []

        all_sample_points: list[tuple[float, float, float]] = (
            list(block_face.vertex_coords) + list(block_face.support_points)
        )
        sample_flags: list[bool] = (
            [False] * len(block_face.vertex_coords)
            + [True] * len(block_face.support_points)
        )

        for solid_idx, (solid, label) in enumerate(solid_label_pairs):
            point_test_face = nearest_face_within_tol(block_face, solid, tol)
            if point_test_face is None:
                continue

            dominant_result = find_dominant_face(
                solid, all_sample_points, sample_flags, tol
            )
            if dominant_result is None:
                dominant_face = point_test_face
                contributing_points: list[tuple[float, float, float]] = []
            else:
                dominant_face, contributing_points = dominant_result

            stype = surface_type_of(dominant_face)
            eff_tol = effective_normal_tolerance(
                stype, normal_angle_tol, curved_normal_angle_tol
            )
            logger.debug(
                "Block face '%s/%s' vs solid '%s': surface_type=%s, eff_tol=%.1f deg",
                block_face.block_name,
                block_face.face_name,
                label,
                stype,
                eff_tol,
            )

            original_support_set = set(
                tuple(p) for p in block_face.support_points
            )
            support_pts_on_face = [
                pt for pt in contributing_points
                if tuple(pt) in original_support_set
            ]

            avg_stp_normal: tuple[float, float, float] | None = None

            if support_pts_on_face:
                normals_sum = [0.0, 0.0, 0.0]
                evaluated = 0
                for pt in support_pts_on_face:
                    n = local_surface_normal(dominant_face, pt)
                    if n is not None:
                        normals_sum[0] += n[0]
                        normals_sum[1] += n[1]
                        normals_sum[2] += n[2]
                        evaluated += 1
                if evaluated > 0:
                    nx, ny, nz = normals_sum[0], normals_sum[1], normals_sum[2]
                    length = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if length > 1e-15:
                        avg_stp_normal = (nx / length, ny / length, nz / length)

            if avg_stp_normal is None:
                centroid = _compute_centroid(block_face.vertex_coords)
                avg_stp_normal = local_surface_normal(dominant_face, centroid)

            if avg_stp_normal is None:
                avg_stp_normal = local_surface_normal(
                    dominant_face, block_face.vertex_coords[0]
                )

            if avg_stp_normal is not None:
                block_normal = compute_outward_normal(block_face.vertex_coords)
                if not normals_consistent(block_normal, avg_stp_normal, eff_tol):
                    continue

            matched.append((solid_idx, dominant_face, label))

        if len(matched) == 0:
            continue
        elif len(matched) == 1:
            face_matches[bf_idx] = matched[0]
        else:
            msg = (
                f"Block face '{block_face.block_name}/{block_face.face_name}' "
                f"matched {len(matched)} STP solids: "
                f"{[m[2] for m in matched]}"
            )
            if strict:
                # The CLI catches MeshingUtilsError and logs it for the user;
                # logging here too would duplicate the line.
                raise MeshingUtilsError(msg)
            else:
                logger.warning(msg + " — using first match.")
            face_matches[bf_idx] = matched[0]

    logger.info("Matched %d block faces to STP solids", len(face_matches))

    # --- Group matched faces by solid label ---
    solid_to_faces: dict[int, tuple[str | None, list[BlockFace]]] = {}
    for bf_idx, (solid_idx, _occ_face, label) in face_matches.items():
        if solid_idx not in solid_to_faces:
            solid_to_faces[solid_idx] = (label, [])
        solid_to_faces[solid_idx][1].append(all_block_faces[bf_idx])

    patches_added = 0
    patches_skipped = 0
    for solid_idx in sorted(solid_to_faces.keys()):
        label, block_faces = solid_to_faces[solid_idx]
        patch_name = label if label and label.strip() else f"patch{solid_idx}"

        if patch_name in bmd.boundary:
            logger.warning(
                "Patch name '%s' already exists in boundary — skipped (no merge).",
                patch_name,
            )
            patches_skipped += 1
            continue

        foam_faces: list[Face] = []
        for bf in block_faces:
            foam_faces.append(Face(vertices_or_string=list(bf.vertex_names)))

        patch = Patch(
            name_or_string=patch_name,
            type=default_patch_type,
            faces=foam_faces,
        )
        bmd.boundary.add(patch)
        patches_added += 1
        logger.info(
            "Added patch '%s' (type=%s) with %d face(s)",
            patch_name,
            default_patch_type,
            len(foam_faces),
        )

    if patches_skipped > 0:
        logger.warning(
            "%d patch(es) skipped due to name conflicts.",
            patches_skipped,
        )

    return patches_added
