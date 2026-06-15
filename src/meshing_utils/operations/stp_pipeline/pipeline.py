"""Public ``run`` entry point and the monolithic ``_run_impl`` orchestrator.

``_run_impl`` may be further decomposed along the inline section banners
(convertToMeters resolution, cell-count strategy selection, append handling).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from meshing_utils.cad.step_loader import find_single_step_file
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Block
from meshing_utils.geometry.cell_count import (
    EuclideanProjectedCellCountStrategy,
    PropagatedCellCountStrategy,
)
from meshing_utils.geometry.hex_topology import (
    CurveInfo,
    assert_block_face_normals_outward,
    check_global_face_consistency,
    enforce_openfoam_face_convention,
    ensure_right_handed,
    order_hex_vertices,
    validate_hex,
)
from meshing_utils.operations.stp_pipeline.assembly import (
    _block_to_ordering_and_faces,
    resolve_block_name,
)
from meshing_utils.operations.stp_pipeline.config import StpPipelineConfig
from meshing_utils.operations.stp_pipeline.conflicts import add_edge_with_conflict_check
from meshing_utils.operations.stp_pipeline.transforms import apply_origin_shift
from meshing_utils.operations.stp_pipeline.units import step_unit_to_meters

logger = logging.getLogger(__name__)


def run(config: StpPipelineConfig, *, case_dir: Path) -> None:
    """Execute the full STEP -> blockMeshDict pipeline.

    Parameters
    ----------
    config:
        Frozen :class:`StpPipelineConfig` carrying every pipeline knob.
        Construct it directly (validation runs in ``__post_init__``).
    case_dir:
        Root directory of the OpenFOAM case.

    See the field docstrings on :class:`StpPipelineConfig` for the meaning
    of individual parameters.
    """
    _run_impl(
        case_dir=case_dir,
        origin=config.origin,
        tol=config.tol,
        n_samples=config.n_samples,
        name_collision=config.name_collision,
        strict=config.strict,
        overwrite=config.overwrite,
        default_patch_name=config.default_patch_name,
        default_patch_name_explicit=config.default_patch_name_explicit,
        fractions=config.fractions,
        cell_conflict=config.cell_conflict,
        use_legacy_cell_count=config.use_legacy_cell_count,
        density=config.density,
        min_cell_count=config.min_cell_count,
        block_overrides=dict(config.block_overrides),
        convert_to_meters=config.convert_to_meters,
    )


def _run_impl(
    case_dir: Path,
    origin: tuple[float, float, float],
    tol: float,
    n_samples: int,
    name_collision: str,
    strict: bool,
    overwrite: bool,
    default_patch_name: str = "defaultFaces",
    default_patch_name_explicit: bool = False,
    fractions: tuple[float, float, float] | None = None,
    cell_conflict: str = "warn-max",
    use_legacy_cell_count: bool = False,
    density: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_cell_count: int | None = None,
    block_overrides: dict[str, tuple[int, int, int]] | None = None,
    convert_to_meters: float | None = None,
) -> None:
    """Internal kwargs-style entry point. See :class:`StpPipelineConfig`."""
    case_dir = Path(case_dir)

    system_dir = case_dir / "system"
    geometry_dir = case_dir / "constant" / "geometry"

    if not system_dir.is_dir():
        raise FileNotFoundError(f"system/ directory not found: {system_dir}")

    stp_path = find_single_step_file(geometry_dir)
    bmd_path = system_dir / "blockMeshDict"
    file_exists = bmd_path.exists()

    # --- Three-path model ---
    preserved_ctm: float | None = None
    if file_exists:
        bak_path = bmd_path.with_suffix(".bak")
        if bak_path.is_symlink():
            logger.warning(
                "Backup target %s is a symlink; skipping backup to avoid following it.",
                bak_path,
            )
        else:
            shutil.copy2(bmd_path, bak_path)
            logger.info("Backed up existing blockMeshDict to %s", bak_path)

        if overwrite:
            existing = BlockMeshDict(bmd_path)
            preserved_ctm = existing.convertToMeters
            bmd = BlockMeshDict()
            append_mode = False
            logger.info(
                "Overwrite mode: existing content discarded (preserving "
                "convertToMeters=%s)",
                preserved_ctm,
            )
        else:
            bmd = BlockMeshDict(bmd_path)
            append_mode = True
            logger.info(
                "Append mode: %d existing vertices, %d blocks, %d edges loaded",
                len(bmd.vertices),
                len(bmd.blocks),
                len(bmd.edges),
            )
    else:
        bmd = BlockMeshDict()
        append_mode = False

    prev_v_count = len(bmd.vertices)
    prev_b_count = len(bmd.blocks)
    prev_e_count = len(bmd.edges)

    if not append_mode or default_patch_name_explicit:
        bmd.default_patch.name = default_patch_name

    # Late lookup so tests that ``patch.object(stp_pipeline, "load_step", ...)``
    # at the package level continue to override the call site (binding
    # ``load_step`` at import time would prevent patching).
    from meshing_utils.operations import stp_pipeline as _pkg
    candidates, pool, unit = _pkg.load_step(stp_path)
    logger.info("Loaded %d solid(s) from %s (unit=%s)", len(candidates), stp_path.name, unit)

    # --- Resolve convertToMeters ---
    # Priority: explicit override > preserved value (overwrite mode) >
    # already-loaded value (append mode) > STEP-unit-derived (new file).
    if convert_to_meters is not None:
        bmd.convertToMeters = float(convert_to_meters)
        logger.info(
            "convertToMeters=%s (from explicit --convertToMeters)",
            bmd.convertToMeters,
        )
    elif preserved_ctm is not None:
        bmd.convertToMeters = preserved_ctm
        logger.info(
            "convertToMeters=%s (preserved from existing blockMeshDict)",
            bmd.convertToMeters,
        )
    elif append_mode:
        logger.info(
            "convertToMeters=%s (from existing blockMeshDict)",
            bmd.convertToMeters,
        )
    else:
        derived = step_unit_to_meters(unit)
        if derived is None:
            logger.warning(
                "STEP LENGTH_UNIT '%s' is unknown; defaulting convertToMeters=1.0. "
                "Use --convertToMeters to override.",
                unit,
            )
        else:
            bmd.convertToMeters = derived
            logger.info(
                "convertToMeters=%s (derived from STEP unit '%s')",
                bmd.convertToMeters,
                unit,
            )

    if origin != (0.0, 0.0, 0.0):
        apply_origin_shift(candidates, pool, origin)

    base_block_idx = bmd.next_block_index() if append_mode else 0

    orderings: list[list[int]] = []
    all_vertex_coords: list[list[tuple[float, float, float]]] = []
    all_vertex_names_per_block: list[list[str]] = []

    for solid_idx, candidate in enumerate(candidates):
        validate_hex(candidate)
        ordering = order_hex_vertices(candidate, pool)
        ordering = ensure_right_handed(ordering, pool)
        ordering = enforce_openfoam_face_convention(ordering, pool)
        block_label = candidate.label or f"solid{solid_idx}"
        assert_block_face_normals_outward(ordering, pool, block_label=block_label)
        orderings.append(ordering)

        block_coords = [pool.coord_at(pool_idx) for pool_idx in ordering]
        all_vertex_coords.append(block_coords)

        vertex_names: list[str] = []
        for pool_idx in ordering:
            coord = pool.coord_at(pool_idx)
            vname = bmd.find_or_add_vertex(coord, tol=tol)
            vertex_names.append(vname)
        all_vertex_names_per_block.append(vertex_names)

    if append_mode:
        strategy = None
        new_block_id_offset = 0
    elif use_legacy_cell_count:
        strategy = PropagatedCellCountStrategy(
            fractions=fractions,
            cell_conflict=cell_conflict,
        )
        strategy.bind_context(
            all_vertex_coords=all_vertex_coords,
            all_vertex_names=all_vertex_names_per_block,
            existing_cells=None,
        )
        new_block_id_offset = 0
    else:
        block_names_for_strategy: list[str] = [
            candidate.label or f"block{base_block_idx + solid_idx}"
            for solid_idx, candidate in enumerate(candidates)
        ]
        strategy = EuclideanProjectedCellCountStrategy(
            density=density,
            min_cell_count=min_cell_count,
            block_overrides=block_overrides,
        )
        strategy.bind_context(
            all_vertex_coords=all_vertex_coords,
            all_vertex_names=all_vertex_names_per_block,
            block_names=block_names_for_strategy,
        )
        new_block_id_offset = 0

    edge_origin: dict[frozenset[str], str] = {}
    if append_mode:
        for e in bmd.edges:
            key: frozenset[str] = frozenset({e.v_start, e.v_end})
            edge_origin.setdefault(key, "<pre-existing>")

    for solid_idx, candidate in enumerate(candidates):
        vertex_names = all_vertex_names_per_block[solid_idx]
        block_coords = all_vertex_coords[solid_idx]
        ordering = orderings[solid_idx]

        label = candidate.label or f"block{base_block_idx + solid_idx}"
        block_name = resolve_block_name(bmd, label, name_collision)
        if block_name != label:
            logger.info("Renamed solid '%s' -> block '%s'", label, block_name)

        if strategy is None:
            counts = (1, 1, 1)
            grading = "simpleGrading (1 1 1)"
        else:
            strategy_block_id = new_block_id_offset + solid_idx
            counts = strategy.counts_for_block(
                strategy_block_id, block_coords, candidate.edge_curves
            )
            grading = strategy.grading_for_block(
                strategy_block_id, block_coords, candidate.edge_curves
            )

        grading_parts = grading.split(None, 1)
        grading_type = grading_parts[0] if grading_parts else "simpleGrading"
        grading_inner = grading_parts[1].strip("() ") if len(grading_parts) > 1 else "1 1 1"
        grading_def = [float(v) for v in grading_inner.split()]

        block = Block(
            name_or_string=block_name,
            vertices=vertex_names,
            cells=list(counts),
            grading_type=grading_type,
            grading_def=grading_def,
        )
        bmd.blocks.add(block)

        _line_sentinel = CurveInfo(kind="line", support_points=[])
        for idx_a, idx_b in candidate.edges:
            curve_key: frozenset[int] = frozenset({idx_a, idx_b})
            curve_info = candidate.edge_curves.get(curve_key, _line_sentinel)
            ca = pool.coord_at(idx_a)
            cb = pool.coord_at(idx_b)
            na = bmd.find_or_add_vertex(ca, tol=tol)
            nb = bmd.find_or_add_vertex(cb, tol=tol)
            add_edge_with_conflict_check(
                bmd=bmd,
                name_a=na,
                name_b=nb,
                coord_a=ca,
                coord_b=cb,
                curve_info=curve_info,
                block_name=block_name,
                edge_origin=edge_origin,
                strict=strict,
                logger=logger,
            )

    # --- Global face consistency check ---
    combined_orderings: list[list[int]] = []
    combined_faces_per_block: list[list[tuple]] = []
    for blk in bmd.blocks:
        ord_ids, face_tuples = _block_to_ordering_and_faces(blk, bmd)
        combined_orderings.append(ord_ids)
        combined_faces_per_block.append(face_tuples)
    check_global_face_consistency(combined_orderings, combined_faces_per_block)

    bmd.write(bmd_path)
    logger.info("Written blockMeshDict to %s", bmd_path)

    if append_mode:
        new_v_count = len(bmd.vertices) - prev_v_count
        new_b_count = len(bmd.blocks) - prev_b_count
        new_e_count = len(bmd.edges) - prev_e_count
        logger.info(
            "Append complete: added %d new vertices, %d new blocks, %d new edges",
            new_v_count,
            new_b_count,
            new_e_count,
        )
