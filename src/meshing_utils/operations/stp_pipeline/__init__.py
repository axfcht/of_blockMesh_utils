"""STEP -> blockMeshDict pipeline subpackage.

The legacy single-module path ``meshing_utils.operations.stp_pipeline``
still resolves: this ``__init__.py`` re-exports every symbol previously
exposed at that module path so existing imports keep working.

Sub-modules:

* :mod:`.units` — STEP ``LENGTH_UNIT`` -> ``convertToMeters`` mapping.
* :mod:`.parsing` — ``parse_origin`` / ``parse_fractions``.
* :mod:`.edges` — OCC edge classification.
* :mod:`.loading` — STEP file loading + name resolution.
* :mod:`.transforms` — origin shift on the loaded pool.
* :mod:`.conflicts` — edge-conflict detection during BMD assembly.
* :mod:`.assembly` — block naming and ordering helpers.
* :mod:`.config` — :class:`StpPipelineConfig`.
* :mod:`.pipeline` — :func:`run` and the monolithic ``_run_impl``.
"""

from __future__ import annotations

from meshing_utils.operations.stp_pipeline.assembly import (
    _block_to_ordering_and_faces,
    resolve_block_name,
)
from meshing_utils.operations.stp_pipeline.config import StpPipelineConfig
from meshing_utils.operations.stp_pipeline.conflicts import (
    _format_curved_vs_curved_conflict,
    _format_line_vs_curved_conflict,
    _squared_distance,
    add_edge_with_conflict_check,
)
from meshing_utils.operations.stp_pipeline.edges import (
    _sample_inner_points,
    classify_edge,
)
from meshing_utils.operations.stp_pipeline.loading import (
    _explore_solids,
    _is_hex_topology,
    _ordered_edge_vertices,
    _read_step_solid_names,
    _read_step_xcaf,
    _solid_to_hex_candidate,
    load_step,
)
from meshing_utils.operations.stp_pipeline.parsing import (
    _FRACTIONS_RE,
    _ORIGIN_RE,
    parse_fractions,
    parse_origin,
)
from meshing_utils.operations.stp_pipeline.pipeline import _run_impl, run
from meshing_utils.operations.stp_pipeline.transforms import apply_origin_shift
from meshing_utils.operations.stp_pipeline.units import (
    _STEP_UNIT_TO_METERS,
    step_unit_to_meters,
)

__all__ = [
    "StpPipelineConfig",
    "add_edge_with_conflict_check",
    "apply_origin_shift",
    "classify_edge",
    "load_step",
    "parse_fractions",
    "parse_origin",
    "resolve_block_name",
    "run",
    "step_unit_to_meters",
]
