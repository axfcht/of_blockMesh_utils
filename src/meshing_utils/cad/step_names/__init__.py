"""STEP solid-name extraction subpackage.

Re-exports every symbol previously exposed at
``meshing_utils.cad.step_names`` so existing imports keep working.

Sub-modules:

* :mod:`.named_solid` — :class:`NamedSolid` data model.
* :mod:`.parsing`     — pure-Python sanitisation / dedup / regex parsers.
* :mod:`.paths`       — OCC-dependent extraction paths A'/A''/A''' + diagnoser.
* :mod:`.resolution`  — :func:`extract_solid_names` five-path orchestrator.
"""

from __future__ import annotations

from meshing_utils.cad.step_names.named_solid import NamedSolid
from meshing_utils.cad.step_names.parsing import (
    _build_assembly_map,
    _build_brep_name_map,
    _dedupe,
    _is_unusable,
    _sanitize,
    _strip_occurrence_index,
)
from meshing_utils.cad.step_names.paths import (
    _BREP_TYPE_NAMES,
    _STEP_TOKEN_TO_OCC_TYPE,
    _diagnose_occ_mapping,
    _extract_via_entity_name,
    _extract_via_step_id,
    _get_external_file_id_from_internal,
    _map_solids_via_model_iteration,
    _ordered_match_by_type,
    _read_entity_name,
)
from meshing_utils.cad.step_names.resolution import extract_solid_names

__all__ = [
    "NamedSolid",
    "extract_solid_names",
]
