"""Five-path orchestrator: :func:`extract_solid_names`."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from meshing_utils.cad.step_names.named_solid import NamedSolid
from meshing_utils.cad.step_names.parsing import (
    _dedupe,
    _is_unusable,
    _sanitize,
    _strip_occurrence_index,
)
from meshing_utils.io.step_text_scan import (
    BrepSolidEntry,
    build_brep_name_map_by_file_id,
    parse_brep_solid_entries,
)

logger = logging.getLogger(__name__)


def extract_solid_names(reader, solids: list, step_path: Path) -> list[NamedSolid]:
    """Assign names to *solids* using the five-path strategy.

    Paths are tried in order; the first path that yields a usable
    candidate wins for each solid. Falls back to ``"solid{i}"``
    (Path C) when all other paths fail.

    Path A0  (``entity_name``): ``EntityFromShapeResult`` → ``entity.Name()``
                                read directly (per-solid, cannot mis-correlate).
    Path A'  (``step_id``):     ``EntityFromShapeResult`` → ``model.Number`` →
                                ``model.StringLabel`` → BREP name map lookup.
    Path A'' (``model_scan``):  full model iteration + geometry matching.
    Path A'''(``ordered_match``): type-sensitive positional correlation.
    Path B   (``assembly``):    NAUO component names via regex.
    Path C   (``generic``):     ``"solid{i}"`` fallback with warning log.

    Pass ``reader=None`` to skip Paths A', A'', and A''' entirely
    (Path B + C only).
    """
    # Late package-level lookup so monkeypatching of these helpers via
    # ``setattr(step_names, '_map_solids_via_model_iteration', ...)`` etc.
    # in tests reaches the call site (binding these symbols at import time
    # would otherwise prevent patching).
    from meshing_utils.cad import step_names as _pkg

    n = len(solids)
    names: list[str | None] = [None] * n
    sources: list[str] = ["generic"] * n

    brep_name_map: dict[int, str] = {}
    brep_entries: list[BrepSolidEntry] = []
    try:
        brep_name_map = build_brep_name_map_by_file_id(step_path)
        brep_entries = parse_brep_solid_entries(step_path)
    except Exception as exc:
        logger.debug("BREP file scan failed: %s", exc)

    if reader is not None:
        # Path A0 (entity_name): read each solid's name straight from its
        # source STEP entity. This is the authoritative path; it runs first
        # and the remaining OCC paths only fill solids it could not resolve.
        try:
            transfer_reader = reader.WS().TransferReader()
            entity_names = _pkg._extract_via_entity_name(transfer_reader, solids)
            for i, raw in enumerate(entity_names):
                candidate = (
                    _strip_occurrence_index(raw) if raw is not None else None
                )
                if not _is_unusable(candidate):
                    names[i] = candidate
                    sources[i] = "entity_name"
        except Exception as exc:
            logger.debug("Path A0 (entity_name) failed entirely: %s", exc)

        missing = [i for i in range(n) if names[i] is None]
        if missing and brep_name_map:
            try:
                ws = reader.WS()
                transfer_reader = ws.TransferReader()
                model = ws.Model()
                results_a_prime = _pkg._extract_via_step_id(
                    transfer_reader, model, solids, brep_name_map
                )
                for i, (name, _file_id) in enumerate(results_a_prime):
                    if names[i] is None and not _is_unusable(name):
                        names[i] = name
                        sources[i] = "step_id"
            except Exception as exc:
                logger.debug("Path A' failed entirely: %s", exc)

        missing = [i for i in range(n) if names[i] is None]
        if missing and brep_name_map:
            try:
                missing_solids = [solids[i] for i in missing]
                scan_map = _pkg._map_solids_via_model_iteration(
                    reader, missing_solids, brep_name_map
                )
                for i in missing:
                    candidate = scan_map.get(id(solids[i]))
                    if not _is_unusable(candidate):
                        names[i] = candidate
                        sources[i] = "model_scan"
            except Exception as exc:
                logger.debug("Path A'' (model_scan) failed entirely: %s", exc)

        missing = [i for i in range(n) if names[i] is None]
        if missing and brep_entries:
            try:
                ordered_map = _pkg._ordered_match_by_type(reader, solids, brep_entries)
                for i in missing:
                    candidate = ordered_map.get(id(solids[i]))
                    if not _is_unusable(candidate):
                        names[i] = candidate
                        sources[i] = "ordered_match"
            except Exception as exc:
                logger.debug("Path A''' (ordered_match) failed entirely: %s", exc)

    missing_indices = [i for i in range(n) if names[i] is None]
    if missing_indices:
        try:
            raw = step_path.read_text(encoding="utf-8", errors="ignore")
            normalised = re.sub(r"\s+", " ", raw)
            nauo_re = re.compile(
                r"#\d+\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\("
                r"\s*'[^']*'\s*,\s*'([^']*)'\s*,",
                re.IGNORECASE,
            )
            nauo_names = [
                m.group(1).strip()
                for m in nauo_re.finditer(normalised)
                if m.group(1).strip()
            ]
        except OSError:
            nauo_names = []

        if len(nauo_names) == n:
            for i in range(n):
                if names[i] is None and not _is_unusable(nauo_names[i]):
                    names[i] = nauo_names[i]
                    sources[i] = "assembly"
        elif nauo_names:
            for idx, i in enumerate(missing_indices):
                if idx < len(nauo_names) and not _is_unusable(nauo_names[idx]):
                    names[i] = nauo_names[idx]
                    sources[i] = "assembly"

    all_generic = True
    for i in range(n):
        if names[i] is None:
            names[i] = f"solid{i}"
            sources[i] = "generic"
            logger.warning(
                "Could not determine name for solid %d; using generic name %r",
                i,
                names[i],
            )
        else:
            all_generic = False

    if all_generic and n > 0 and reader is not None:
        logger.debug(
            "All %d solid(s) received generic names. Running OCC mapping diagnostics "
            "for %s",
            n,
            step_path,
        )
        diag_entries = _pkg._diagnose_occ_mapping(reader, solids, step_path)
        for entry in diag_entries:
            logger.debug(
                "  solid[%d]: entity_type=%r entity_number=%r "
                "has_name_method=%r name_value=%r",
                entry["index"],
                entry["entity_type"],
                entry["entity_number"],
                entry["has_name_method"],
                entry["name_value"],
            )

    sanitised = [_sanitize(name) for name in names]  # type: ignore[arg-type]

    deduped = _dedupe(sanitised)

    return [
        NamedSolid(solid=solids[i], name=deduped[i], source=sources[i])
        for i in range(n)
    ]
