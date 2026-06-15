"""OCC-dependent extraction paths A'/A''/A''' and the OCC mapping diagnoser.

These helpers all require an active ``STEPControl_Reader`` and the OCC
``WS()`` chain. They are called by :func:`extract_solid_names` in
:mod:`.resolution` in fall-through order.
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

from meshing_utils.io.step_text_scan import BrepSolidEntry

logger = logging.getLogger(__name__)

# STEP entity type names that represent solid BREP geometry.
_BREP_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "StepShape_ManifoldSolidBrep",
        "StepShape_BrepWithVoids",
        "StepShape_FacetedBrep",
        "StepShape_ShellBasedSurfaceModel",
    }
)

# Mapping from STEP file token (upper-cased) to OCC class name string.
_STEP_TOKEN_TO_OCC_TYPE: dict[str, str] = {
    "MANIFOLD_SOLID_BREP":       "StepShape_ManifoldSolidBrep",
    "BREP_WITH_VOIDS":           "StepShape_BrepWithVoids",
    "FACETED_BREP":              "StepShape_FacetedBrep",
    "SHELL_BASED_SURFACE_MODEL": "StepShape_ShellBasedSurfaceModel",
}


def _read_entity_name(entity) -> str | None:
    """Read the first string argument (``Name``) of a STEP entity, or ``None``.

    Handles the various return shapes of ``entity.Name()`` across OCC/OCP
    builds: a ``TCollection_HAsciiString`` handle (``ToCString``), an
    extended-string handle (``ToExtString``), or a plain ``str``. Null
    handles and empty/whitespace names yield ``None``.
    """
    if entity is None or not hasattr(entity, "Name"):
        return None
    try:
        handle = entity.Name()
    except Exception as exc:
        logger.debug("entity.Name() failed: %s", exc)
        return None
    if handle is None:
        return None
    try:
        if hasattr(handle, "IsNull") and handle.IsNull():
            return None
    except Exception:
        pass
    value: str | None = None
    if hasattr(handle, "ToCString"):
        try:
            value = handle.ToCString()
        except Exception:
            value = None
    elif hasattr(handle, "ToExtString"):
        try:
            value = handle.ToExtString()
        except Exception:
            value = None
    elif isinstance(handle, str):
        value = handle
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _extract_via_entity_name(transfer_reader, solids: list) -> list[str | None]:
    """Read each solid's name directly from its source STEP entity (Path A0).

    For every solid, ``EntityFromShapeResult`` returns the STEP entity that
    produced it; its ``Name()`` carries the relevant name — the BREP solid
    name for single parts (e.g. ``'hex_1'``) or the NAUO occurrence name for
    assembly instances (e.g. ``'Aussenring:1'``). Because this is a per-solid
    query keyed on the exact result shape, it cannot mis-correlate solids and
    names the way the positional / entity-number fallback paths can.

    Returns one name (or ``None``) per solid, in input order.
    """
    results: list[str | None] = []
    for solid in solids:
        name: str | None = None
        try:
            entity = transfer_reader.EntityFromShapeResult(solid, 1)
            name = _read_entity_name(entity)
        except Exception as exc:
            logger.debug("Path A0 (entity_name) failed for solid: %s", exc)
        results.append(name)
    return results


def _get_external_file_id_from_internal(model, internal_number: int) -> int | None:
    """Convert an OCC-internal entity number to the external STEP file id.

    Uses ``Interface_InterfaceModel.StringLabel`` which typically returns
    a handle to a ``TCollection_HAsciiString`` like ``'#56'`` or ``'56'``.
    Returns ``None`` when unavailable or unparseable.
    """
    if model is None or internal_number is None or internal_number <= 0:
        return None

    try:
        label_handle = model.StringLabel(internal_number)
        if label_handle is None:
            return None
        if hasattr(label_handle, "IsNull") and label_handle.IsNull():
            return None
        if hasattr(label_handle, "ToCString"):
            label_str = label_handle.ToCString()
        elif isinstance(label_handle, str):
            label_str = label_handle
        else:
            return None
    except Exception as exc:
        logger.debug("model.StringLabel(%d) failed: %s", internal_number, exc)
        return None

    if not label_str:
        return None

    match = re.match(r"^\s*#?(\d+)\s*$", label_str)
    if not match:
        return None
    return int(match.group(1))


def _extract_via_step_id(
    transfer_reader,
    model,
    solids: list,
    brep_name_map: dict[int, str],
) -> list[tuple[str | None, int | None]]:
    """Look up the name for each solid via its external STEP file id (Path A').

    Returns one ``(name, external_file_id)`` tuple per solid; either
    value may be ``None`` when unavailable.
    """
    results: list[tuple[str | None, int | None]] = []
    for solid in solids:
        name: str | None = None
        file_id: int | None = None
        try:
            entity = transfer_reader.EntityFromShapeResult(solid, 1)
            if entity is not None:
                try:
                    internal = model.Number(entity)
                except Exception as exc:
                    logger.debug("model.Number failed for entity: %s", exc)
                    internal = None
                file_id = _get_external_file_id_from_internal(model, internal)
                if file_id is not None and file_id in brep_name_map:
                    name = brep_name_map[file_id]
        except Exception as exc:
            logger.debug("Path A' failed for solid: %s", exc)
        results.append((name, file_id))
    return results


def _map_solids_via_model_iteration(
    reader,
    solids: list,
    brep_name_map: dict[int, str],
) -> dict[int, str]:
    """Iterate all BREP-solid entities in the STEP model and match by geometry (Path A'').

    For every entity in the OCC model that represents a BREP solid type,
    the produced ``TopoDS_Shape`` is compared to each unresolved solid
    using ``IsSame``/``IsPartner``. On a match the entity's external
    STEP file id is looked up in *brep_name_map*. This path succeeds
    where ``EntityFromShapeResult`` fails (e.g. compound unpacking).

    Returns a dict ``id(solid) → name``.
    """
    result: dict[int, str] = {}

    try:
        ws = reader.WS()
        model = ws.Model()
        tp = ws.TransferReader().TransientProcess()
    except Exception as exc:
        logger.debug("_map_solids_via_model_iteration: failed to obtain WS/model/tp: %s", exc)
        return result

    try:
        nb_entities = model.NbEntities()
    except Exception as exc:
        logger.debug("_map_solids_via_model_iteration: NbEntities() failed: %s", exc)
        return result

    for i in range(1, nb_entities + 1):
        try:
            entity = model.Value(i)
        except Exception as exc:
            logger.debug("model.Value(%d) failed: %s", i, exc)
            continue

        if entity is None:
            continue

        try:
            raw = entity.DynamicType().Name()
            type_name = raw.decode("ascii") if isinstance(raw, bytes) else raw
        except Exception:
            continue

        if type_name not in _BREP_TYPE_NAMES:
            continue

        shape = None
        try:
            binder = tp.Find(entity)
            if binder is not None:
                binder_module = (type(binder).__module__ or "")
                is_occ_handle = binder_module.startswith(("OCC.", "OCP."))
                try:
                    if not is_occ_handle:
                        raise TypeError("binder is not an OCC handle; skipping DownCast")
                    from OCP.TransferBRep import TransferBRep_ShapeBinder  # type: ignore[import]
                    _DownCast = (
                        getattr(TransferBRep_ShapeBinder, "DownCast_s", None)
                        or TransferBRep_ShapeBinder.DownCast
                    )
                    shape_binder = _DownCast(binder)
                    if shape_binder is not None:
                        is_null = False
                        try:
                            null_val = shape_binder.IsNull()
                            is_null = bool(null_val)
                        except Exception:
                            pass
                        if not is_null:
                            shape = shape_binder.Result()
                except Exception as exc:
                    logger.debug(
                        "TransferBRep_ShapeBinder.DownCast failed for entity %d: %s", i, exc
                    )
                if shape is None:
                    try:
                        shape = binder.Result()
                    except Exception as exc:
                        logger.debug("binder.Result() fallback failed for entity %d: %s", i, exc)
        except Exception as exc:
            logger.debug("tp.Find() failed for entity %d: %s", i, exc)

        if shape is None:
            try:
                transient = tp.FindTransient(entity)
                if transient is not None and hasattr(transient, "Shape"):
                    shape = transient.Shape()
            except Exception as exc:
                logger.debug("tp.FindTransient() failed for entity %d: %s", i, exc)

        if shape is None:
            continue

        entity_name: str | None = None
        file_id = _get_external_file_id_from_internal(model, i)
        if file_id is not None and file_id in brep_name_map:
            entity_name = brep_name_map[file_id]

        for solid in solids:
            if id(solid) in result:
                continue
            try:
                matched = shape.IsSame(solid)
            except Exception:
                matched = False
            if not matched:
                try:
                    matched = shape.IsPartner(solid)
                except Exception:
                    matched = False
            if matched:
                if entity_name is not None:
                    result[id(solid)] = entity_name
                break

    return result


def _ordered_match_by_type(
    reader,
    solids: list,
    brep_entries: list[BrepSolidEntry],
) -> dict[int, str]:
    """Type-sensitive positional correlation (Path A''').

    Solids are grouped by OCC BREP type; within each type they are
    sorted by internal entity number and correlated 1:1 with file-order
    entries of the same type. Mismatched counts cause that type to be
    skipped with a warning. Returns ``id(solid) → name``.
    """
    result: dict[int, str] = {}

    try:
        ws = reader.WS()
        model = ws.Model()
        transfer_reader = ws.TransferReader()
    except Exception as exc:
        logger.debug("_ordered_match_by_type: failed to obtain WS/model/transfer_reader: %s", exc)
        return result

    file_by_type: dict[str, list[BrepSolidEntry]] = {}
    for entry in brep_entries:
        occ_type = _STEP_TOKEN_TO_OCC_TYPE.get(entry.type_token)
        if occ_type is None:
            continue
        file_by_type.setdefault(occ_type, []).append(entry)

    solid_info: list[tuple[int, str, int]] = []
    for solid in solids:
        try:
            entity = transfer_reader.EntityFromShapeResult(solid, 1)
            if entity is None:
                continue
            try:
                raw = entity.DynamicType().Name()
                occ_type = raw.decode("ascii") if isinstance(raw, bytes) else raw
            except Exception as exc:
                logger.debug("Ordered-match DynamicType().Name() failed: %s", exc)
                continue
            if occ_type not in _BREP_TYPE_NAMES:
                continue
            try:
                occ_number = model.Number(entity)
            except Exception as exc:
                logger.debug("Ordered-match model.Number() failed: %s", exc)
                continue
            solid_info.append((id(solid), occ_type, occ_number))
        except Exception as exc:
            logger.debug("Ordered-match entity lookup failed: %s", exc)

    grouped: dict[str, list[tuple[int, int]]] = {}
    for sid, typ, num in solid_info:
        grouped.setdefault(typ, []).append((num, sid))
    occ_by_type: dict[str, list[int]] = {}
    for typ, lst in grouped.items():
        lst.sort()
        occ_by_type[typ] = [sid for (_n, sid) in lst]

    for occ_type, sid_list in occ_by_type.items():
        file_list = file_by_type.get(occ_type, [])
        if len(file_list) != len(sid_list):
            logger.warning(
                "Ordered-match count mismatch for type %s: file=%d occ=%d — skipping type",
                occ_type,
                len(file_list),
                len(sid_list),
            )
            continue
        for i, sid in enumerate(sid_list):
            result[sid] = file_list[i].name

    return result


def _diagnose_occ_mapping(
    reader,
    solids: list,
    step_path: Path,
) -> list[dict]:
    """Collect debug information about OCC entity↔solid mappings.

    Called at DEBUG level after all primary naming paths have failed,
    to allow future diagnosis without re-running interactively.
    """
    if reader is None:
        return []

    diagnostics: list[dict] = []

    try:
        ws = reader.WS()
        transfer_reader = ws.TransferReader()
        model = ws.Model()
    except Exception as exc:
        logger.debug("_diagnose_occ_mapping: cannot access WS: %s", exc)
        return []

    for idx, solid in enumerate(solids):
        info: dict = {
            "index": idx,
            "step_path": str(step_path),
            "entity_type": None,
            "entity_number": None,
            "has_name_method": False,
            "name_value": None,
        }
        try:
            entity = transfer_reader.EntityFromShapeResult(solid, 1)
            if entity is not None:
                try:
                    raw = entity.DynamicType().Name()
                    info["entity_type"] = raw.decode("ascii") if isinstance(raw, bytes) else raw
                except Exception:
                    info["entity_type"] = "<unknown>"
                with contextlib.suppress(Exception):
                    info["entity_number"] = model.Number(entity)
                info["has_name_method"] = hasattr(entity, "Name")
                try:
                    name_handle = entity.Name()
                    if name_handle is None:
                        info["name_value"] = None
                    elif hasattr(name_handle, "ToCString"):
                        info["name_value"] = name_handle.ToCString()
                    elif isinstance(name_handle, str):
                        info["name_value"] = name_handle
                except Exception as exc:
                    info["name_value"] = f"<error: {exc}>"
        except Exception as exc:
            info["entity_type"] = f"<EntityFromShapeResult error: {exc}>"

        diagnostics.append(info)

    return diagnostics
