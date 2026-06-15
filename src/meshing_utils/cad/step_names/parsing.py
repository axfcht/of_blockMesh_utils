"""Pure-Python helpers for STEP solid-name extraction.

Contains name sanitisation / dedup utilities and the raw STEP-text
regex parsers that do not require OCC.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from meshing_utils.io.step_text_scan import build_brep_name_map_by_file_id

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    """Return an OpenFOAM-compatible identifier derived from *name*.

    Replaces non-word characters with ``_``, collapses runs of ``_``,
    strips leading/trailing ``_``, prepends ``z_`` when the first
    character is a digit, and returns ``solid`` for an empty result.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        return "solid"
    if cleaned[0].isdigit():
        cleaned = "z_" + cleaned
    return cleaned


def _is_unusable(name: str | None) -> bool:
    """Return ``True`` when *name* should not be used as a solid name.

    Filters ``None``, empty strings, and known placeholder names
    produced by CAD exporters or OCC itself.
    """
    if name is None:
        return True
    stripped = name.strip()
    if not stripped:
        return True
    upper = stripped.upper()
    unusable_keywords = {
        "NONE",
        "PART",
        "SOLID",
    }
    if upper in unusable_keywords:
        return True
    return bool(upper.startswith("OPEN CASCADE"))


def _strip_occurrence_index(name: str) -> str:
    """Strip a trailing STEP NAUO occurrence index (``:N``) from *name*.

    Assembly occurrence names carry a colon-separated instance index, e.g.
    ``'Aussenring:1'`` or ``'zyl_1:14'``. The index is removed so that a
    single instance keeps a clean base name (``'Aussenring'``) and genuine
    duplicates are numbered by :func:`_dedupe` instead. Names without a
    trailing ``:N`` (e.g. ``'hex_1'``) are returned unchanged.
    """
    return re.sub(r":\d+$", "", name)


def _dedupe(names: list[str]) -> list[str]:
    """Resolve duplicate names by appending ``_1``, ``_2``, ... suffixes.

    The first occurrence keeps its name; subsequent occurrences receive
    incrementing numeric suffixes. A ``logging.warning`` is emitted for
    every collision.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"
            logger.warning(
                "Duplicate solid name %r → renamed to %r", name, new_name
            )
            result.append(new_name)
    return result


def _build_brep_name_map(step_path: Path) -> dict[int, str]:
    """Thin compatibility wrapper around
    :func:`meshing_utils.io.step_text_scan.build_brep_name_map_by_file_id`.
    """
    return build_brep_name_map_by_file_id(step_path)


def _build_assembly_map(step_path: Path) -> dict[str, str]:
    """Build a mapping from STEP entity id to component name using regex.

    Parses ``PRODUCT`` and ``NEXT_ASSEMBLY_USAGE_OCCURRENCE`` records
    from the raw STEP file text. Lenient: any parse error silently
    produces an empty dict rather than raising.
    """
    try:
        raw = step_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    normalised = re.sub(r"\s+", " ", raw)

    product_re = re.compile(
        r"(#\d+)\s*=\s*PRODUCT\s*\(\s*'([^']*)'\s*,",
        re.IGNORECASE,
    )
    product_names: dict[str, str] = {}
    for m in product_re.finditer(normalised):
        entity_id = m.group(1)
        name = m.group(2).strip()
        if name:
            product_names[entity_id] = name

    nauo_re = re.compile(
        r"#\d+\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\("
        r"\s*'[^']*'\s*,\s*'([^']*)'\s*,",
        re.IGNORECASE,
    )

    assembly_names: list[str] = []
    for m in nauo_re.finditer(normalised):
        component_name = m.group(1).strip()
        if component_name:
            assembly_names.append(component_name)

    result: dict[str, str] = {}
    for idx, name in enumerate(assembly_names):
        result[f"#{idx + 1}"] = name

    result.update(product_names)

    return result
