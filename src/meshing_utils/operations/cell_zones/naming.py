"""Zone-name sanitisation and uniqueness allocation."""

from __future__ import annotations

import re


def _sanitize_zone_name(raw: str | None, fallback_idx: int) -> str:
    """Convert a raw solid label to a valid OpenFOAM identifier.

    ``None`` / empty → ``f"solid{fallback_idx}"``.  Non-word characters
    are replaced with ``_``; a leading digit gets a ``z_`` prefix.
    """
    if not raw:
        return f"solid{fallback_idx}"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not cleaned:
        return f"solid{fallback_idx}"
    if cleaned[0].isdigit():
        cleaned = "z_" + cleaned
    return cleaned


def _assign_unique_zone_names(solid_label_pairs: list[tuple]) -> list[str]:
    """Produce a unique zone name per solid, suffixing duplicates.

    Assignment is solid-index-based (not raw-label-based) so two solids
    with the same sanitised name always get deterministic suffixes
    ``_2``, ``_3``, ... regardless of input order.
    """
    result: list[str] = []
    used: dict[str, int] = {}
    allocated: set[str] = set()

    for idx, (_, raw_label) in enumerate(solid_label_pairs):
        base = _sanitize_zone_name(raw_label, idx)
        if base not in allocated:
            used[base] = 2
            allocated.add(base)
            result.append(base)
        else:
            counter = used.get(base, 2)
            candidate = f"{base}_{counter}"
            while candidate in allocated:
                counter += 1
                candidate = f"{base}_{counter}"
            used[base] = counter + 1
            allocated.add(candidate)
            result.append(candidate)

    return result
