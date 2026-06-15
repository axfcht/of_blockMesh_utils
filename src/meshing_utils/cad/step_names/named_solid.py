"""Public data model: :class:`NamedSolid`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NamedSolid:
    """A STEP solid together with its resolved name and provenance.

    ``source`` records how the name was obtained: one of ``"entity_name"``,
    ``"step_id"``, ``"model_scan"``, ``"ordered_match"``, ``"assembly"``, or
    ``"generic"``.
    """

    solid: object  # TopoDS_Solid, typed as object to avoid OCC import at class level
    name: str
    source: str = field(default="generic")
