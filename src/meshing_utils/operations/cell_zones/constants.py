"""Public constants and result containers for cell-zone assignment."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLING_CENTROID = "centroid"
SAMPLING_INSET = "inset"
VALID_SAMPLING_STRATEGIES = (SAMPLING_CENTROID, SAMPLING_INSET)

VOTE_MAJORITY = "majority"
VALID_VOTE_POLICIES = (VOTE_MAJORITY,)


@dataclass(frozen=True)
class BlockSolidCounts:
    """Per-solid classification counts for a single hex block.

    Carries the IN / ON / OUT tallies for the interior sample set
    (centroid + insets) and the vertex sample set (8 raw vertices).
    """

    interior_in: int
    interior_on: int
    interior_out: int
    vertex_in: int
    vertex_on: int
    vertex_out: int

    @property
    def vertex_hits(self) -> int:
        """Combined ``vertex_in + vertex_on`` count."""
        return self.vertex_in + self.vertex_on


class _StageANotApplicable:
    """Singleton sentinel returned by ``_resolve_vertex_dominant`` when Stage A
    cannot decide because no solid reaches the vertex-dominant threshold."""


STAGE_A_NOT_APPLICABLE = _StageANotApplicable()
