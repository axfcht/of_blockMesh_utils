"""Stage A (vertex-dominant) and Stage B (interior-fallback) winner selection."""

from __future__ import annotations

import logging

from meshing_utils.geometry.containment import VERTEX_DOMINANT_THRESHOLD
from meshing_utils.operations.cell_zones.classification import (
    _majority_vote,
    _perturbation_fallback,
)
from meshing_utils.operations.cell_zones.constants import (
    STAGE_A_NOT_APPLICABLE,
    BlockSolidCounts,
    _StageANotApplicable,
)

logger = logging.getLogger(__name__)


def _resolve_vertex_dominant(
    block,
    counts: list[BlockSolidCounts],
    zone_names: list[str],
    strict: bool,
) -> int | _StageANotApplicable:
    """Stage A: choose the winning solid based on vertex hit counts.

    Returns the chosen solid index, or :data:`STAGE_A_NOT_APPLICABLE`
    when no solid reaches :data:`VERTEX_DOMINANT_THRESHOLD` vertex hits.

    Tie-break order: ``vertex_hits``, ``vertex_on``, ``vertex_in``,
    ``interior_in``. Remaining ambiguity raises in strict mode and
    logs+falls back to the first candidate otherwise.
    """
    if not counts:
        return STAGE_A_NOT_APPLICABLE

    vertex_hits_per_solid = [c.vertex_hits for c in counts]
    max_hits = max(vertex_hits_per_solid, default=0)

    if max_hits < VERTEX_DOMINANT_THRESHOLD:
        return STAGE_A_NOT_APPLICABLE

    cands = [i for i, h in enumerate(vertex_hits_per_solid) if h == max_hits]

    if len(cands) > 1:
        max_on = max(counts[i].vertex_on for i in cands)
        cands = [i for i in cands if counts[i].vertex_on == max_on]

    if len(cands) > 1:
        max_vin = max(counts[i].vertex_in for i in cands)
        cands = [i for i in cands if counts[i].vertex_in == max_vin]

    if len(cands) > 1:
        max_iin = max(counts[i].interior_in for i in cands)
        cands = [i for i in cands if counts[i].interior_in == max_iin]

    if len(cands) == 1:
        logger.debug(
            "Block %r: Stage A winner: solid index %d (%s)",
            block.name,
            cands[0],
            zone_names[cands[0]],
        )
        return cands[0]

    msg = (
        f"Block {block.name!r}: Stage-A ambiguous after vertex tie-breaks: "
        f"{[zone_names[i] for i in cands]}"
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning("%s -> using first", msg)
    return cands[0]


def _resolve_interior_candidates(
    block,
    interior_samples: list[tuple[float, float, float]],
    counts: list[BlockSolidCounts],
    solid_label_pairs: list[tuple],
    zone_names: list[str],
    tol: float,
    epsilon: float,
    strict: bool,
) -> int | None:
    """Stage B: majority vote over interior samples with perturbation fallback.

    Resolution order:

    1. Single IN candidate → direct win.
    2. Multiple IN candidates (tie) → strict raises, else first wins.
    3. Single ON candidate → direct win.
    4. Multiple ON candidates → perturbation; inconclusive → strict
       raises, else first ON wins.
    5. No candidates → return ``None`` (block stays unzoned).
    """
    interior_tuples = [
        (c.interior_in, c.interior_on, c.interior_out) for c in counts
    ]
    in_cands, on_cands = _majority_vote(interior_tuples)

    if len(in_cands) == 1:
        logger.debug(
            "Block %r: Stage B single IN-majority winner: solid index %d (%s)",
            block.name,
            in_cands[0],
            zone_names[in_cands[0]],
        )
        return in_cands[0]

    if len(in_cands) >= 2:
        msg = (
            f"Block {block.name!r}: ambiguous IN-majority: "
            f"{[zone_names[i] for i in in_cands]}"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s -> using first", msg)
        return in_cands[0]

    if len(on_cands) == 1:
        return on_cands[0]

    if len(on_cands) >= 2:
        chosen = _perturbation_fallback(
            interior_samples, on_cands, solid_label_pairs, tol, epsilon
        )
        if chosen is not None:
            return chosen
        msg = (
            f"Block {block.name!r}: ambiguous ON, perturbation inconclusive: "
            f"{[zone_names[i] for i in on_cands]}"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s -> using first ON", msg)
        return on_cands[0]

    return None
