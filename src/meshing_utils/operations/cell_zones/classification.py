"""Sample generation, per-solid classification, majority vote, perturbation."""

from __future__ import annotations

import logging
import math

from meshing_utils.geometry.containment import (
    AABB,
    STATE_IN,
    STATE_ON,
    aabbs_overlap,
    classify_point_in_solid,
    classify_point_with_classifier,
    compute_block_centroid,
    compute_block_sample_sets,
    compute_points_aabb,
)
from meshing_utils.operations.cell_zones.constants import (
    SAMPLING_CENTROID,
    SAMPLING_INSET,
    BlockSolidCounts,
)

logger = logging.getLogger(__name__)


def _compute_samples_for_block(
    coords: list[tuple[float, float, float]],
    sampling_strategy: str,
    inset_factor: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return ``(interior_samples, vertex_samples)`` for a block.

    ``"centroid"``: interior holds the centroid only, vertex set is empty
    (Stage A is always skipped). ``"inset"``: 9 interior samples and the
    8 raw vertices.
    """
    if sampling_strategy == SAMPLING_CENTROID:
        return ([compute_block_centroid(coords)], [])
    if sampling_strategy == SAMPLING_INSET:
        return compute_block_sample_sets(coords, inset_factor)
    raise ValueError(f"unknown sampling_strategy: {sampling_strategy!r}")


def _classify_samples_against_solid(
    solid,
    samples: list[tuple[float, float, float]],
    tol: float,
    classifier=None,
) -> tuple[int, int, int]:
    """Return ``(in_count, on_count, out_count)`` for *samples* against *solid*.

    Uses *classifier* when provided to avoid repeated OCC initialisation.
    """
    in_c = on_c = out_c = 0
    for P in samples:
        if classifier is not None:
            state = classify_point_with_classifier(classifier, P, tol)
        else:
            state = classify_point_in_solid(solid, P, tol)
        if state == STATE_IN:
            in_c += 1
        elif state == STATE_ON:
            on_c += 1
        else:
            out_c += 1
    return (in_c, on_c, out_c)


def _classify_block_two_pass(
    interior_samples: list[tuple[float, float, float]],
    vertex_samples: list[tuple[float, float, float]],
    solid_label_pairs: list[tuple],
    tol: float,
    solid_aabbs: list[AABB] | None = None,
    classifiers: list | None = None,
) -> list[BlockSolidCounts]:
    """Return per-solid :class:`BlockSolidCounts` with sharpened early-exit.

    Early-exit triggers only when a solid receives all interior samples
    as IN *and* all vertex samples as IN or ON (full containment). The
    optional AABB filter skips solids whose bounding box does not
    overlap the block bounding box. Reuses *classifiers* when given.
    """
    interior_total = len(interior_samples)
    vertex_total = len(vertex_samples)
    results: list[BlockSolidCounts] = []
    early_exit_active = False

    if solid_aabbs is not None:
        all_samples = interior_samples + vertex_samples
        block_aabb = compute_points_aabb(all_samples)
    else:
        block_aabb = None

    for idx, (solid, _) in enumerate(solid_label_pairs):
        if early_exit_active:
            results.append(BlockSolidCounts(
                0, 0, interior_total,
                0, 0, vertex_total,
            ))
            continue

        if (
            solid_aabbs is not None
            and block_aabb is not None
            and not aabbs_overlap(block_aabb, solid_aabbs[idx])
        ):
            logger.debug(
                "AABB filter: block AABB disjoint from solid %d — skipping", idx
            )
            results.append(BlockSolidCounts(
                0, 0, interior_total,
                0, 0, vertex_total,
            ))
            continue

        classifier = classifiers[idx] if classifiers is not None else None

        i_in, i_on, i_out = _classify_samples_against_solid(
            solid, interior_samples, tol, classifier
        )
        v_in, v_on, v_out = _classify_samples_against_solid(
            solid, vertex_samples, tol, classifier
        )
        results.append(BlockSolidCounts(i_in, i_on, i_out, v_in, v_on, v_out))

        if (
            interior_total > 1
            and i_in == interior_total
            and vertex_total > 0
            and (v_in + v_on) == vertex_total
        ):
            early_exit_active = True
            logger.debug("Early exit at solid %d (full containment)", idx)

    return results


def _majority_vote(
    counts_per_solid: list[tuple[int, int, int]],
) -> tuple[list[int], list[int]]:
    """Apply majority vote over per-solid classification counts.

    Preference order: IN > ON > OUT. Returns ``(in_candidates,
    on_candidates)`` — indices of solids with the maximum IN count (if
    > 0) or maximum ON count (otherwise). Both lists are empty when all
    counts are zero.
    """
    max_in = max((c[0] for c in counts_per_solid), default=0)
    if max_in > 0:
        in_cands = [i for i, c in enumerate(counts_per_solid) if c[0] == max_in]
        return (in_cands, [])
    max_on = max((c[1] for c in counts_per_solid), default=0)
    if max_on > 0:
        on_cands = [i for i, c in enumerate(counts_per_solid) if c[1] == max_on]
        return ([], on_cands)
    return ([], [])


def _perturbation_fallback(
    samples: list[tuple[float, float, float]],
    on_cands: list[int],
    solid_label_pairs: list[tuple],
    tol: float,
    epsilon: float,
) -> int | None:
    """Re-test *on_cands* with radially perturbed sample points.

    Each sample is shifted by *epsilon* along the unit vector from the
    centroid (``samples[0]``); the centroid itself shifts by
    ``(+eps, +eps, +eps)``. Returns the winning solid index or ``None``
    when the perturbation is inconclusive.
    """
    M = samples[0]
    perturbed: list[tuple[float, float, float]] = [
        (M[0] + epsilon, M[1] + epsilon, M[2] + epsilon)
    ]
    for P in samples[1:]:
        dx, dy, dz = P[0] - M[0], P[1] - M[1], P[2] - M[2]
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L < 1e-30:
            perturbed.append((P[0] + epsilon, P[1] + epsilon, P[2] + epsilon))
        else:
            perturbed.append((
                P[0] + epsilon * dx / L,
                P[1] + epsilon * dy / L,
                P[2] + epsilon * dz / L,
            ))

    pert_counts: list[tuple[int, int]] = []
    for idx in on_cands:
        solid = solid_label_pairs[idx][0]
        in_c, _, _ = _classify_samples_against_solid(solid, perturbed, tol)
        pert_counts.append((idx, in_c))

    max_in = max((c[1] for c in pert_counts), default=0)
    winners = [idx for (idx, in_c) in pert_counts if in_c == max_in and in_c > 0]
    if len(winners) == 1:
        return winners[0]
    return None
