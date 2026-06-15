"""Cell count and grading strategies for hex blocks.

Provides an abstract base class and two concrete strategies:

- :class:`PropagatedCellCountStrategy` (legacy): fraction-of-bbox heuristic.
- :class:`EuclideanProjectedCellCountStrategy` (default): L2-projected
  density on the longest parallel edge per axis class, with axis
  equivalence propagation and per-block overrides.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from meshing_utils.geometry.hex_axes import (
    AxisEquivalenceClasses,
    TopologyError,
    build_block_axes,
    compute_longest_edges_per_axis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------

class BBox:
    """Axis-aligned bounding box computed from a collection of coordinates.

    Attributes
    ----------
    min_coord:
        ``(x_min, y_min, z_min)``
    max_coord:
        ``(x_max, y_max, z_max)``
    length:
        ``(dx, dy, dz)`` — extent along each global axis.
    """

    def __init__(
        self,
        coords: Sequence[Sequence[float]],
    ) -> None:
        if not coords:
            self.min_coord = (0.0, 0.0, 0.0)
            self.max_coord = (0.0, 0.0, 0.0)
            self.length = (0.0, 0.0, 0.0)
            return

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]

        self.min_coord = (min(xs), min(ys), min(zs))
        self.max_coord = (max(xs), max(ys), max(zs))
        self.length = (
            self.max_coord[0] - self.min_coord[0],
            self.max_coord[1] - self.min_coord[1],
            self.max_coord[2] - self.min_coord[2],
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CellCountStrategy(ABC):
    """Abstract strategy for computing cell counts and grading per block."""

    @abstractmethod
    def counts_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> tuple[int, int, int]:
        """Return the (nx, ny, nz) cell count for a block.

        Parameters
        ----------
        block_id:
            Zero-based index of the block in the candidate list.
        block_vertex_coords:
            Sequence of 8 vertex coordinates for the block (may be None for
            strategies that ignore geometry).
        edge_curves:
            Mapping of edge keys to CurveInfo objects (may be None).
        """

    @abstractmethod
    def grading_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> str:
        """Return the OpenFOAM grading string for a block.

        Parameters
        ----------
        block_id:
            Zero-based index of the block in the candidate list.
        block_vertex_coords:
            Sequence of 8 vertex coordinates for the block (may be None).
        edge_curves:
            Mapping of edge keys to CurveInfo objects (may be None).
        """


# ---------------------------------------------------------------------------
# PropagatedCellCountStrategy
# ---------------------------------------------------------------------------

class PropagatedCellCountStrategy(CellCountStrategy):
    """Strategy that propagates cell counts through Union-Find equivalence classes.

    Algorithm
    ---------
    1. ``bind_context`` is called once before any ``counts_for_block`` call.
       It builds ``BlockAxis`` objects and ``AxisEquivalenceClasses`` for all
       blocks, then:

       a. **Pre-Lock Pass**: if *existing_cells* is given (append mode), the
          cells of each pre-existing block lock their equivalence classes.
          Conflicts between two locked classes are handled by *cell_conflict*.

       b. **Fallback Pass**: for each unlocked class, the representative axis
          is the one with the largest ``edge_length`` in the class.  If
          *fractions* is set, ``n = max(1, round(edge_length /
          (fractions[dominant] * bbox.length[dominant])))``.  If *fractions*
          is ``None`` or the target cell size is ``<= 0``, ``n = 1``.

    2. ``counts_for_block(block_id, ...)`` looks up the three class cell
       counts for the given block and returns them as ``(ni, nj, nk)``.

    Parameters
    ----------
    fractions:
        ``(fx, fy, fz)`` — fraction of the global bounding box extent to use
        as the target cell size per global axis.  ``None`` → ``n = 1``
        everywhere (default, equivalent to the old FixedCellCountStrategy(1)).
    cell_conflict:
        How to handle conflicts when two pre-existing blocks have different
        cell counts in the same equivalence class.
        - ``"error"``     → raise ``ValueError``
        - ``"warn-max"``  → warn, keep maximum (default)
        - ``"warn-first"``→ warn, keep first seen
    """

    def __init__(
        self,
        fractions: tuple[float, float, float] | None = None,
        cell_conflict: str = "warn-max",
    ) -> None:
        if cell_conflict not in ("error", "warn-max", "warn-first"):
            raise ValueError(
                f"cell_conflict must be 'error', 'warn-max', or 'warn-first'; "
                f"got {cell_conflict!r}"
            )
        self._fractions = fractions
        self._cell_conflict = cell_conflict
        self._class_counts: dict[int, int] = {}   # class_root → n
        self._block_counts: dict[int, tuple[int, int, int]] = {}  # block_id → (ni,nj,nk)
        self._context_bound: bool = False

    # ------------------------------------------------------------------
    # Context binding
    # ------------------------------------------------------------------

    def bind_context(
        self,
        all_vertex_coords: list[list[tuple[float, float, float]]],
        all_vertex_names: list[list[str]],
        existing_cells: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        """Bind global geometry context and compute all cell counts.

        Must be called once before any ``counts_for_block`` or
        ``grading_for_block`` call.

        Parameters
        ----------
        all_vertex_coords:
            One list of 8 vertex coordinates per block, in block order.
        all_vertex_names:
            One list of 8 vertex names per block, matching *all_vertex_coords*.
        existing_cells:
            Mapping from block_id to ``(ni, nj, nk)`` for blocks that already
            exist in a loaded ``blockMeshDict`` (append mode).  Pass ``None``
            (or empty dict) for fresh creation.
        """
        n_blocks = len(all_vertex_coords)
        if n_blocks == 0:
            self._context_bound = True
            return

        # --- Build global bounding box ---
        all_coords_flat = [
            coord
            for block_coords in all_vertex_coords
            for coord in block_coords
        ]
        bbox = BBox(all_coords_flat)

        # --- Build axes and equivalence classes ---
        aec = AxisEquivalenceClasses()
        for block_id, (coords, names) in enumerate(
            zip(all_vertex_coords, all_vertex_names, strict=False)
        ):
            try:
                aec.add_block_axes(build_block_axes(block_id, coords, names))
            except TopologyError as exc:
                raise TopologyError(f"block_id={block_id}: {exc}") from exc
        aec.build()

        # --- Pre-lock pass ---
        # class_root → locked cell count
        locked: dict[int, int] = {}
        # For each local axis of each pre-existing block, lock its class.
        # All conflicts in this pass are between pre-existing blocks; the
        # cell_conflict policy is not applied here — existing data is always
        # preserved as-is (first value wins, warning is emitted).
        if existing_cells:
            for block_id, (ni, nj, nk) in existing_cells.items():
                # Each block contributes 3 axes; axis global index = block_id*3 + axis_idx
                for axis_local_idx, n in enumerate((ni, nj, nk)):
                    global_idx = block_id * 3 + axis_local_idx
                    root = aec.class_of(global_idx)
                    if root in locked:
                        if locked[root] != n:
                            self._handle_pre_lock_conflict(root, locked[root], n)
                            # Preserve first-seen value; do not overwrite.
                    else:
                        locked[root] = n

        # --- Fallback pass ---
        # For each class, find the representative axis (max edge_length)
        classes = aec.all_classes()
        class_counts: dict[int, int] = {}

        for root, member_indices in classes.items():
            if root in locked:
                class_counts[root] = locked[root]
                continue

            # Find axis with maximum edge_length in this class
            best_axis = None
            best_length = -1.0
            for idx in member_indices:
                ax = aec.get_axis(idx)
                if ax.edge_length > best_length:
                    best_length = ax.edge_length
                    best_axis = ax

            if best_axis is None or self._fractions is None:
                class_counts[root] = 1
                continue

            dom = best_axis.dominant_global_axis
            target_length = self._fractions[dom] * bbox.length[dom]

            if target_length <= 0.0:
                # EC6: degenerate bbox or zero fraction → fallback n=1
                class_counts[root] = 1
            else:
                n = max(1, round(best_length / target_length))
                class_counts[root] = n

        self._class_counts = class_counts

        # --- Write back to blocks ---
        for block_id in range(n_blocks):
            counts_per_axis: list[int] = []
            for axis_local_idx in range(3):
                global_idx = block_id * 3 + axis_local_idx
                root = aec.class_of(global_idx)
                counts_per_axis.append(class_counts[root])
            self._block_counts[block_id] = (
                counts_per_axis[0],
                counts_per_axis[1],
                counts_per_axis[2],
            )

        self._context_bound = True

    def _handle_pre_lock_conflict(self, root: int, existing_n: int, new_n: int) -> None:
        msg = (
            f"Cell-count conflict in equivalence class (root={root}): "
            f"existing={existing_n}, incoming={new_n}."
        )
        logger.warning(msg)
        # No raise, no overwrite — preserve existing (first-seen) value.

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def counts_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> tuple[int, int, int]:
        """Return ``(ni, nj, nk)`` for the given block.

        Falls back to ``(1, 1, 1)`` if ``bind_context`` was not called
        (e.g. when *fractions* is ``None`` and no context is needed).
        """
        if not self._context_bound or block_id not in self._block_counts:
            return (1, 1, 1)
        return self._block_counts[block_id]

    def grading_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> str:
        """Return ``"simpleGrading (1 1 1)"`` (uniform grading)."""
        return "simpleGrading (1 1 1)"


# ---------------------------------------------------------------------------
# EuclideanProjectedCellCountStrategy (new default)
# ---------------------------------------------------------------------------


@dataclass
class _UnifiedCellCount:
    """Internal: resolved cell count for one axis equivalence class.

    - ``raw_count``: from the L2 projection (lower-bounded by 1).
    - ``raise_min_to``: optional global floor (``--minCellCount``).
    - ``lock_to``: hard override from ``--blockCount`` (first wins).
    """

    raw_count: int
    raise_min_to: int | None = None
    lock_to: int | None = None

    def resolved(self) -> int:
        if self.lock_to is not None:
            return self.lock_to
        n = self.raw_count
        if self.raise_min_to is not None and self.raise_min_to > n:
            n = self.raise_min_to
        return n


def compute_cell_counts_new(
    all_vertex_coords: list[list[tuple[float, float, float]]],
    all_vertex_names: list[list[str]],
    block_names: list[str],
    density: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_cell_count: int | None = None,
    block_overrides: dict[str, tuple[int, int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    """Compute per-block cell counts via L2 projection + EC propagation.

    Algorithm (six phases):

    1. Build :class:`AxisEquivalenceClasses` for all blocks.
    2. Compute the longest of the four parallel edges per (block, axis).
    3. For each EC class, evaluate
       ``ceil(sqrt((ax*L*dx)^2 + (ay*L*dy)^2 + (az*L*dz)^2))`` per member
       axis (with ``a = (ax, ay, az) = density`` and ``L`` the longest edge
       of that axis in the owning block), take the maximum, and lower-bound
       by 1.
    4. Apply the optional global floor *min_cell_count* to each non-locked
       class.
    5. Apply per-block overrides (lock_to). The first override wins per EC
       class; subsequent overrides for already-locked classes are logged
       and ignored. Override values are NOT clamped to *min_cell_count*
       (an info log is emitted if they fall below).
    6. Return a mapping block_id -> (n0, n1, n2).

    Parameters
    ----------
    all_vertex_coords:
        One list of 8 coordinates per block (block order).
    all_vertex_names:
        One list of 8 vertex names per block (same order as coords).
    block_names:
        Pre-resolved block name per block (typically ``candidate.label`` or
        a default ``blockN``).
    density:
        ``(ax, ay, az)`` -- cells per length unit per global axis. Default
        ``(1, 1, 1)``.
    min_cell_count:
        Optional global floor (>= 1). ``None`` disables the floor.
    block_overrides:
        Mapping ``name -> (n0, n1, n2)`` of explicit per-block overrides.
        Iteration order = CLI insertion order. Names not present among
        *block_names* are skipped with an info log.

    Returns
    -------
    List of ``(n0, n1, n2)`` per block in the same order as the input lists.
    """
    n_blocks = len(all_vertex_coords)
    if n_blocks == 0:
        return []
    if len(all_vertex_names) != n_blocks or len(block_names) != n_blocks:
        raise ValueError(
            "all_vertex_coords, all_vertex_names, and block_names must have "
            "the same length."
        )

    overrides = block_overrides or {}

    # --- Phase 1: build AEC ---
    aec = AxisEquivalenceClasses()
    for block_id, (coords, names) in enumerate(
        zip(all_vertex_coords, all_vertex_names, strict=False)
    ):
        try:
            aec.add_block_axes(build_block_axes(block_id, coords, names))
        except TopologyError as exc:
            raise TopologyError(f"block_id={block_id}: {exc}") from exc
    aec.build()

    # --- Phase 2: longest edge per (block, axis) ---
    longest_per_block: list[tuple[float, float, float]] = [
        compute_longest_edges_per_axis(coords) for coords in all_vertex_coords
    ]

    # --- Phase 3: raw counts via L2 projection, max-reduced per EC class ---
    ax, ay, az = density
    raw_per_class: dict[int, int] = {}

    for root, members in aec.all_classes().items():
        class_raw = 1  # safe lower bound
        for global_idx in members:
            axis = aec.get_axis(global_idx)
            dxu, dyu, dzu = axis.direction
            block_id = axis.block_id
            axis_local = axis.axis_index
            L = longest_per_block[block_id][axis_local]

            cx = ax * L * dxu
            cy = ay * L * dyu
            cz = az * L * dzu
            value_f = math.sqrt(cx * cx + cy * cy + cz * cz)
            value_i = max(1, math.ceil(value_f))
            if value_i > class_raw:
                class_raw = value_i
        raw_per_class[root] = class_raw

    # --- Phase 4: build UnifiedCellCount per class, apply floor ---
    ucc_per_class: dict[int, _UnifiedCellCount] = {}
    for root, raw in raw_per_class.items():
        ucc_per_class[root] = _UnifiedCellCount(
            raw_count=raw,
            raise_min_to=min_cell_count,
            lock_to=None,
        )

    # --- Phase 5: per-block overrides (first wins) ---
    name_to_block_id: dict[str, int] = {}
    for idx, name in enumerate(block_names):
        if name not in name_to_block_id:
            name_to_block_id[name] = idx  # first occurrence wins

    for override_name, override_values in overrides.items():
        if override_name not in name_to_block_id:
            logger.info(
                "Block-count override for '%s' not found among loaded "
                "blocks; skipping.",
                override_name,
            )
            continue
        block_id = name_to_block_id[override_name]
        n0, n1, n2 = override_values
        for axis_local, n in enumerate((n0, n1, n2)):
            global_idx = block_id * 3 + axis_local
            root = aec.class_of(global_idx)
            ucc = ucc_per_class[root]
            if ucc.lock_to is not None:
                logger.info(
                    "Block '%s' axis %d shares equivalence class with an "
                    "already-overridden axis; skipping (first override wins).",
                    override_name,
                    axis_local,
                )
                continue
            if min_cell_count is not None and n < min_cell_count:
                logger.info(
                    "Block override for '%s' axis %d sets cell count %d "
                    "below --minCellCount %d; using override value as "
                    "requested.",
                    override_name,
                    axis_local,
                    n,
                    min_cell_count,
                )
            ucc.lock_to = n

    # --- Phase 6: per-block mapping ---
    result: list[tuple[int, int, int]] = []
    for block_id in range(n_blocks):
        triple: list[int] = []
        for axis_local in range(3):
            global_idx = block_id * 3 + axis_local
            root = aec.class_of(global_idx)
            triple.append(ucc_per_class[root].resolved())
        result.append((triple[0], triple[1], triple[2]))
    return result


class EuclideanProjectedCellCountStrategy(CellCountStrategy):
    """Default cell-count strategy using L2 density projection.

    See :func:`compute_cell_counts_new` for the algorithm. This class is a
    thin stateful wrapper that fulfils the :class:`CellCountStrategy`
    interface so it can be plugged into the STEP -> blockMeshDict pipeline.

    Notes
    -----
    The :meth:`bind_context` signature differs from
    :meth:`PropagatedCellCountStrategy.bind_context`: it requires
    *block_names* (one per block, matching the order of *all_vertex_coords*)
    and does NOT accept *existing_cells* -- this strategy is intended for
    fresh creation only, not append mode.
    """

    def __init__(
        self,
        density: tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_cell_count: int | None = None,
        block_overrides: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self._density = density
        self._min_cell_count = min_cell_count
        self._block_overrides: dict[str, tuple[int, int, int]] = (
            dict(block_overrides) if block_overrides else {}
        )
        self._block_counts: dict[int, tuple[int, int, int]] = {}
        self._context_bound: bool = False

    def bind_context(
        self,
        all_vertex_coords: list[list[tuple[float, float, float]]],
        all_vertex_names: list[list[str]],
        block_names: list[str],
    ) -> None:
        """Bind context and compute all cell counts up-front."""
        if not all_vertex_coords:
            self._context_bound = True
            return
        counts = compute_cell_counts_new(
            all_vertex_coords=all_vertex_coords,
            all_vertex_names=all_vertex_names,
            block_names=block_names,
            density=self._density,
            min_cell_count=self._min_cell_count,
            block_overrides=self._block_overrides,
        )
        for block_id, triple in enumerate(counts):
            self._block_counts[block_id] = triple
        self._context_bound = True

    def counts_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> tuple[int, int, int]:
        if not self._context_bound or block_id not in self._block_counts:
            fallback = self._min_cell_count or 1
            return (fallback, fallback, fallback)
        return self._block_counts[block_id]

    def grading_for_block(
        self,
        block_id: int,
        block_vertex_coords: Any,
        edge_curves: Any,
    ) -> str:
        return "simpleGrading (1 1 1)"
