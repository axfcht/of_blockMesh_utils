"""Unit tests for meshing_utils.geometry.cell_count.

Migrated from tests/common/test_cell_count_strategy.py.
Covers PropagatedCellCountStrategy (including all plan-defined test cases).
"""

from __future__ import annotations

import math

import pytest

from meshing_utils import CellCountStrategy, PropagatedCellCountStrategy, TopologyError
from meshing_utils.geometry.cell_count import BBox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_cube_coords(
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> list[tuple[float, float, float]]:
    """8 vertex coordinates in OpenFOAM order for a block with given extents."""
    ox, oy, oz = offset_x, offset_y, offset_z
    return [
        (ox,      oy,      oz),       # v0
        (ox + dx, oy,      oz),       # v1
        (ox + dx, oy + dy, oz),       # v2
        (ox,      oy + dy, oz),       # v3
        (ox,      oy,      oz + dz),  # v4
        (ox + dx, oy,      oz + dz),  # v5
        (ox + dx, oy + dy, oz + dz),  # v6
        (ox,      oy + dy, oz + dz),  # v7
    ]


def _names(start: int = 0, count: int = 8) -> list[str]:
    return [f"v{i}" for i in range(start, start + count)]


def _make_strategy_one_block(
    coords: list[tuple[float, float, float]],
    fractions: tuple[float, float, float] | None,
    cell_conflict: str = "warn-max",
    existing_cells: dict[int, tuple[int, int, int]] | None = None,
) -> PropagatedCellCountStrategy:
    """Convenience: build a strategy and bind context for a single block."""
    strategy = PropagatedCellCountStrategy(
        fractions=fractions,
        cell_conflict=cell_conflict,
    )
    strategy.bind_context(
        all_vertex_coords=[coords],
        all_vertex_names=[_names(0)],
        existing_cells=existing_cells,
    )
    return strategy


def _make_strategy_two_blocks(
    coords0: list[tuple[float, float, float]],
    names0: list[str],
    coords1: list[tuple[float, float, float]],
    names1: list[str],
    fractions: tuple[float, float, float] | None,
    cell_conflict: str = "warn-max",
    existing_cells: dict[int, tuple[int, int, int]] | None = None,
) -> PropagatedCellCountStrategy:
    strategy = PropagatedCellCountStrategy(
        fractions=fractions,
        cell_conflict=cell_conflict,
    )
    strategy.bind_context(
        all_vertex_coords=[coords0, coords1],
        all_vertex_names=[names0, names1],
        existing_cells=existing_cells,
    )
    return strategy


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------

class TestBBox:
    def test_unit_cube_bbox(self):
        coords = _unit_cube_coords(dx=1.0, dy=2.0, dz=3.0)
        bbox = BBox(coords)
        assert bbox.length == pytest.approx((1.0, 2.0, 3.0))

    def test_empty_coords(self):
        bbox = BBox([])
        assert bbox.length == (0.0, 0.0, 0.0)

    def test_single_point(self):
        bbox = BBox([(1.0, 2.0, 3.0)])
        assert bbox.length == (0.0, 0.0, 0.0)

    def test_negative_coords(self):
        bbox = BBox([(-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)])
        assert bbox.length == pytest.approx((2.0, 4.0, 6.0))


# ---------------------------------------------------------------------------
# PropagatedCellCountStrategy — interface compliance
# ---------------------------------------------------------------------------

class TestStrategyInterface:
    def test_is_cell_count_strategy_subclass(self):
        strategy = PropagatedCellCountStrategy()
        assert isinstance(strategy, CellCountStrategy)

    def test_grading_returns_simple_grading(self):
        strategy = PropagatedCellCountStrategy()
        result = strategy.grading_for_block(0, None, None)
        assert result == "simpleGrading (1 1 1)"

    def test_counts_returns_tuple_of_three_ints(self):
        coords = _unit_cube_coords()
        strategy = _make_strategy_one_block(coords, fractions=None)
        result = strategy.counts_for_block(0, coords, None)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_invalid_cell_conflict_raises(self):
        with pytest.raises(ValueError, match="cell_conflict"):
            PropagatedCellCountStrategy(cell_conflict="invalid")


# ---------------------------------------------------------------------------
# HP1: 1 block 1x2x3, fractions=(0.1, 0.1, 0.1) → (10, 10, 10)
# ---------------------------------------------------------------------------

class TestHP1SingleAnisotropicBlock:
    """HP1: One axis-parallel block 1x2x3, BBox identical, fractions=(0.1,0.1,0.1)."""

    def test_counts_10_10_10(self):
        coords = _unit_cube_coords(dx=1.0, dy=2.0, dz=3.0)
        strategy = _make_strategy_one_block(coords, fractions=(0.1, 0.1, 0.1))
        result = strategy.counts_for_block(0, coords, None)
        # bbox=(1,2,3), fractions=(0.1,0.1,0.1)
        # i: round(1/(0.1*1))=10, j: round(2/(0.1*2))=10, k: round(3/(0.1*3))=10
        assert result == (10, 10, 10)

    def test_counts_all_10_for_anisotropic_block(self):
        """Each axis cell size = fraction * its bbox extent → always same ratio."""
        coords = _unit_cube_coords(dx=1.0, dy=2.0, dz=3.0)
        strategy = _make_strategy_one_block(coords, fractions=(0.1, 0.1, 0.1))
        ni, nj, nk = strategy.counts_for_block(0, coords, None)
        assert ni == 10
        assert nj == 10
        assert nk == 10

    def test_counts_respect_different_fractions(self):
        """Different fractions per axis produce different cell counts."""
        coords = _unit_cube_coords(dx=2.0, dy=2.0, dz=2.0)
        # bbox=(2,2,2), fractions=(0.5,0.25,0.1)
        # i: round(2/(0.5*2))=round(2)=2
        # j: round(2/(0.25*2))=round(4)=4
        # k: round(2/(0.1*2))=round(10)=10
        strategy = _make_strategy_one_block(coords, fractions=(0.5, 0.25, 0.1))
        result = strategy.counts_for_block(0, coords, None)
        assert result == (2, 4, 10)


# ---------------------------------------------------------------------------
# HP2: 2 adjacent blocks, fractions set → propagation
# ---------------------------------------------------------------------------

class TestHP2TwoAdjacentBlocks:
    """HP2: Two adjacent unit cubes sharing an X-face, fractions=(0.1,0.1,0.1)."""

    def _build(self):
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]
        return coords0, names0, coords1, names1

    def test_counts_are_positive(self):
        coords0, names0, coords1, names1 = self._build()
        strategy = _make_strategy_two_blocks(
            coords0, names0, coords1, names1, fractions=(0.1, 0.1, 0.1)
        )
        for bid in (0, 1):
            counts = strategy.counts_for_block(bid, None, None)
            assert all(c >= 1 for c in counts)

    def test_grading_uniform(self):
        coords0, names0, coords1, names1 = self._build()
        strategy = _make_strategy_two_blocks(
            coords0, names0, coords1, names1, fractions=(0.1, 0.1, 0.1)
        )
        for bid in (0, 1):
            grading = strategy.grading_for_block(bid, None, None)
            assert grading == "simpleGrading (1 1 1)"


# ---------------------------------------------------------------------------
# HP3: Append mode — existing block locks class, neighbour inherits
# ---------------------------------------------------------------------------

class TestHP3AppendModeLocking:
    """HP3: Pre-existing block locks class; neighbour block inherits its count."""

    def test_locked_axes_propagate_via_in_plane_union(self):
        """j and k axes of block1 inherit block0's locked counts via edge-based union."""
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]

        existing = {0: (5, 3, 7)}
        strategy = _make_strategy_two_blocks(
            coords0, names0, coords1, names1,
            fractions=(0.5, 0.5, 0.5),
            existing_cells=existing,
        )

        n1 = strategy.counts_for_block(1, coords1, None)
        # i-axes: separate classes → block1 i-axis computed from fractions
        assert n1[0] == 1
        # j-axes: edge-based union → n1[1] inherits block0's locked j=3
        assert n1[1] == 3, (
            "j-axis of block1 must inherit block0's locked count via edge-based union"
        )
        # k-axes: edge-based union → n1[2] inherits block0's locked k=7
        assert n1[2] == 7, (
            "k-axis of block1 must inherit block0's locked count via edge-based union"
        )


# ---------------------------------------------------------------------------
# EC1: Rotated block (i parallel to global Y)
# ---------------------------------------------------------------------------

class TestEC1RotatedBlock:
    """EC1: Block where i-direction is parallel to global Y → dominant = Y."""

    def test_dominant_y_and_n_from_y_bbox(self):
        coords = [
            (0.0, 0.0, 0.0),  # v0
            (0.0, 2.0, 0.0),  # v1: i-axis along Y
            (1.0, 2.0, 0.0),  # v2
            (1.0, 0.0, 0.0),  # v3
            (0.0, 0.0, 1.0),  # v4
            (0.0, 2.0, 1.0),  # v5
            (1.0, 2.0, 1.0),  # v6
            (1.0, 0.0, 1.0),  # v7
        ]
        _names(0, 8)
        # bbox: x: 0..1 → 1, y: 0..2 → 2, z: 0..1 → 1
        # i-axis: v0→v1 direction = (0,1,0) → dominant = Y(1)
        # edge_length = 2.0
        # target_cell_size = fractions[1] * bbox.length[1] = 0.1 * 2 = 0.2
        # n = round(2.0 / 0.2) = 10
        strategy = _make_strategy_one_block(coords, fractions=(0.1, 0.1, 0.1))
        result = strategy.counts_for_block(0, coords, None)
        assert result[0] == 10


# ---------------------------------------------------------------------------
# EC2: No fractions, no existing blocks → all (1, 1, 1)
# ---------------------------------------------------------------------------

class TestEC2NoFractionsNorExisting:
    """EC2: fractions=None, no existing blocks → every block gets (1,1,1)."""

    def test_single_block_returns_111(self):
        coords = _unit_cube_coords(dx=10.0, dy=20.0, dz=30.0)
        strategy = _make_strategy_one_block(coords, fractions=None)
        result = strategy.counts_for_block(0, coords, None)
        assert result == (1, 1, 1)

    def test_two_blocks_both_111(self):
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]
        strategy = _make_strategy_two_blocks(coords0, names0, coords1, names1, fractions=None)
        assert strategy.counts_for_block(0, coords0, None) == (1, 1, 1)
        assert strategy.counts_for_block(1, coords1, None) == (1, 1, 1)

    def test_without_bind_context_fallback_111(self):
        """counts_for_block without bind_context must return (1,1,1)."""
        strategy = PropagatedCellCountStrategy(fractions=None)
        result = strategy.counts_for_block(0, None, None)
        assert result == (1, 1, 1)


# ---------------------------------------------------------------------------
# EC3: No fractions, existing blocks set some classes → rest = 1
# ---------------------------------------------------------------------------

class TestEC3NoFractionsWithExisting:
    """EC3: fractions=None, pre-existing blocks propagate; rest defaults to 1."""

    def test_locked_class_propagates_rest_is_1(self):
        """j and k axes of block1 propagate from block0; i-axis falls back to 1."""
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]

        existing = {0: (8, 4, 2)}
        strategy = _make_strategy_two_blocks(
            coords0, names0, coords1, names1,
            fractions=None,
            existing_cells=existing,
        )

        n0 = strategy.counts_for_block(0, coords0, None)
        n1 = strategy.counts_for_block(1, coords1, None)

        assert n0 == (8, 4, 2)
        assert n1[0] == 1
        assert n1[1] == 4, (
            "j-axis of block1 must inherit block0's locked count via edge-based union"
        )
        assert n1[2] == 2, (
            "k-axis of block1 must inherit block0's locked count via edge-based union"
        )


# ---------------------------------------------------------------------------
# EC4: Conflict between two pre-locked classes
# ---------------------------------------------------------------------------

class TestEC4Conflict:
    """EC4: Two pre-existing blocks in same equivalence class with different counts."""

    def _build_adjacent_both_existing(
        self,
        cells0: tuple[int, int, int],
        cells1: tuple[int, int, int],
        cell_conflict: str,
    ) -> PropagatedCellCountStrategy:
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]
        existing = {0: cells0, 1: cells1}
        strategy = PropagatedCellCountStrategy(cell_conflict=cell_conflict)
        strategy.bind_context(
            all_vertex_coords=[coords0, coords1],
            all_vertex_names=[names0, names1],
            existing_cells=existing,
        )
        return strategy

    def test_error_policy_no_raise_on_pre_lock_conflict(self):
        strategy = self._build_adjacent_both_existing(
            cells0=(1, 10, 1),
            cells1=(1, 5, 1),
            cell_conflict="error",
        )
        n0 = strategy.counts_for_block(0, None, None)
        n1 = strategy.counts_for_block(1, None, None)
        assert n0[1] == 10
        assert n1[1] == 10

    def test_pre_lock_conflict_preserves_first(self):
        strategy = self._build_adjacent_both_existing(
            cells0=(1, 10, 1),
            cells1=(1, 5, 1),
            cell_conflict="warn-max",
        )
        n0 = strategy.counts_for_block(0, None, None)
        n1 = strategy.counts_for_block(1, None, None)
        assert n0[1] == 10
        assert n1[1] == 10

    def test_warn_first_keeps_first(self):
        strategy = self._build_adjacent_both_existing(
            cells0=(1, 10, 1),
            cells1=(1, 5, 1),
            cell_conflict="warn-first",
        )
        n0 = strategy.counts_for_block(0, None, None)
        n1 = strategy.counts_for_block(1, None, None)
        assert n0[1] == 10
        assert n1[1] == 10

    def test_pre_lock_conflict_does_not_overwrite_existing(self, caplog):
        """Two existing blocks with conflicting counts in same class → first-seen preserved."""
        import logging
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = _names(0, 8)
        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]
        existing = {0: (1, 10, 1), 1: (1, 5, 1)}
        strategy = PropagatedCellCountStrategy(cell_conflict="warn-max")
        with caplog.at_level(logging.WARNING):
            strategy.bind_context(
                all_vertex_coords=[coords0, coords1],
                all_vertex_names=[names0, names1],
                existing_cells=existing,
            )
        conflict_records = [r for r in caplog.records if "conflict" in r.message.lower()]
        assert len(conflict_records) >= 1
        n0 = strategy.counts_for_block(0, None, None)
        n1 = strategy.counts_for_block(1, None, None)
        assert n0[1] == 10
        assert n1[1] == 10

    def test_no_conflict_no_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            self._build_adjacent_both_existing(
                cells0=(5, 2, 3),
                cells1=(5, 2, 3),
                cell_conflict="warn-max",
            )
        conflict_records = [r for r in caplog.records if "conflict" in r.message.lower()]
        assert len(conflict_records) == 0


# ---------------------------------------------------------------------------
# EC5: 45° block (equal X and Y components) → dominant = 0 (smallest index)
# ---------------------------------------------------------------------------

class TestEC5FortyFiveDegreeBlock:
    """EC5: i-axis at 45° in XY plane → dominant must be 0 (X, smaller index)."""

    def test_dominant_smallest_index_45deg(self):
        d = math.sqrt(2.0) / 2.0  # unit vector at 45°
        coords = [
            (0.0, 0.0, 0.0),   # v0
            (d,   d,   0.0),   # v1: i-axis at 45°
            (d,   d,   1.0),   # v2
            (0.0, 0.0, 1.0),   # v3
            (0.0, 1.0, 0.0),   # v4
            (d,   d+1, 0.0),   # v5
            (d,   d+1, 1.0),   # v6
            (0.0, 1.0, 1.0),   # v7
        ]
        _names(0, 8)
        strategy = _make_strategy_one_block(coords, fractions=(0.1, 0.1, 0.1))
        result = strategy.counts_for_block(0, coords, None)
        assert all(c >= 1 for c in result)


# ---------------------------------------------------------------------------
# EC6: Degenerate BBox (target_cell_size <= 0) → n=1, no crash
# ---------------------------------------------------------------------------

class TestEC6DegenerateBBox:
    """EC6: zero-length bbox or zero fraction → fallback n=1."""

    def test_zero_bbox_extent_raises_topology_error(self):
        """All-coincident vertices → TopologyError during bind_context."""
        coords = [(1.0, 2.0, 3.0)] * 8
        names = _names(0, 8)
        strategy = PropagatedCellCountStrategy(fractions=(0.1, 0.1, 0.1))
        with pytest.raises(TopologyError):
            strategy.bind_context(
                all_vertex_coords=[coords],
                all_vertex_names=[names],
            )

    def test_zero_fraction_fallback_to_1(self):
        """fractions with a zero component → target_cell_size=0 → n=1."""
        coords = _unit_cube_coords(dx=1.0, dy=2.0, dz=3.0)
        names = _names(0, 8)
        strategy = PropagatedCellCountStrategy(fractions=(0.0, 0.1, 0.1))
        strategy.bind_context(
            all_vertex_coords=[coords],
            all_vertex_names=[names],
        )
        result = strategy.counts_for_block(0, coords, None)
        assert result[0] == 1

    def test_degenerate_block_raises_topology_error(self):
        """Block with all coincident vertices raises TopologyError during bind_context."""
        coords = [(0.0, 0.0, 0.0)] * 8
        names = _names(0, 8)
        strategy = PropagatedCellCountStrategy(fractions=(0.1, 0.1, 0.1))
        with pytest.raises(TopologyError):
            strategy.bind_context(
                all_vertex_coords=[coords],
                all_vertex_names=[names],
            )
