"""Unit tests for compute_cell_counts_new and EuclideanProjectedCellCountStrategy."""

from __future__ import annotations

import logging
import math

import pytest

from meshing_utils.geometry.cell_count import (
    EuclideanProjectedCellCountStrategy,
    compute_cell_counts_new,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(dx: float, dy: float, dz: float, ox: float = 0.0,
           oy: float = 0.0, oz: float = 0.0) -> list[tuple[float, float, float]]:
    return [
        (ox,      oy,      oz),
        (ox + dx, oy,      oz),
        (ox + dx, oy + dy, oz),
        (ox,      oy + dy, oz),
        (ox,      oy,      oz + dz),
        (ox + dx, oy,      oz + dz),
        (ox + dx, oy + dy, oz + dz),
        (ox,      oy + dy, oz + dz),
    ]


def _names(prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(8)]


def _face_shared_pair() -> tuple[
    list[list[tuple[float, float, float]]],
    list[list[str]],
    list[str],
]:
    """Two cubes sharing the x=1 face. Block A vertices share names with B."""
    a_coords = _block(1.0, 1.0, 1.0)
    b_coords = _block(1.0, 1.0, 1.0, ox=1.0)

    # Build vertex names so the shared face is identified by shared names.
    # Block A: a0..a7 (canonical OpenFOAM order). The x=1 face is v1,v2,v6,v5.
    # Block B: same face is its v0,v3,v7,v4 (left face).
    a_names = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
    # Map B's v0<-a1, v3<-a2, v7<-a6, v4<-a5.
    b_names = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

    return [a_coords, b_coords], [a_names, b_names], ["blockA", "blockB"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    assert compute_cell_counts_new([], [], []) == []


def test_single_cube_default_density() -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    result = compute_cell_counts_new(coords, names, ["block0"])
    # default density (1,1,1), L=1 -> ceil(sqrt(1^2)) = 1 per axis
    assert result == [(1, 1, 1)]


def test_single_cube_density_10() -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    result = compute_cell_counts_new(
        coords, names, ["block0"], density=(10.0, 10.0, 10.0)
    )
    assert result == [(10, 10, 10)]


def test_anisotropic_density() -> None:
    """Different density per axis -> different counts."""
    coords = [_block(2.0, 3.0, 5.0)]
    names = [_names("v")]
    result = compute_cell_counts_new(
        coords, names, ["block0"], density=(1.0, 1.0, 1.0)
    )
    # axis-aligned: each axis -> ceil(L * 1) = L
    assert sorted(result[0]) == [2, 3, 5]


def test_zero_density_falls_back_to_one() -> None:
    coords = [_block(5.0, 5.0, 5.0)]
    names = [_names("v")]
    result = compute_cell_counts_new(
        coords, names, ["block0"], density=(0.0, 0.0, 0.0)
    )
    assert result == [(1, 1, 1)]


def test_zero_density_one_axis_only() -> None:
    coords = [_block(2.0, 3.0, 5.0)]
    names = [_names("v")]
    # density: only y active
    result = compute_cell_counts_new(
        coords, names, ["block0"], density=(0.0, 4.0, 0.0)
    )
    # y axis: ceil(3 * 4) = 12; others fall back to 1
    triple = result[0]
    assert max(triple) == 12
    assert sorted(triple) == [1, 1, 12]


def test_min_cell_count_floor() -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    result = compute_cell_counts_new(
        coords, names, ["block0"],
        density=(1.0, 1.0, 1.0),
        min_cell_count=5,
    )
    assert result == [(5, 5, 5)]


def test_override_below_min_wins(caplog: pytest.LogCaptureFixture) -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    caplog.set_level(logging.INFO, logger="meshing_utils.geometry.cell_count")
    result = compute_cell_counts_new(
        coords, names, ["block0"],
        density=(10.0, 10.0, 10.0),
        min_cell_count=10,
        block_overrides={"block0": (2, 2, 2)},
    )
    assert result == [(2, 2, 2)]
    # info log about under-min override
    assert any("below --minCellCount" in r.message for r in caplog.records)


def test_override_unknown_block_skipped(caplog: pytest.LogCaptureFixture) -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    caplog.set_level(logging.INFO, logger="meshing_utils.geometry.cell_count")
    result = compute_cell_counts_new(
        coords, names, ["block0"],
        density=(3.0, 3.0, 3.0),
        block_overrides={"missing": (7, 7, 7)},
    )
    assert result == [(3, 3, 3)]
    assert any("not found among loaded blocks" in r.message for r in caplog.records)


def test_face_shared_blocks_propagation() -> None:
    """Two cubes sharing a face: cell count along shared axis is max-reduced."""
    coords_list, names_list, bnames = _face_shared_pair()
    result = compute_cell_counts_new(
        coords_list, names_list, bnames, density=(2.0, 2.0, 2.0)
    )
    # Both cubes have L=1 along each axis -> ceil(2*1) = 2 everywhere.
    assert result[0] == (2, 2, 2)
    assert result[1] == (2, 2, 2)


def test_face_shared_blocks_override_propagates() -> None:
    """Override on blockA along shared face axis locks the EC class for blockB."""
    coords_list, names_list, bnames = _face_shared_pair()
    result = compute_cell_counts_new(
        coords_list, names_list, bnames,
        density=(1.0, 1.0, 1.0),
        block_overrides={"blockA": (4, 5, 6)},
    )
    # blockA: (4,5,6). blockB shares the y- and z- axes with blockA via the
    # x=1 face (4 shared vertices). blockB's local axis order may differ
    # but the two propagated axes should equal 5 and 6 in some permutation.
    a = result[0]
    b = result[1]
    assert a == (4, 5, 6)
    # blockB has three counts: one from its own un-shared x-axis (=1), and
    # two from the EC propagation (= 5 and 6 in some order).
    sorted_b = sorted(b)
    assert sorted_b == [1, 5, 6]


def test_tapered_block_uses_longest_edge() -> None:
    """Tapered block: cell count is based on the longest of the four parallel edges."""
    # Tapered along x: bottom x-edges length 1, top x-edges length 2.
    coords = [[
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (2.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]]
    names = [_names("v")]
    result = compute_cell_counts_new(coords, names, ["b"], density=(3.0, 0.0, 0.0))
    # Some axis should contain ceil(3*2) = 6 (using the longest x-edge).
    assert max(result[0]) == 6


def test_diagonal_block_l2_projection() -> None:
    """A 45-degree-rotated block: L2 norm projection."""
    # Rotate unit square by 45 deg in xy. The block is 1x1x1 but the x-axis
    # internally points along (1/sqrt2, 1/sqrt2, 0).
    s = math.sqrt(2.0) / 2.0
    coords = [[
        (0.0, 0.0, 0.0),
        (s,   s,   0.0),
        (0.0, 2 * s, 0.0),
        (-s,  s,   0.0),
        (0.0, 0.0, 1.0),
        (s,   s,   1.0),
        (0.0, 2 * s, 1.0),
        (-s,  s,   1.0),
    ]]
    names = [_names("v")]
    # density (2, 2, 0): the diagonal axis has L=1, direction (s, s, 0)
    # raw = ceil(sqrt((2*1*s)^2 + (2*1*s)^2)) = ceil(sqrt(2)) = 2
    result = compute_cell_counts_new(coords, names, ["b"], density=(2.0, 2.0, 0.0))
    # The block has two horizontal axes in the rotated plane and one vertical.
    # Both rotated axes should yield 2; vertical falls back to 1.
    assert sorted(result[0]) == [1, 2, 2]


def test_first_override_wins_on_shared_class(caplog: pytest.LogCaptureFixture) -> None:
    """Override A wins; conflicting override on EC-shared axis of B is logged & skipped."""
    coords_list, names_list, bnames = _face_shared_pair()
    caplog.set_level(logging.INFO, logger="meshing_utils.geometry.cell_count")
    result = compute_cell_counts_new(
        coords_list, names_list, bnames,
        density=(1.0, 1.0, 1.0),
        block_overrides={"blockA": (1, 5, 5), "blockB": (9, 9, 9)},
    )
    # blockA fixed (1,5,5). The y- and z-EC classes are locked at 5.
    # blockB's override of 9 on those locked axes is ignored; only its
    # un-shared x-axis can be locked to 9.
    a = result[0]
    b = result[1]
    assert a == (1, 5, 5)
    # blockB still has 5 on the propagated axes, and 9 on its own x-axis.
    assert sorted(b) == [5, 5, 9]
    assert any("already-overridden axis" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Strategy wrapper
# ---------------------------------------------------------------------------


def test_strategy_counts_match_function() -> None:
    coords = [_block(1.0, 1.0, 1.0)]
    names = [_names("v")]
    strategy = EuclideanProjectedCellCountStrategy(density=(4.0, 4.0, 4.0))
    strategy.bind_context(coords, names, ["b"])
    assert strategy.counts_for_block(0, None, None) == (4, 4, 4)


def test_strategy_grading_constant() -> None:
    strategy = EuclideanProjectedCellCountStrategy()
    assert strategy.grading_for_block(0, None, None) == "simpleGrading (1 1 1)"


def test_strategy_before_bind_returns_fallback() -> None:
    strategy = EuclideanProjectedCellCountStrategy(min_cell_count=3)
    # No bind_context() called: fallback uses min_cell_count.
    assert strategy.counts_for_block(0, None, None) == (3, 3, 3)


def test_strategy_before_bind_default_one() -> None:
    strategy = EuclideanProjectedCellCountStrategy()
    assert strategy.counts_for_block(0, None, None) == (1, 1, 1)


def test_mismatched_input_lengths_raises() -> None:
    with pytest.raises(ValueError):
        compute_cell_counts_new(
            [_block(1, 1, 1)],
            [_names("v"), _names("w")],  # too many
            ["b"],
        )
