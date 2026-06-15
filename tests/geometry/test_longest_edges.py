"""Unit tests for meshing_utils.geometry.hex_axes.compute_longest_edges_per_axis."""

from __future__ import annotations

import pytest

from meshing_utils.geometry.hex_axes import (
    build_block_axes,
    compute_longest_edges_per_axis,
)


def _unit_cube_coords() -> list[tuple[float, float, float]]:
    """8 vertices of a unit cube in OpenFOAM order."""
    return [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]


def _anisotropic_block(dx: float, dy: float, dz: float) -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (dx,  0.0, 0.0),
        (dx,  dy,  0.0),
        (0.0, dy,  0.0),
        (0.0, 0.0, dz),
        (dx,  0.0, dz),
        (dx,  dy,  dz),
        (0.0, dy,  dz),
    ]


def test_unit_cube_all_axes_equal_one() -> None:
    """A unit cube has all longest edges = 1.0 per axis."""
    coords = _unit_cube_coords()
    longest = compute_longest_edges_per_axis(coords)
    assert longest[0] == pytest.approx(1.0)
    assert longest[1] == pytest.approx(1.0)
    assert longest[2] == pytest.approx(1.0)


def test_anisotropic_block() -> None:
    """An axis-aligned block with distinct extents returns those extents."""
    coords = _anisotropic_block(2.0, 3.0, 5.0)
    longest = compute_longest_edges_per_axis(coords)
    assert sorted(longest) == pytest.approx(sorted([2.0, 3.0, 5.0]))


def test_axis_indexing_matches_build_block_axes() -> None:
    """The local axis indexing must match build_block_axes (Variant 2)."""
    coords = _anisotropic_block(2.0, 3.0, 5.0)
    longest = compute_longest_edges_per_axis(coords)

    names = [f"v{i}" for i in range(8)]
    axes = build_block_axes(0, coords, names)

    # In a regular axis-aligned block, the V0-incident edge length equals the
    # longest parallel edge length for that axis class.
    for axis in axes:
        assert longest[axis.axis_index] == pytest.approx(axis.edge_length)


def test_tapered_block_returns_maximum_of_four_edges() -> None:
    """For a tapered block the helper returns the maximum of the four parallel edges."""
    # A block tapered along x: x-edges at z=0 are length 1, at z=1 are length 2.
    # The y- and z-axes remain uniform with lengths 1.0.
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (2.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    longest = compute_longest_edges_per_axis(coords)

    # Sorted assertion: max x-edge=2.0, y-edges=1.0, z-edges in [1.0, 2.236]
    # (the diagonal taper). The two "x"-edges at the top are length 2.0,
    # bottom 1.0 -> x-axis longest = 2.0. y-axis remains 1.0. z-axis: edge
    # between (1,0,0) and (2,0,1) has length sqrt(2), etc.
    # We assert max of axis containing x-edges is 2.0.
    assert max(longest) == pytest.approx(2.0)


def test_wrong_vertex_count_raises() -> None:
    coords = [(0.0, 0.0, 0.0)] * 7
    with pytest.raises(ValueError):
        compute_longest_edges_per_axis(coords)


def test_returns_tuple_of_floats() -> None:
    coords = _unit_cube_coords()
    result = compute_longest_edges_per_axis(coords)
    assert isinstance(result, tuple)
    assert len(result) == 3
    for v in result:
        assert isinstance(v, float)
