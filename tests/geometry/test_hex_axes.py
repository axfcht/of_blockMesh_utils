"""Unit tests for meshing_utils.geometry.hex_axes.

Merged from tests/common/test_block_axis.py and tests/common/test_hex_face_detection.py.
Covers BlockAxis construction, dominant_global_axis determination, the Union-Find based
AxisEquivalenceClasses merging logic, detect_hex_faces, compute_axis_class_per_edge, and
map_class_to_local_axis_index.
"""

from __future__ import annotations

import math

import pytest

from meshing_utils import AxisEquivalenceClasses, TopologyError
from meshing_utils.geometry.hex_axes import (
    _dominant_axis,
    _validate_hex_topology,
    build_block_axes,
    compute_axis_class_per_edge,
    detect_hex_faces,
    map_class_to_local_axis_index,
)

# ---------------------------------------------------------------------------
# Helpers (from test_block_axis.py)
# ---------------------------------------------------------------------------

def _unit_cube_coords(offset_x: float = 0.0) -> list[tuple[float, float, float]]:
    """8 vertices of a unit cube in OpenFOAM order, optionally shifted in X."""
    return [
        (offset_x + 0.0, 0.0, 0.0),  # v0
        (offset_x + 1.0, 0.0, 0.0),  # v1
        (offset_x + 1.0, 1.0, 0.0),  # v2
        (offset_x + 0.0, 1.0, 0.0),  # v3
        (offset_x + 0.0, 0.0, 1.0),  # v4
        (offset_x + 1.0, 0.0, 1.0),  # v5
        (offset_x + 1.0, 1.0, 1.0),  # v6
        (offset_x + 0.0, 1.0, 1.0),  # v7
    ]


def _unit_cube_names(prefix: str = "v", start: int = 0) -> list[str]:
    """8 vertex names for a unit cube."""
    return [f"{prefix}{i + start}" for i in range(8)]


def _anisotropic_block_coords(dx: float, dy: float, dz: float) -> list[tuple[float, float, float]]:
    """8 vertices of a block with given extents dx, dy, dz in OpenFOAM order."""
    return [
        (0.0, 0.0, 0.0),   # v0
        (dx,  0.0, 0.0),   # v1
        (dx,  dy,  0.0),   # v2
        (0.0, dy,  0.0),   # v3
        (0.0, 0.0, dz),    # v4
        (dx,  0.0, dz),    # v5
        (dx,  dy,  dz),    # v6
        (0.0, dy,  dz),    # v7
    ]


# ---------------------------------------------------------------------------
# Helpers (from test_hex_face_detection.py)
# ---------------------------------------------------------------------------

def _canonical_unit_cube() -> list[tuple[float, float, float]]:
    """8 vertices of a unit cube in canonical OpenFOAM V0..V7 order."""
    return [
        (0.0, 0.0, 0.0),  # v0
        (1.0, 0.0, 0.0),  # v1
        (1.0, 1.0, 0.0),  # v2
        (0.0, 1.0, 0.0),  # v3
        (0.0, 0.0, 1.0),  # v4
        (1.0, 0.0, 1.0),  # v5
        (1.0, 1.0, 1.0),  # v6
        (0.0, 1.0, 1.0),  # v7
    ]


def _reorder(
    coords: list[tuple[float, float, float]], perm: list[int]
) -> list[tuple[float, float, float]]:
    """Return coords reordered by perm."""
    return [coords[i] for i in perm]


# ---------------------------------------------------------------------------
# _dominant_axis
# ---------------------------------------------------------------------------

class TestDominantAxis:
    def test_x_dominant(self):
        assert _dominant_axis(1.0, 0.0, 0.0) == 0

    def test_y_dominant(self):
        assert _dominant_axis(0.0, 1.0, 0.0) == 1

    def test_z_dominant(self):
        assert _dominant_axis(0.0, 0.0, 1.0) == 2

    def test_negative_x_dominant(self):
        assert _dominant_axis(-0.9, 0.3, 0.1) == 0

    def test_tie_breaks_to_smallest_index(self):
        # x and y equal: x wins (index 0)
        assert _dominant_axis(1.0, 1.0, 0.0) == 0
        # y and z equal, x less: y wins (index 1)
        assert _dominant_axis(0.0, 1.0, 1.0) == 1
        # all equal: x wins (index 0) — EC5
        assert _dominant_axis(1.0, 1.0, 1.0) == 0


# ---------------------------------------------------------------------------
# build_block_axes
# ---------------------------------------------------------------------------

class TestBuildBlockAxes:
    def test_returns_three_axes(self):
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        assert len(axes) == 3

    def test_axis_indices(self):
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        assert [a.axis_index for a in axes] == [0, 1, 2]

    def test_block_id_set(self):
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(42, coords, names)
        assert all(a.block_id == 42 for a in axes)

    def test_unit_cube_edge_lengths(self):
        """All three axes of a unit cube must have edge length 1.0."""
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        for ax in axes:
            assert ax.edge_length == pytest.approx(1.0, abs=1e-10)

    def test_anisotropic_block_edge_lengths(self):
        """Axes of a 1x2x3 block must report their respective lengths."""
        coords = _anisotropic_block_coords(1.0, 2.0, 3.0)
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        assert axes[0].edge_length == pytest.approx(1.0, abs=1e-10)  # i-axis
        assert axes[1].edge_length == pytest.approx(2.0, abs=1e-10)  # j-axis
        assert axes[2].edge_length == pytest.approx(3.0, abs=1e-10)  # k-axis

    def test_unit_cube_dominant_axes(self):
        """Unit cube: i→X(0), j→Y(1), k→Z(2)."""
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        assert axes[0].dominant_global_axis == 0  # i → X
        assert axes[1].dominant_global_axis == 1  # j → Y
        assert axes[2].dominant_global_axis == 2  # k → Z

    def test_rotated_block_i_parallel_to_global_y(self):
        """EC1: Block rotated so i-direction is parallel to global Y.

        Build a block where v0→v1 runs along Y.
        """
        # Swap x and y for v1
        coords = [
            (0.0, 0.0, 0.0),  # v0
            (0.0, 1.0, 0.0),  # v1  (i-axis along Y)
            (1.0, 1.0, 0.0),  # v2
            (1.0, 0.0, 0.0),  # v3
            (0.0, 0.0, 1.0),  # v4
            (0.0, 1.0, 1.0),  # v5
            (1.0, 1.0, 1.0),  # v6
            (1.0, 0.0, 1.0),  # v7
        ]
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        # i-axis direction should be (0,1,0) → dominant = 1 (Y)
        assert axes[0].dominant_global_axis == 1

    def test_wrong_vertex_count_raises(self):
        coords = _unit_cube_coords()[:7]  # only 7 vertices
        names = _unit_cube_names()[:7]
        with pytest.raises(ValueError, match="8 vertices"):
            build_block_axes(0, coords, names)

    def test_45_degree_block_dominant_smallest_index(self):
        """EC5: Block with i-axis at 45° (equal X and Y) → dominant is 0 (X)."""
        d = 1.0 / math.sqrt(2.0)
        coords = [
            (0.0, 0.0, 0.0),
            (d,   d,   0.0),  # v1: 45° in XY plane
            (d,   d,   1.0),  # v2
            (0.0, 0.0, 1.0),  # v3
            (0.0, 1.0, 0.0),  # v4
            (d,   d+1, 0.0),  # v5
            (d,   d+1, 1.0),  # v6
            (0.0, 1.0, 1.0),  # v7
        ]
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)
        # i-axis: v0→v1 = (d, d, 0) → dominant must be 0 (tie → smallest)
        assert axes[0].dominant_global_axis == 0


# ---------------------------------------------------------------------------
# AxisEquivalenceClasses (from test_block_axis.py)
# ---------------------------------------------------------------------------

class TestAxisEquivalenceClassesSingleBlock:
    def test_single_block_three_singleton_classes(self):
        """A single block's three axes must be in three separate classes."""
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(axes)
        aec.build()

        classes = aec.all_classes()
        # No two axes of the same block share a face in the same orientation
        # so they must remain in distinct classes
        assert len(classes) == 3

    def test_query_before_build_raises(self):
        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, _unit_cube_coords(), _unit_cube_names()))
        with pytest.raises(RuntimeError, match="build"):
            aec.class_of(0)


class TestAxisEquivalenceClassesTwoAdjacentBlocks:
    """HP2: Two unit cubes sharing a face (block0 at X=0..1, block1 at X=1..2)."""

    def _build_two_adjacent(self):
        """Return AxisEquivalenceClasses for two adjacent unit cubes."""
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = [f"v{i}" for i in range(8)]

        coords1 = _unit_cube_coords(offset_x=1.0)
        names1 = ["v1", "v8", "v9", "v2", "v5", "v10", "v11", "v6"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords0, names0))
        aec.add_block_axes(build_block_axes(1, coords1, names1))
        aec.build()
        return aec

    def test_six_axes_registered(self):
        aec = self._build_two_adjacent()
        assert aec.num_axes() == 6

    def test_i_axes_in_separate_classes(self):
        """i-axes of both blocks are perpendicular to the shared face and share no
        edges with each other → they remain in separate equivalence classes."""
        aec = self._build_two_adjacent()
        # block0 i-axis is index 0; block1 i-axis is index 3
        assert aec.class_of(0) != aec.class_of(3)

    def test_j_axes_in_same_class(self):
        """j-axes of both blocks must be in the same class via edge-based union."""
        aec = self._build_two_adjacent()
        assert aec.class_of(1) == aec.class_of(4)

    def test_k_axes_in_same_class(self):
        """k-axes of both blocks must be in the same class via edge-based union."""
        aec = self._build_two_adjacent()
        assert aec.class_of(2) == aec.class_of(5)

    def test_four_classes_total(self):
        """Two adjacent unit cubes → 4 classes total."""
        aec = self._build_two_adjacent()
        classes = aec.all_classes()
        assert len(classes) == 4


class TestAxisEquivalenceClassesDisjointBlocks:
    """Two unit cubes with no shared vertices → all 6 axes in separate classes."""

    def test_six_separate_classes(self):
        coords0 = _unit_cube_coords(offset_x=0.0)
        names0 = [f"v{i}" for i in range(8)]
        coords1 = _unit_cube_coords(offset_x=5.0)
        names1 = [f"v{i}" for i in range(8, 16)]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords0, names0))
        aec.add_block_axes(build_block_axes(1, coords1, names1))
        aec.build()

        classes = aec.all_classes()
        assert len(classes) == 6


class TestInPlaneAxisUnion:
    """Tests for the edge-based axis union mechanism."""

    def _build_two_blocks_shared_y_face(self):
        coords_a = [
            (0.0,  0.0,   0.0),  # v0
            (4.0,  0.0,   0.0),  # v1
            (4.0, 11.25,  0.0),  # v2
            (0.0, 11.25,  0.0),  # v3
            (0.0,  0.0,   1.0),  # v4
            (4.0,  0.0,   1.0),  # v5
            (4.0, 11.25,  1.0),  # v6
            (0.0, 11.25,  1.0),  # v7
        ]
        names_a = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

        coords_b = [
            (0.0,   0.0,    0.0),  # b0 = s0
            (4.0,   0.0,    0.0),  # b1 = s1
            (4.0, -16.17,   0.0),  # b2
            (0.0, -16.17,   0.0),  # b3
            (0.0,   0.0,    1.0),  # b4 = s4
            (4.0,   0.0,    1.0),  # b5 = s5
            (4.0, -16.17,   1.0),  # b6
            (0.0, -16.17,   1.0),  # b7
        ]
        names_b = ["s0", "s1", "b2", "b3", "s4", "s5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()
        return aec

    def test_t1_i_axes_united_via_in_plane(self):
        """T1: i-axes of two blocks sharing a face are united via edge-based union."""
        aec = self._build_two_blocks_shared_y_face()
        assert aec.class_of(0) == aec.class_of(3), (
            "i-axes must be in the same equivalence class via edge-based union"
        )

    def test_t1_k_axes_united_via_in_plane(self):
        """T1: k-axes of two blocks sharing a face are united via edge-based union."""
        aec = self._build_two_blocks_shared_y_face()
        assert aec.class_of(2) == aec.class_of(5), (
            "k-axes must be in the same equivalence class via edge-based union"
        )

    def test_t1_four_classes_total(self):
        """T1: two blocks sharing Y=0 face → 4 equivalence classes."""
        aec = self._build_two_blocks_shared_y_face()
        classes = aec.all_classes()
        assert len(classes) == 4, (
            f"Expected 4 classes (i merged, j separate, k merged), got {len(classes)}"
        )

    def test_t2_l_arrangement_transitive(self):
        """T2: Three blocks in L-shape — in-plane axes transitively in same class."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]
        names_b = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

        coords_c = [
            (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0),
            (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (0.0, 2.0, 1.0),
        ]
        names_c = ["a3", "a2", "c2", "c3", "a7", "a6", "c6", "c7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.add_block_axes(build_block_axes(2, coords_c, names_c))
        aec.build()

        k_a = aec.class_of(2)
        k_b = aec.class_of(5)
        k_c = aec.class_of(8)
        assert k_a == k_b, "A.k and B.k must be in same class (edge-based on X=1 face)"
        assert k_a == k_c, "A.k and C.k must be in same class (edge-based on Y=1 face)"

    def test_t3a_near_parallel_united(self):
        """T3a: Two blocks sharing a face with aligned i-axes must be united."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (0.0,         0.0, 0.0),
            (1.0,         0.0, 0.0),
            (1.0,        -1.0, 0.0),
            (0.0,        -1.0, 0.0),
            (0.0,         0.0, 1.0),
            (1.0,         0.0, 1.0),
            (1.0,        -1.0, 1.0),
            (0.0,        -1.0, 1.0),
        ]
        names_b = ["a0", "a1", "b2", "b3", "a4", "a5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        assert aec.class_of(0) == aec.class_of(3), (
            "i-axes sharing edges must be united"
        )

    def test_t3b_non_parallel_not_united(self):
        """T3b: Block B rotated 30 degrees, no shared vertices → no union."""
        import math as _math
        angle = _math.radians(30)
        c, s = _math.cos(angle), _math.sin(angle)

        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = [f"a{i}" for i in range(8)]

        coords_b = [
            (5.0,        5.0,       0.0),
            (5.0 + c,    5.0 + s,   0.0),
            (5.0 + c - s, 5.0 + s + c, 0.0),
            (5.0 - s,    5.0 + c,   0.0),
            (5.0,        5.0,       1.0),
            (5.0 + c,    5.0 + s,   1.0),
            (5.0 + c - s, 5.0 + s + c, 1.0),
            (5.0 - s,    5.0 + c,   1.0),
        ]
        names_b = [f"b{i}" for i in range(8)]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        classes = aec.all_classes()
        assert len(classes) == 6, (
            f"Disjoint blocks with no shared vertices must have 6 separate classes, "
            f"got {len(classes)}"
        )

    def test_t4_edge_sharing_no_union(self):
        """T4: Two blocks sharing exactly one edge must NOT be united."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = [f"a{i}" for i in range(8)]

        coords_b = [
            (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (2.0, 2.0, 0.0), (1.0, 2.0, 0.0),
            (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 2.0, 1.0), (1.0, 2.0, 1.0),
        ]
        names_b = ["a2", "b1", "b2", "b3", "a6", "b5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        classes = aec.all_classes()
        assert len(classes) == 6, (
            f"Edge-only contact must produce 6 separate classes, got {len(classes)}"
        )

    def test_t5_mirrored_vertex_numbering(self):
        """T5: Block B has mirrored local vertex numbering; shared edges still unite axes."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 0.0, 1.0), (1.0, 0.0, 1.0),
        ]
        names_b = ["a2", "b1", "b2", "a1", "a6", "b5", "b6", "a5"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        assert aec.class_of(0) != aec.class_of(3), (
            "A.i and B.i (both perpendicular to shared face) share no edges → separate classes"
        )
        assert aec.class_of(1) == aec.class_of(4), (
            "A.j must be united with B.j via shared edge"
        )
        assert aec.class_of(2) == aec.class_of(5), (
            "A.k must be united with B.k via shared edge"
        )


class TestAxisEquivalenceClassesDegenerateEdgeLength:
    """EC6: degenerate block where all vertices are coplanar."""

    def test_coplanar_block_raises_topology_error(self):
        """Block with all vertices in a single plane raises TopologyError."""
        coords = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        names = _unit_cube_names()
        with pytest.raises(TopologyError):
            build_block_axes(0, coords, names)


# ---------------------------------------------------------------------------
# TestEdgeBasedUnion
# ---------------------------------------------------------------------------

class TestEdgeBasedUnion:
    """Tests specifically targeting the _union_via_shared_edges mechanism."""

    def test_union_skewed_block_shares_edge_with_axis_aligned(self):
        """Skewed block shares edges on one face with axis-aligned neighbour."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 0.0, 0.0),  # b0 = a1
            (2.0, 0.3, 0.0),  # b1 — skewed in y
            (2.0, 1.3, 0.0),  # b2 — skewed
            (1.0, 1.0, 0.0),  # b3 = a2
            (1.0, 0.0, 1.0),  # b4 = a5
            (2.0, 0.3, 1.0),  # b5 — skewed
            (2.0, 1.3, 1.0),  # b6 — skewed
            (1.0, 1.0, 1.0),  # b7 = a6
        ]
        names_b = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        assert aec.class_of(0) != aec.class_of(3), (
            "A.i and B.i share no edges and must be in separate classes"
        )
        assert aec.class_of(1) == aec.class_of(4), (
            "A.j and B.j must be united via shared edge {a1,a2}"
        )
        assert aec.class_of(2) == aec.class_of(5), (
            "A.k and B.k must be united via shared edges {a1,a5} and {a2,a6}"
        )

    def test_no_union_when_only_vertex_shared(self):
        """Two hexes sharing exactly one vertex stay in separate classes."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (2.0, 2.0, 0.0), (1.0, 2.0, 0.0),
            (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 2.0, 1.0), (1.0, 2.0, 1.0),
        ]
        names_b = ["a2", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        classes = aec.all_classes()
        assert len(classes) == 6, (
            f"Vertex-only contact must produce 6 separate classes, got {len(classes)}"
        )

    def test_union_collinear_three_blocks(self):
        """Three blocks in a row along x — all i-axes in one class (transitivity)."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]
        names_b = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

        coords_c = [
            (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0), (3.0, 0.0, 1.0), (3.0, 1.0, 1.0), (2.0, 1.0, 1.0),
        ]
        names_c = ["b1", "c1", "c2", "b2", "b5", "c5", "c6", "b6"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.add_block_axes(build_block_axes(2, coords_c, names_c))
        aec.build()

        assert aec.class_of(0) != aec.class_of(3), "A.i and B.i share no edges → separate"
        assert aec.class_of(3) != aec.class_of(6), "B.i and C.i share no edges → separate"
        assert aec.class_of(1) == aec.class_of(4), "A.j and B.j must be in same class"
        assert aec.class_of(4) == aec.class_of(7), "B.j and C.j must be in same class"
        assert aec.class_of(2) == aec.class_of(5), "A.k and B.k must be in same class"
        assert aec.class_of(5) == aec.class_of(8), "B.k and C.k must be in same class"

        classes = aec.all_classes()
        assert len(classes) == 5, (
            f"Three collinear blocks: i-axes separate, j/k merged → 5 classes, got {len(classes)}"
        )

    def test_no_union_when_only_edge_shared(self):
        """Two hexes sharing exactly one edge must NOT be united."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = [f"a{i}" for i in range(8)]

        coords_b = [
            (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (2.0, 2.0, 0.0), (1.0, 2.0, 0.0),
            (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 2.0, 1.0), (1.0, 2.0, 1.0),
        ]
        names_b = ["a2", "b1", "b2", "b3", "a6", "b5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        classes = aec.all_classes()
        assert len(classes) == 6, (
            f"Edge-only contact must produce 6 separate classes, got {len(classes)}"
        )
        assert aec.class_of(2) != aec.class_of(5), (
            "k-axes must NOT be united when only an edge is shared"
        )

    def test_block_vertex_names_and_coords_attached_to_axes(self):
        """After build_block_axes, all 3 BlockAxis objects have block_vertex_names and
        block_vertex_coords of length 8."""
        coords = _unit_cube_coords()
        names = _unit_cube_names()
        axes = build_block_axes(0, coords, names)

        assert len(axes) == 3
        expected_names = tuple(names)
        expected_coords = tuple((float(c[0]), float(c[1]), float(c[2])) for c in coords)
        for ax in axes:
            assert ax.block_vertex_names == expected_names
            assert ax.block_vertex_coords == expected_coords
            assert len(ax.block_vertex_names) == 8
            assert len(ax.block_vertex_coords) == 8

    def test_face_sharing_unites_two_in_plane_axis_pairs(self):
        """Two face-adjacent blocks → exactly 2 unions, total 4 equivalence classes."""
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]
        names_b = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        assert aec.class_of(0) != aec.class_of(3), "A.i and B.i must be separate"
        assert aec.class_of(1) == aec.class_of(4), "A.j and B.j must be united"
        assert aec.class_of(2) == aec.class_of(5), "A.k and B.k must be united"

        classes = aec.all_classes()
        assert len(classes) == 4, (
            f"Face-adjacent blocks must produce 4 classes, got {len(classes)}"
        )

    def test_skewed_block_face_sharing(self):
        """Axis-aligned + skewed block sharing a face.

        Yields the correct 2 unions via vertex-edge matching.
        """
        coords_a = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        names_a = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

        coords_b = [
            (1.0, 0.0, 0.0),  # b0 = a1
            (2.0, 0.3, 0.0),  # b1 — skewed
            (2.0, 1.3, 0.0),  # b2 — skewed
            (1.0, 1.0, 0.0),  # b3 = a2
            (1.0, 0.0, 1.0),  # b4 = a5
            (2.0, 0.3, 1.0),  # b5 — skewed
            (2.0, 1.3, 1.0),  # b6 — skewed
            (1.0, 1.0, 1.0),  # b7 = a6
        ]
        names_b = ["a1", "b1", "b2", "a2", "a5", "b5", "b6", "a6"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords_a, names_a))
        aec.add_block_axes(build_block_axes(1, coords_b, names_b))
        aec.build()

        assert aec.class_of(1) == aec.class_of(4), "A.j and B.j must be united via shared face"
        assert aec.class_of(2) == aec.class_of(5), "A.k and B.k must be united via shared face"
        assert aec.class_of(0) != aec.class_of(3), "A.i and B.i must be separate"

        classes = aec.all_classes()
        assert len(classes) == 4, (
            f"Skewed block face-sharing must produce 4 classes, got {len(classes)}"
        )


# ---------------------------------------------------------------------------
# detect_hex_faces (from test_hex_face_detection.py)
# ---------------------------------------------------------------------------

class TestDetectHexFaces:

    def test_detect_canonical_unit_cube(self):
        """Canonical unit cube → exactly 6 faces, valid hex topology."""
        coords = _canonical_unit_cube()
        faces = detect_hex_faces(coords)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)

    def test_detect_pattern_a_top_ring_reversed(self):
        """Unit cube with V4..V7 in CW instead of CCW → 6 faces detected."""
        coords = _canonical_unit_cube()
        perm = [0, 1, 2, 3, 5, 4, 6, 7]
        reordered = _reorder(coords, perm)
        faces = detect_hex_faces(reordered)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)

    def test_detect_pattern_b_swapped(self):
        """Unit cube with V1 and V3 swapped → 6 faces detected."""
        coords = _canonical_unit_cube()
        perm = [0, 3, 2, 1, 4, 7, 6, 5]
        reordered = _reorder(coords, perm)
        faces = detect_hex_faces(reordered)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)

    def test_detect_pattern_c_elongated(self):
        """Elongated block (z=10x xy) in canonical order → 6 faces detected."""
        coords = [
            (0.0, 0.0,  0.0),
            (1.0, 0.0,  0.0),
            (1.0, 1.0,  0.0),
            (0.0, 1.0,  0.0),
            (0.0, 0.0, 10.0),
            (1.0, 0.0, 10.0),
            (1.0, 1.0, 10.0),
            (0.0, 1.0, 10.0),
        ]
        faces = detect_hex_faces(coords)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)

    def test_detect_skewed_hex(self):
        """Lightly skewed hex (parallelogram-like) → 6 faces detected."""
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.1, 0.0),
            (1.1, 1.1, 0.0),
            (0.1, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.1, 1.0),
            (1.1, 1.1, 1.0),
            (0.1, 1.0, 1.0),
        ]
        faces = detect_hex_faces(coords)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)

    def test_detect_degenerate_all_coplanar(self):
        """8 points all in one plane → TopologyError raised."""
        coords = [
            (float(i), float(j), 0.0)
            for i in range(4)
            for j in range(2)
        ]
        with pytest.raises(TopologyError):
            detect_hex_faces(coords)

    def test_detect_degenerate_coincident(self):
        """8 coincident points → TopologyError raised."""
        coords = [(1.0, 2.0, 3.0)] * 8
        with pytest.raises(TopologyError):
            detect_hex_faces(coords)

    def test_detect_hex_with_curved_edge_emulation(self):
        """Unit cube with one vertex slightly off-plane.

        6 faces detected (tolerance escalation).
        """
        import math as _math
        coords = [
            (0.0, 0.0, 0.0),  # v0
            (1.0, 0.0, 0.0),  # v1
            (1.0, 1.0, 0.0),  # v2
            (0.0, 1.0, 0.0),  # v3
            (0.0, 0.0, 1.0),  # v4
            (1.0, 0.0, 1.0),  # v5
            (1.0, 1.0, 1.0),  # v6
            (0.0, 1.0, 1.0),  # v7
        ]
        max_diag = _math.sqrt(3.0)
        perturbation = 0.001 * max_diag
        coords[6] = (1.0, 1.0, 1.0 + perturbation)

        faces = detect_hex_faces(coords)
        assert len(faces) == 6
        assert _validate_hex_topology(faces)


# ---------------------------------------------------------------------------
# compute_axis_class_per_edge (from test_hex_face_detection.py)
# ---------------------------------------------------------------------------

class TestComputeAxisClassPerEdge:

    def test_canonical_three_classes_four_edges_each(self):
        """Canonical unit cube → exactly 3 axis classes, each with 4 edges."""
        coords = _canonical_unit_cube()
        faces = detect_hex_faces(coords)
        edge_to_class = compute_axis_class_per_edge(faces)

        assert len(edge_to_class) == 12

        classes: dict = {}
        for edge, root in edge_to_class.items():
            classes.setdefault(root, []).append(edge)

        assert len(classes) == 3, f"Expected 3 classes, got {len(classes)}"
        for root, edges in classes.items():
            assert len(edges) == 4, f"Class {root} has {len(edges)} edges, expected 4"

    def test_canonical_edge_0_1_class_contains_expected_edges(self):
        """Edge {0,1} must be in the same class as {2,3}, {4,5}, {6,7}."""
        coords = _canonical_unit_cube()
        faces = detect_hex_faces(coords)
        edge_to_class = compute_axis_class_per_edge(faces)

        root_01 = edge_to_class[frozenset({0, 1})]
        same_class = {e for e, r in edge_to_class.items() if r == root_01}
        expected = {
            frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5}), frozenset({6, 7}),
        }
        assert same_class == expected


# ---------------------------------------------------------------------------
# map_class_to_local_axis_index (from test_hex_face_detection.py)
# ---------------------------------------------------------------------------

class TestMapClassToLocalAxisIndex:

    def test_canonical_v0_neighbours_sorted(self):
        """Canonical cube: V0 neighbours are V1, V3, V4 → axis 0/1/2 respectively."""
        coords = _canonical_unit_cube()
        faces = detect_hex_faces(coords)
        edge_to_class = compute_axis_class_per_edge(faces)
        class_to_axis = map_class_to_local_axis_index(edge_to_class, faces)

        assert len(class_to_axis) == 3

        root_v0v1 = edge_to_class[frozenset({0, 1})]
        assert class_to_axis[root_v0v1] == 0

        root_v0v3 = edge_to_class[frozenset({0, 3})]
        assert class_to_axis[root_v0v3] == 1

        root_v0v4 = edge_to_class[frozenset({0, 4})]
        assert class_to_axis[root_v0v4] == 2

    def test_all_three_axis_indices_assigned(self):
        """Each axis index 0, 1, 2 must be assigned exactly once."""
        coords = _canonical_unit_cube()
        faces = detect_hex_faces(coords)
        edge_to_class = compute_axis_class_per_edge(faces)
        class_to_axis = map_class_to_local_axis_index(edge_to_class, faces)

        assigned = sorted(class_to_axis.values())
        assert assigned == [0, 1, 2]


# ---------------------------------------------------------------------------
# Integration: AxisEquivalenceClasses with non-canonical orderings
# (from test_hex_face_detection.py)
# ---------------------------------------------------------------------------

class TestUnionPatternAMeetsCanonical:
    """Block 1 canonical, Block 2 with top ring reversed, shared top face → axes united."""

    def _build(self):
        coords0 = _canonical_unit_cube()
        names0 = [f"a{i}" for i in range(8)]

        coords1 = [
            (0.0, 0.0, 1.0),  # b0 = a4
            (1.0, 0.0, 1.0),  # b1 = a5
            (1.0, 1.0, 1.0),  # b2 = a6
            (0.0, 1.0, 1.0),  # b3 = a7
            (0.0, 0.0, 2.0),  # b4
            (1.0, 0.0, 2.0),  # b5
            (1.0, 1.0, 2.0),  # b6
            (0.0, 1.0, 2.0),  # b7
        ]
        names1 = ["a4", "a5", "a6", "a7", "b4", "b5", "b6", "b7"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords0, names0))
        aec.add_block_axes(build_block_axes(1, coords1, names1))
        aec.build()
        return aec

    def test_four_classes_total(self):
        """Canonical stacked blocks → 4 classes (k-axes separate, i/j axes merged)."""
        aec = self._build()
        classes = aec.all_classes()
        assert len(classes) == 4

    def test_in_plane_axes_united(self):
        """i and j axes of both blocks must be in the same equivalence classes."""
        aec = self._build()
        assert aec.class_of(0) == aec.class_of(3), "i-axes must be united"
        assert aec.class_of(1) == aec.class_of(4), "j-axes must be united"
        assert aec.class_of(2) != aec.class_of(5), "k-axes must remain separate"


class TestUnionPatternBMeetsCanonical:
    """Block 1 canonical, Block 2 with V1/V3 swapped (pattern B), shared face → axes united."""

    def _build(self):
        coords0 = _canonical_unit_cube()
        names0 = [f"a{i}" for i in range(8)]

        coords1 = [
            (1.0, 1.0, 0.0),  # b0 = a2
            (2.0, 1.0, 0.0),  # b1
            (2.0, 0.0, 0.0),  # b2
            (1.0, 0.0, 0.0),  # b3 = a1
            (1.0, 1.0, 1.0),  # b4 = a6
            (2.0, 1.0, 1.0),  # b5
            (2.0, 0.0, 1.0),  # b6
            (1.0, 0.0, 1.0),  # b7 = a5
        ]
        names1 = ["a2", "b1", "b2", "a1", "a6", "b5", "b6", "a5"]

        aec = AxisEquivalenceClasses()
        aec.add_block_axes(build_block_axes(0, coords0, names0))
        aec.add_block_axes(build_block_axes(1, coords1, names1))
        aec.build()
        return aec

    def test_four_classes_total(self):
        """Pattern B adjacent blocks → 4 classes."""
        aec = self._build()
        classes = aec.all_classes()
        assert len(classes) == 4

    def test_in_plane_axes_united(self):
        """In-plane axes on shared X=1 face must be united across blocks."""
        aec = self._build()
        assert aec.class_of(0) != aec.class_of(3), "i-axes must remain separate"
        j_united = aec.class_of(1) == aec.class_of(4)
        k_united = aec.class_of(2) == aec.class_of(5)
        j_b_united = aec.class_of(1) == aec.class_of(5)
        k_b_united = aec.class_of(2) == aec.class_of(4)
        assert (j_united and k_united) or (j_b_united and k_b_united), (
            "In-plane axes must be united (possibly with axis index swap due to orientation)"
        )
