"""Unit tests for meshing_utils.operations.scale.

Covers validate_factors, is_uniform, scale() with vertices/edges, empty
blocks warning, arc warning, and immutability of the source mesh.
"""

from __future__ import annotations

import pytest

from meshing_utils import Block, BlockMeshDict, Edge, Face, Patch, Vertex
from meshing_utils.operations.scale import is_uniform, scale, validate_factors

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_bmd(
    vertices: list[tuple[float, float, float]] | None = None,
    edges: list[Edge] | None = None,
    with_block: bool = True,
) -> BlockMeshDict:
    """Return a BlockMeshDict populated with the given vertices and edges.

    If *with_block* is True (default) a single hex block is added so that
    the mesh is "valid" from a topology perspective.  When *vertices* is
    None the default unit-cube vertices (v0..v7) are used.
    """
    bmd = BlockMeshDict()

    if vertices is None:
        default_coords = [
            (0.0, 0.0, 0.0),  # v0
            (1.0, 0.0, 0.0),  # v1
            (1.0, 1.0, 0.0),  # v2
            (0.0, 1.0, 0.0),  # v3
            (0.0, 0.0, 1.0),  # v4
            (1.0, 0.0, 1.0),  # v5
            (1.0, 1.0, 1.0),  # v6
            (0.0, 1.0, 1.0),  # v7
        ]
        for i, c in enumerate(default_coords):
            bmd.vertices.add(Vertex(f"v{i}", list(c)))
    else:
        for i, c in enumerate(vertices):
            bmd.vertices.add(Vertex(f"v{i}", list(c)))

    if edges:
        for e in edges:
            bmd.edges.add(e)

    if with_block and len(bmd.vertices) >= 8:
        bmd.blocks.add(
            Block(
                name_or_string="block0",
                vertices=[f"v{i}" for i in range(8)],
                cells=[1, 1, 1],
                type="hex",
                grading_type="simpleGrading",
                grading_def=[1.0, 1.0, 1.0],
            )
        )

    return bmd


# ---------------------------------------------------------------------------
# validate_factors
# ---------------------------------------------------------------------------

class TestValidateFactors:

    def test_valid_uniform_factors(self):
        """Three equal positive floats must pass without error."""
        result = validate_factors([2.0, 2.0, 2.0])
        assert result == pytest.approx((2.0, 2.0, 2.0))

    def test_valid_non_uniform_factors(self):
        """Three distinct positive floats must pass without error."""
        result = validate_factors([1.0, 2.0, 3.0])
        assert result == pytest.approx((1.0, 2.0, 3.0))

    def test_small_positive_factor_allowed(self):
        """Very small positive values (> 0) must be accepted."""
        result = validate_factors([1e-15, 1e-15, 1e-15])
        assert result[0] == pytest.approx(1e-15)

    def test_zero_factor_raises(self):
        """A factor of exactly 0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            validate_factors([1.0, 0.0, 1.0])

    def test_negative_factor_raises(self):
        """A negative factor must raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            validate_factors([-1.0, 1.0, 1.0])

    def test_nan_raises(self):
        """A NaN factor must raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            validate_factors([1.0, float("nan"), 1.0])

    def test_inf_raises(self):
        """An infinite factor must raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            validate_factors([float("inf"), 1.0, 1.0])

    def test_neg_inf_raises(self):
        """A negative-infinite factor must raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            validate_factors([1.0, float("-inf"), 1.0])

    def test_too_few_raises(self):
        """Fewer than 3 factors must raise ValueError."""
        with pytest.raises(ValueError, match="3"):
            validate_factors([1.0, 2.0])

    def test_too_many_raises(self):
        """More than 3 factors must raise ValueError."""
        with pytest.raises(ValueError, match="3"):
            validate_factors([1.0, 2.0, 3.0, 4.0])

    def test_returns_tuple(self):
        """Return type must be a 3-tuple."""
        result = validate_factors([1.0, 2.0, 3.0])
        assert isinstance(result, tuple)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# is_uniform
# ---------------------------------------------------------------------------

class TestIsUniform:

    def test_all_equal_is_uniform(self):
        assert is_uniform(2.0, 2.0, 2.0) is True

    def test_different_factors_not_uniform(self):
        assert is_uniform(1.0, 2.0, 1.0) is False

    def test_near_equal_within_rtol_is_uniform(self):
        """Values within the default relative tolerance must be considered uniform."""
        v = 1.0
        assert is_uniform(v, v + 1e-12, v + 1e-12) is True

    def test_outside_rtol_is_not_uniform(self):
        assert is_uniform(1.0, 1.1, 1.0) is False


# ---------------------------------------------------------------------------
# scale — vertex coordinates
# ---------------------------------------------------------------------------

class TestScaleVertices:

    def test_uniform_scale_doubles_coords(self):
        """Uniform factor 2: all coordinates must be doubled."""
        bmd = _make_bmd()
        result = scale(bmd, 2.0, 2.0, 2.0)

        original_coords = [list(v.coords) for v in bmd.vertices]
        for i, v in enumerate(result.vertices):
            expected = [c * 2.0 for c in original_coords[i]]
            assert v.coords == pytest.approx(expected, abs=1e-12)

    def test_non_uniform_scale_applies_per_axis(self):
        """Non-uniform factors (fx=2, fy=3, fz=4) scale each axis independently."""
        bmd = _make_bmd(vertices=[(1.0, 1.0, 1.0)])
        result = scale(bmd, 2.0, 3.0, 4.0)

        v = result.vertices[0]
        assert v.coords == pytest.approx([2.0, 3.0, 4.0], abs=1e-12)

    def test_factor_one_leaves_coords_unchanged(self):
        """Identity scale (1, 1, 1) must not change any coordinates."""
        bmd = _make_bmd()
        original_coords = [list(v.coords) for v in bmd.vertices]
        result = scale(bmd, 1.0, 1.0, 1.0)

        for i, v in enumerate(result.vertices):
            assert v.coords == pytest.approx(original_coords[i], abs=1e-12)

    def test_vertex_names_preserved(self):
        """Vertex names must be identical in the result."""
        bmd = _make_bmd()
        original_names = [v.name for v in bmd.vertices]
        result = scale(bmd, 2.0, 2.0, 2.0)

        assert [v.name for v in result.vertices] == original_names

    def test_zero_coord_scales_to_zero(self):
        """A coordinate at the origin stays at zero regardless of factor."""
        bmd = _make_bmd(vertices=[(0.0, 0.0, 0.0)])
        result = scale(bmd, 5.0, 7.0, 3.0)
        assert result.vertices[0].coords == pytest.approx([0.0, 0.0, 0.0])

    def test_fractional_factor(self):
        """A factor < 1 (but > 0) must shrink coordinates."""
        bmd = _make_bmd(vertices=[(4.0, 6.0, 8.0)])
        result = scale(bmd, 0.5, 0.5, 0.5)
        assert result.vertices[0].coords == pytest.approx([2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# scale — edge control points
# ---------------------------------------------------------------------------

class TestScaleEdges:

    def test_spline_edge_points_are_scaled(self):
        """A spline edge with multiple control points must have all of them scaled."""
        edge = Edge(
            "spline", "v0", "v1",
            points=[[0.5, 0.0, 0.0], [0.75, 0.1, 0.0], [0.9, 0.0, 0.0]],
        )
        bmd = _make_bmd(edges=[edge])
        result = scale(bmd, 2.0, 3.0, 4.0)

        scaled_points = result.edges[0].points
        assert scaled_points[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
        assert scaled_points[1] == pytest.approx([1.5, 0.3, 0.0], abs=1e-12)
        assert scaled_points[2] == pytest.approx([1.8, 0.0, 0.0], abs=1e-12)

    def test_arc_edge_midpoint_scaled_uniformly(self):
        """An arc edge with uniform scale: midpoint is correctly scaled."""
        arc = Edge("arc", "v0", "v1", points=[[1.5, 0.1, 0.0]])
        bmd = _make_bmd(edges=[arc])
        result = scale(bmd, 2.0, 2.0, 2.0)

        assert result.edges[0].points[0] == pytest.approx([3.0, 0.2, 0.0], abs=1e-12)

    def test_arc_edge_non_uniform_warns_once(self, caplog):
        """Non-uniform scaling of arc edges must emit exactly one warning."""
        import logging
        arc1 = Edge("arc", "v0", "v1", points=[[1.5, 0.1, 0.0]])
        arc2 = Edge("arc", "v2", "v3", points=[[2.5, 0.2, 0.0]])
        bmd = _make_bmd(edges=[arc1, arc2])

        with caplog.at_level(logging.WARNING, logger="meshing_utils.operations.scale"):
            scale(bmd, 2.0, 3.0, 4.0)

        warning_records = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "arc" in r.message.lower()
        ]
        assert len(warning_records) == 1, (
            f"Expected exactly 1 arc warning, got {len(warning_records)}"
        )

    def test_arc_edge_uniform_no_warning(self, caplog):
        """Uniform scaling of arc edges must NOT emit an arc warning."""
        import logging
        arc = Edge("arc", "v0", "v1", points=[[1.5, 0.1, 0.0]])
        bmd = _make_bmd(edges=[arc])

        with caplog.at_level(logging.WARNING, logger="meshing_utils.operations.scale"):
            scale(bmd, 2.0, 2.0, 2.0)

        arc_warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "arc" in r.message.lower()
        ]
        assert len(arc_warnings) == 0

    def test_non_arc_edge_non_uniform_no_arc_warning(self, caplog):
        """Non-uniform scaling of non-arc edges must not trigger the arc warning."""
        import logging
        spline = Edge("spline", "v0", "v1", points=[[0.5, 0.1, 0.0]])
        bmd = _make_bmd(edges=[spline])

        with caplog.at_level(logging.WARNING, logger="meshing_utils.operations.scale"):
            scale(bmd, 1.0, 2.0, 3.0)

        arc_warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "arc" in r.message.lower()
        ]
        assert len(arc_warnings) == 0

    def test_no_edges_does_not_raise(self):
        """A mesh with no edges must scale without error."""
        bmd = _make_bmd()
        result = scale(bmd, 2.0, 2.0, 2.0)
        assert len(result.edges) == 0


# ---------------------------------------------------------------------------
# scale — unchanged sections
# ---------------------------------------------------------------------------

class TestScaleUnchangedSections:

    def test_convert_to_meters_unchanged(self):
        """convertToMeters must not be modified by scale()."""
        bmd = _make_bmd()
        bmd.convertToMeters = 0.001
        result = scale(bmd, 2.0, 2.0, 2.0)
        assert result.convertToMeters == pytest.approx(0.001)

    def test_geometry_body_unchanged(self):
        """The geometry_body string must be copied verbatim."""
        bmd = _make_bmd()
        bmd.geometry_body = "sphere { type searchableSphere; }"
        result = scale(bmd, 2.0, 2.0, 2.0)
        assert result.geometry_body == bmd.geometry_body

    def test_blocks_unchanged(self):
        """Block vertex references and cell counts must be identical in the result."""
        bmd = _make_bmd()
        src_block = bmd.blocks[0]
        result = scale(bmd, 3.0, 3.0, 3.0)
        res_block = result.blocks[0]
        assert res_block.vertices == src_block.vertices
        assert res_block.cells == src_block.cells
        assert res_block.grading_def == pytest.approx(src_block.grading_def)

    def test_boundary_patches_unchanged(self):
        """Boundary patch names, types, and face vertex lists must be unchanged."""
        bmd = _make_bmd()
        patch = Patch(
            name_or_string="outlet",
            type="patch",
            faces=[Face(vertices_or_string=["v4", "v5", "v6", "v7"])],
        )
        bmd.boundary.add(patch)

        result = scale(bmd, 2.0, 2.0, 2.0)
        res_patch = result.boundary.get("outlet")
        assert res_patch.type == "patch"
        assert res_patch.faces[0].vertices == ["v4", "v5", "v6", "v7"]


# ---------------------------------------------------------------------------
# scale — immutability of source mesh
# ---------------------------------------------------------------------------

class TestScaleDoesNotMutateSource:

    def test_source_vertices_unchanged(self):
        """scale() must not modify the source mesh vertices."""
        bmd = _make_bmd()
        original_coords = [list(v.coords) for v in bmd.vertices]
        scale(bmd, 5.0, 5.0, 5.0)

        for i, v in enumerate(bmd.vertices):
            assert v.coords == pytest.approx(original_coords[i], abs=1e-12)

    def test_source_edges_unchanged(self):
        """scale() must not modify the source mesh edge control points."""
        arc = Edge("arc", "v0", "v1", points=[[1.5, 0.1, 0.0]])
        bmd = _make_bmd(edges=[arc])
        original_points = [list(p) for p in bmd.edges[0].points]

        scale(bmd, 2.0, 3.0, 4.0)

        for i, pt in enumerate(bmd.edges[0].points):
            assert pt == pytest.approx(original_points[i], abs=1e-12)

    def test_source_block_count_unchanged(self):
        """scale() must not add or remove blocks from the source mesh."""
        bmd = _make_bmd()
        original_count = len(bmd.blocks)
        scale(bmd, 2.0, 2.0, 2.0)
        assert len(bmd.blocks) == original_count


# ---------------------------------------------------------------------------
# scale — empty blocks warning
# ---------------------------------------------------------------------------

class TestScaleEmptyBlocksWarning:

    def test_empty_blocks_emits_warning(self, caplog):
        """A mesh with no blocks must cause a WARNING log message."""
        import logging
        bmd = BlockMeshDict()
        bmd.vertices.add(Vertex("v0", [1.0, 2.0, 3.0]))

        with caplog.at_level(logging.WARNING, logger="meshing_utils.operations.scale"):
            scale(bmd, 2.0, 2.0, 2.0)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_empty_blocks_does_not_raise(self):
        """A mesh with no blocks must not raise an exception."""
        bmd = BlockMeshDict()
        bmd.vertices.add(Vertex("v0", [1.0, 0.0, 0.0]))
        result = scale(bmd, 2.0, 2.0, 2.0)
        assert result.vertices[0].coords == pytest.approx([2.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# scale — invalid factors passed directly
# ---------------------------------------------------------------------------

class TestScaleInvalidFactors:

    def test_zero_factor_raises(self):
        """scale() with a zero factor must raise ValueError."""
        bmd = _make_bmd()
        with pytest.raises(ValueError, match="strictly positive"):
            scale(bmd, 0.0, 1.0, 1.0)

    def test_negative_factor_raises(self):
        """scale() with a negative factor must raise ValueError."""
        bmd = _make_bmd()
        with pytest.raises(ValueError, match="strictly positive"):
            scale(bmd, 1.0, -2.0, 1.0)

    def test_nan_factor_raises(self):
        """scale() with a NaN factor must raise ValueError."""
        bmd = _make_bmd()
        with pytest.raises(ValueError, match="finite"):
            scale(bmd, float("nan"), 1.0, 1.0)
