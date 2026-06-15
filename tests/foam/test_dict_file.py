"""Tests for BlockMeshDict — merged from tests/common/test_block_mesh_dict.py
and tests/common/test_block_mesh_dict_api.py.

All imports use meshing_utils (stable public API gateway) or
meshing_utils.foam.dict_file for the new canonical location.
"""

from pathlib import Path

import pytest

from meshing_utils import (
    Block,
    BlockMeshDict,
    DefaultPatch,
    Edge,
    Face,
    Patch,
    Vertex,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "example_blockMeshDict"


# ===========================================================================
# Empty / direct construction
# ===========================================================================

def test_empty_construction():
    bmd = BlockMeshDict()
    assert len(bmd.vertices) == 0
    assert len(bmd.edges) == 0
    assert len(bmd.blocks) == 0
    assert len(bmd.boundary) == 0
    assert bmd.convertToMeters == 1.0


def test_nonexistent_path_yields_empty(tmp_path):
    bmd = BlockMeshDict(tmp_path / "does_not_exist")
    assert len(bmd.vertices) == 0


# ===========================================================================
# Loading the example file
# ===========================================================================

@pytest.fixture
def example_bmd():
    return BlockMeshDict(EXAMPLE)


def test_load_example_has_vertices(example_bmd):
    assert len(example_bmd.vertices) > 50


def test_load_example_has_edges(example_bmd):
    assert len(example_bmd.edges) > 0


def test_load_example_has_blocks(example_bmd):
    assert len(example_bmd.blocks) >= 110


def test_load_example_has_boundary(example_bmd):
    assert len(example_bmd.boundary) > 0


def test_load_example_convertToMeters(example_bmd):
    assert example_bmd.convertToMeters == pytest.approx(0.001)


def test_load_example_default_patch(example_bmd):
    assert example_bmd.default_patch.name == "connectors"
    assert example_bmd.default_patch.type == "empty"


def test_load_example_specific_vertex(example_bmd):
    v17 = example_bmd.vertices.get("v17")
    assert v17.coords == pytest.approx([0.0, 96.5, -34.75])


def test_load_example_specific_block(example_bmd):
    b0 = example_bmd.blocks.get("block0")
    assert b0.vertices == ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    assert b0.cells == [25, 25, 139]


def test_load_example_marker_on_v0(example_bmd):
    """The example file has `name v0 (...) //* f1 (1 2 3)` — marker must round-trip."""
    v0 = example_bmd.vertices.get("v0")
    assert v0.marker == "f1 (1 2 3)"


# ===========================================================================
# Marker extraction
# ===========================================================================

def test_get_marked_returns_v0(example_bmd):
    marked = example_bmd.get_marked()
    names = [m.name for m in marked if isinstance(m, Vertex)]
    assert "v0" in names


def test_get_marked_filter_by_type(example_bmd):
    only_vertices = example_bmd.get_marked(type_filter=Vertex)
    assert all(isinstance(m, Vertex) for m in only_vertices)


def test_get_marked_empty_when_no_markers():
    bmd = BlockMeshDict()
    bmd.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
    assert bmd.get_marked() == []


def test_get_marked_face():
    bmd = BlockMeshDict()
    f = Face(["v0", "v1", "v2", "v3"], marker="tag")
    bmd.boundary.add(Patch("p0", "patch", [f]))
    marked = bmd.get_marked()
    assert f in marked


# ===========================================================================
# Writing — basic structure
# ===========================================================================

def _populated_bmd() -> BlockMeshDict:
    bmd = BlockMeshDict()
    coords = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    for i, c in enumerate(coords):
        bmd.vertices.add(Vertex(f"v{i}", list(c)))
    bmd.blocks.add(Block("block0", [f"v{i}" for i in range(8)], [10, 10, 10]))
    bmd.default_patch = DefaultPatch("connectors", "empty")
    bmd.boundary.add(Patch("inlet", "patch", [Face(["v0", "v1", "v5", "v4"])]))
    return bmd


def test_write_produces_all_sections(tmp_path):
    bmd = _populated_bmd()
    out = tmp_path / "out"
    bmd.write(out)
    text = out.read_text()
    for keyword in ("convertToMeters", "geometry", "vertices", "edges",
                    "blocks", "defaultPatch", "boundary"):
        assert keyword in text


def test_write_face_block_ref_annotation(tmp_path):
    bmd = _populated_bmd()
    out = tmp_path / "out"
    bmd.write(out)
    text = out.read_text()
    assert "// block0" in text


def test_write_vertex_grouping_comment(tmp_path):
    bmd = _populated_bmd()
    out = tmp_path / "out"
    bmd.write(out)
    text = out.read_text()
    assert "// Vertices block0" in text


def test_write_unreferenced_vertices_at_end(tmp_path):
    bmd = _populated_bmd()
    bmd.vertices.add(Vertex("v_extra", [99.0, 99.0, 99.0]))
    out = tmp_path / "out"
    bmd.write(out)
    text = out.read_text()
    assert "Unreferenced" in text
    referenced_pos = text.index("name v0")
    unref_pos = text.index("name v_extra")
    assert unref_pos > referenced_pos


def test_write_face_marker_takes_precedence_over_block_ref(tmp_path):
    bmd = _populated_bmd()
    bmd.boundary[0].faces[0].marker = "tagX"
    out = tmp_path / "out"
    bmd.write(out)
    text = out.read_text()
    assert "//* tagX" in text


# ===========================================================================
# Roundtrip
# ===========================================================================

def test_roundtrip_preserves_counts(tmp_path):
    bmd = BlockMeshDict(EXAMPLE)
    out = tmp_path / "rt"
    bmd.write(out)
    bmd2 = BlockMeshDict(out)
    assert len(bmd2.vertices) == len(bmd.vertices)
    assert len(bmd2.edges) == len(bmd.edges)
    assert len(bmd2.blocks) == len(bmd.blocks)
    assert len(bmd2.boundary) == len(bmd.boundary)


def test_roundtrip_preserves_marker(tmp_path):
    bmd = BlockMeshDict(EXAMPLE)
    out = tmp_path / "rt"
    bmd.write(out)
    bmd2 = BlockMeshDict(out)
    assert bmd2.vertices.get("v0").marker == "f1 (1 2 3)"


def test_roundtrip_preserves_block_definition(tmp_path):
    bmd = BlockMeshDict(EXAMPLE)
    out = tmp_path / "rt"
    bmd.write(out)
    bmd2 = BlockMeshDict(out)
    b46_a = bmd.blocks.get("block46")
    b46_b = bmd2.blocks.get("block46")
    assert b46_a.vertices == b46_b.vertices
    assert b46_a.cells == b46_b.cells


def test_roundtrip_preserves_boundary_patch_names(tmp_path):
    bmd = BlockMeshDict(EXAMPLE)
    out = tmp_path / "rt"
    bmd.write(out)
    bmd2 = BlockMeshDict(out)
    names_a = [p.name for p in bmd.boundary]
    names_b = [p.name for p in bmd2.boundary]
    assert names_a == names_b


def test_roundtrip_preserves_convertToMeters(tmp_path):
    bmd = BlockMeshDict(EXAMPLE)
    out = tmp_path / "rt"
    bmd.write(out)
    bmd2 = BlockMeshDict(out)
    assert bmd2.convertToMeters == pytest.approx(bmd.convertToMeters)


# ===========================================================================
# API methods: _next_vertex_name
# ===========================================================================

class TestNextVertexName:

    def test_empty_vertices_returns_v0(self):
        bmd = BlockMeshDict()
        assert bmd._next_vertex_name() == "v0"

    def test_after_v0_returns_v1(self):
        bmd = BlockMeshDict()
        bmd.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        assert bmd._next_vertex_name() == "v1"

    def test_non_sequential_names_uses_max_plus_one(self):
        """Names v0, v3, v7 exist → next should be v8."""
        bmd = BlockMeshDict()
        for n in ("v0", "v3", "v7"):
            bmd.vertices.add(Vertex(n, [0.0, 0.0, 0.0]))
        assert bmd._next_vertex_name() == "v8"

    def test_non_v_prefixed_names_ignored(self):
        """Vertices without 'v<int>' names should not affect the counter."""
        bmd = BlockMeshDict()
        bmd.vertices.add(Vertex("custom_pt", [0.0, 0.0, 0.0]))
        assert bmd._next_vertex_name() == "v0"

    def test_mixed_names(self):
        """v0 and 'inlet' exist → next is v1."""
        bmd = BlockMeshDict()
        bmd.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd.vertices.add(Vertex("inlet", [1.0, 0.0, 0.0]))
        assert bmd._next_vertex_name() == "v1"


# ===========================================================================
# API methods: find_or_add_vertex
# ===========================================================================

class TestFindOrAddVertex:

    def test_happy_path_adds_new_vertex(self):
        bmd = BlockMeshDict()
        name = bmd.find_or_add_vertex((1.0, 2.0, 3.0), tol=1e-6)
        assert name == "v0"
        assert len(bmd.vertices) == 1
        assert bmd.vertices.get("v0").coords == pytest.approx([1.0, 2.0, 3.0])

    def test_exact_duplicate_returns_existing(self):
        bmd = BlockMeshDict()
        name1 = bmd.find_or_add_vertex((1.0, 2.0, 3.0), tol=1e-6)
        name2 = bmd.find_or_add_vertex((1.0, 2.0, 3.0), tol=1e-6)
        assert name1 == name2
        assert len(bmd.vertices) == 1

    def test_within_tolerance_returns_existing(self):
        bmd = BlockMeshDict()
        name1 = bmd.find_or_add_vertex((0.0, 0.0, 0.0), tol=0.01)
        name2 = bmd.find_or_add_vertex((0.005, 0.005, 0.005), tol=0.01)
        assert name1 == name2
        assert len(bmd.vertices) == 1

    def test_outside_tolerance_creates_new(self):
        bmd = BlockMeshDict()
        name1 = bmd.find_or_add_vertex((0.0, 0.0, 0.0), tol=0.01)
        name2 = bmd.find_or_add_vertex((0.02, 0.0, 0.0), tol=0.01)
        assert name1 != name2
        assert len(bmd.vertices) == 2

    def test_sequential_names_for_multiple_vertices(self):
        bmd = BlockMeshDict()
        names = [bmd.find_or_add_vertex((float(i), 0.0, 0.0), tol=1e-6) for i in range(3)]
        assert names == ["v0", "v1", "v2"]

    def test_tolerance_grid_snapping_deterministic(self):
        bmd = BlockMeshDict()
        name1 = bmd.find_or_add_vertex((0.4, 0.4, 0.4), tol=1.0)
        name2 = bmd.find_or_add_vertex((0.4, 0.4, 0.4), tol=1.0)
        assert name1 == name2

    def test_different_grid_cells_create_different_vertices(self):
        bmd = BlockMeshDict()
        name1 = bmd.find_or_add_vertex((0.4, 0.0, 0.0), tol=1.0)
        name2 = bmd.find_or_add_vertex((1.6, 0.0, 0.0), tol=1.0)
        assert name1 != name2

    def test_existing_v_prefix_vertices_not_overwritten(self):
        """Pre-existing v0 … v2 → find_or_add for new coord yields v3."""
        bmd = BlockMeshDict()
        for i in range(3):
            bmd.vertices.add(Vertex(f"v{i}", [float(i) * 10, 0.0, 0.0]))
        name = bmd.find_or_add_vertex((999.0, 0.0, 0.0), tol=1e-6)
        assert name == "v3"


# ===========================================================================
# API methods: has_block_with_vertex_set
# ===========================================================================

def _make_unit_cube_bmd() -> BlockMeshDict:
    """Return a BlockMeshDict with 8 vertices forming a unit cube and one block."""
    bmd = BlockMeshDict()
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for i, c in enumerate(coords):
        bmd.vertices.add(Vertex(f"v{i}", list(c)))
    bmd.blocks.add(
        Block("block0", [f"v{i}" for i in range(8)], [1, 1, 1])
    )
    return bmd


class TestHasBlockWithVertexSet:

    def test_exact_match_returns_true(self):
        bmd = _make_unit_cube_bmd()
        names = [f"v{i}" for i in range(8)]
        assert bmd.has_block_with_vertex_set(names) is True

    def test_disjoint_set_returns_false(self):
        bmd = _make_unit_cube_bmd()
        assert bmd.has_block_with_vertex_set(
            ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
        ) is False

    def test_subset_returns_false(self):
        bmd = _make_unit_cube_bmd()
        assert bmd.has_block_with_vertex_set(["v0", "v1", "v2", "v3"]) is False

    def test_superset_returns_false(self):
        bmd = _make_unit_cube_bmd()
        names = [f"v{i}" for i in range(8)] + ["v_extra"]
        assert bmd.has_block_with_vertex_set(names) is False

    def test_order_independent(self):
        bmd = _make_unit_cube_bmd()
        shuffled = ["v7", "v6", "v5", "v4", "v3", "v2", "v1", "v0"]
        assert bmd.has_block_with_vertex_set(shuffled) is True

    def test_empty_bmd_returns_false(self):
        bmd = BlockMeshDict()
        assert bmd.has_block_with_vertex_set(["v0", "v1"]) is False


# ===========================================================================
# API methods: find_edge
# ===========================================================================

class TestFindEdge:

    def _bmd_with_edge(self) -> BlockMeshDict:
        bmd = BlockMeshDict()
        e = Edge("arc", "v0", "v1", coords=[0.5, 0.5, 0.0])
        bmd.edges.add(e)
        return bmd

    def test_find_edge_forward(self):
        bmd = self._bmd_with_edge()
        result = bmd.find_edge("v0", "v1")
        assert result is not None
        assert result.type == "arc"

    def test_find_edge_reverse(self):
        bmd = self._bmd_with_edge()
        result = bmd.find_edge("v1", "v0")
        assert result is not None
        assert result.type == "arc"

    def test_find_edge_missing_returns_none(self):
        bmd = self._bmd_with_edge()
        assert bmd.find_edge("v2", "v3") is None

    def test_find_edge_empty_edges(self):
        bmd = BlockMeshDict()
        assert bmd.find_edge("v0", "v1") is None

    def test_find_edge_multiple_edges_correct_one(self):
        bmd = BlockMeshDict()
        bmd.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.0, 0.0]))
        bmd.edges.add(Edge("bspline", "v2", "v3", coords=[0.0, 0.5, 0.0]))
        result = bmd.find_edge("v2", "v3")
        assert result is not None
        assert result.type == "bspline"


# ===========================================================================
# API methods: next_block_index
# ===========================================================================

class TestNextBlockIndex:

    def test_empty_returns_zero(self):
        bmd = BlockMeshDict()
        assert bmd.next_block_index() == 0

    def test_one_block_returns_one(self):
        bmd = BlockMeshDict()
        bmd.blocks.add(Block("block0", ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]))
        assert bmd.next_block_index() == 1

    def test_sequential_blocks(self):
        bmd = BlockMeshDict()
        for i in range(5):
            bmd.blocks.add(Block(f"block{i}", [f"v{j}" for j in range(8)]))
        assert bmd.next_block_index() == 5

    def test_with_gap_returns_max_plus_one(self):
        bmd = BlockMeshDict()
        for n in ("block0", "block3", "block5"):
            bmd.blocks.add(Block(n, [f"v{j}" for j in range(8)]))
        assert bmd.next_block_index() == 6

    def test_unnamed_blocks_ignored(self):
        bmd = BlockMeshDict()
        bmd.blocks.add(Block("", [f"v{j}" for j in range(8)]))
        assert bmd.next_block_index() == 0

    def test_non_block_prefix_names_ignored(self):
        bmd = BlockMeshDict()
        bmd.blocks.add(Block("custom_block", [f"v{j}" for j in range(8)]))
        assert bmd.next_block_index() == 0


# ===========================================================================
# API methods: has_block_named
# ===========================================================================

class TestHasBlockNamed:

    def test_present_returns_true(self):
        bmd = BlockMeshDict()
        bmd.blocks.add(Block("block0", [f"v{i}" for i in range(8)]))
        assert bmd.has_block_named("block0") is True

    def test_absent_returns_false(self):
        bmd = BlockMeshDict()
        assert bmd.has_block_named("block0") is False

    def test_empty_bmd_returns_false(self):
        bmd = BlockMeshDict()
        assert bmd.has_block_named("anything") is False

    def test_similar_name_returns_false(self):
        bmd = BlockMeshDict()
        bmd.blocks.add(Block("block0", [f"v{i}" for i in range(8)]))
        assert bmd.has_block_named("block1") is False


# ===========================================================================
# default_patch name default
# ===========================================================================

class TestDefaultPatchNameDefault:

    def test_empty_construction_has_defaultFaces(self):
        bmd = BlockMeshDict()
        assert bmd.default_patch.name == "defaultFaces"

    def test_to_foam_string_contains_name_defaultFaces(self):
        bmd = BlockMeshDict()
        output = bmd.default_patch.to_foam_string()
        assert "name defaultFaces;" in output

    def test_loaded_file_preserves_name(self):
        if not EXAMPLE.exists():
            pytest.skip("example_blockMeshDict not found")
        bmd = BlockMeshDict(EXAMPLE)
        assert bmd.default_patch.name == "connectors"
