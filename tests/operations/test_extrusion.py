"""Unit tests for meshing_utils.operations.extrusion."""

import pytest

from meshing_utils import (
    Block,
    BlockMeshDict,
    Edge,
    Face,
    OrderingConsistencyError,
    Patch,
    Vertex,
    assert_hex_outward_from_coords,
)
from meshing_utils.operations.extrusion import (
    EXTRUSION_LOCAL_FACE_INDICES,
    AmbiguousFaceError,
    LayerStep,
    NoMarkersFoundError,
    NonCoplanarVerticesError,
    ParseError,
    _check_coplanar,
    cumulative_offsets,
    extrude,
    extrude_with_steps,
    parse_layer_steps,
    parse_offsets,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal BlockMeshDict objects in memory
# ---------------------------------------------------------------------------

def _make_bmd_with_marked_top_face() -> BlockMeshDict:
    """Return a BlockMeshDict with one hex block whose top 4 vertices are marked.

    Block layout (standard OpenFOAM hex, k-axis points up):
      v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)  <- bottom (face 0)
      v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)  <- top    (face 1)

    Vertices v4..v7 carry a //* marker.
    Block carries a //* marker.
    """
    bmd = BlockMeshDict()
    for name, coords in [
        ("v0", [0, 0, 0]),
        ("v1", [1, 0, 0]),
        ("v2", [1, 1, 0]),
        ("v3", [0, 1, 0]),
    ]:
        bmd.vertices.add(Vertex(name, coords))
    for name, coords in [
        ("v4", [0, 0, 1]),
        ("v5", [1, 0, 1]),
        ("v6", [1, 1, 1]),
        ("v7", [0, 1, 1]),
    ]:
        bmd.vertices.add(Vertex(name, coords, marker="top"))

    bmd.blocks.add(
        Block(
            "b0",
            vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            cells=[5, 6, 7],
            grading_type="simpleGrading",
            grading_def=[1.0, 1.0, 1.0],
            marker="extrudeMe",
        )
    )
    return bmd


def _make_bmd_vertex_only() -> BlockMeshDict:
    """Return a BlockMeshDict with one hex block whose top 4 vertices are marked
    but the block itself is NOT marked (VERTEX_ONLY discovery mode)."""
    bmd = BlockMeshDict()
    for name, coords in [
        ("v0", [0, 0, 0]),
        ("v1", [1, 0, 0]),
        ("v2", [1, 1, 0]),
        ("v3", [0, 1, 0]),
    ]:
        bmd.vertices.add(Vertex(name, coords))
    for name, coords in [
        ("v4", [0, 0, 1]),
        ("v5", [1, 0, 1]),
        ("v6", [1, 1, 1]),
        ("v7", [0, 1, 1]),
    ]:
        bmd.vertices.add(Vertex(name, coords, marker=""))  # //* ohne Label

    bmd.blocks.add(
        Block(
            "b0",
            vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            cells=[1, 1, 1],
            grading_type="simpleGrading",
            grading_def=[1.0, 1.0, 1.0],
            marker=None,  # Block NICHT markiert
        )
    )
    return bmd


# ===========================================================================
# parse_offsets — happy path
# ===========================================================================

def test_parse_offsets_single():
    result = parse_offsets("((0 0 0.5))")
    assert result == [(0.0, 0.0, 0.5)]


def test_parse_offsets_multiple():
    result = parse_offsets("((1 2 3) (4 5 6))")
    assert result == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_parse_offsets_negative_values():
    result = parse_offsets("((-1 -2 -3))")
    assert result == [(-1.0, -2.0, -3.0)]


def test_parse_offsets_scientific_notation():
    result = parse_offsets("((1e-2 2e-2 3e-2))")
    assert pytest.approx(result[0][0]) == 0.01


# ===========================================================================
# parse_offsets — error cases
# ===========================================================================

def test_parse_offsets_invalid_format():
    with pytest.raises(ValueError):
        parse_offsets("0 0 0.5")


def test_parse_offsets_empty():
    with pytest.raises(ValueError):
        parse_offsets("()")


def test_parse_offsets_zero_offset_raises():
    with pytest.raises(ValueError, match="Zero-length"):
        parse_offsets("((0 0 0))")


def test_parse_offsets_missing_outer_parens():
    with pytest.raises(ValueError):
        parse_offsets("(0 0 0.5)")


# ===========================================================================
# cumulative_offsets
# ===========================================================================

def test_cumulative_offsets_single():
    result = cumulative_offsets([(0.0, 0.0, 0.2)])
    assert result == [(0.0, 0.0, 0.2)]


def test_cumulative_offsets_multiple():
    result = cumulative_offsets([(0.0, 0.0, 0.2), (0.0, 0.0, 0.6)])
    assert result[0] == pytest.approx((0.0, 0.0, 0.2))
    assert result[1] == pytest.approx((0.0, 0.0, 0.8))


def test_cumulative_offsets_three_layers():
    result = cumulative_offsets([(0, 0, 1), (0, 0, 2), (0, 0, 3)])
    assert result[2] == pytest.approx((0, 0, 6))


# ===========================================================================
# Planarity check
# ===========================================================================

def test_check_coplanar_four_coplanar_vertices():
    vertices = [
        Vertex("a", [0, 0, 1]),
        Vertex("b", [1, 0, 1]),
        Vertex("c", [1, 1, 1]),
        Vertex("d", [0, 1, 1]),
    ]
    _check_coplanar(vertices)  # must not raise


def test_non_planar_vertices_raises():
    vertices = [
        Vertex("a", [0, 0, 0]),
        Vertex("b", [1, 0, 0]),
        Vertex("c", [1, 1, 0]),
        Vertex("d", [0, 1, 0.5]),  # lifted
    ]
    with pytest.raises(NonCoplanarVerticesError):
        _check_coplanar(vertices)


def test_collinear_vertices_raises():
    vertices = [
        Vertex("a", [0, 0, 0]),
        Vertex("b", [1, 0, 0]),
        Vertex("c", [2, 0, 0]),
    ]
    with pytest.raises(NonCoplanarVerticesError):
        _check_coplanar(vertices)


# ===========================================================================
# extrude — single block, single layer (BLOCK_DRIVEN)
# ===========================================================================

def test_extrude_single_block_single_layer():
    bmd = _make_bmd_with_marked_top_face()
    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    # 4 new vertices added (layer 1)
    assert len(result.vertices) == 12
    assert "v4e1" in result.vertices
    assert "v7e1" in result.vertices

    # 1 new block added
    assert len(result.blocks) == 2

    new_block = result.blocks[1]
    # New block must reference the top (extruded) face as its top
    assert "v4e1" in new_block.vertices
    assert "v7e1" in new_block.vertices


def test_extrude_two_layers():
    bmd = _make_bmd_with_marked_top_face()
    result = extrude(bmd, [(0.0, 0.0, 0.2), (0.0, 0.0, 0.6)])

    assert len(result.vertices) == 16  # 8 original + 4*2 new
    assert "v4e1" in result.vertices
    assert "v4e2" in result.vertices

    # Cumulative: layer 1 at z=1+0.2=1.2, layer 2 at z=1+0.8=1.8
    v4e1 = result.vertices.get("v4e1")
    assert v4e1.coords[2] == pytest.approx(1.2)

    v4e2 = result.vertices.get("v4e2")
    assert v4e2.coords[2] == pytest.approx(1.8)

    assert len(result.blocks) == 3  # 1 original + 2 new


# ===========================================================================
# VERTEX_ONLY mode
# ===========================================================================

def test_vertex_only_mode():
    bmd = _make_bmd_vertex_only()
    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    # 1 Original + 1 Extrusion
    assert len(result.blocks) == 2
    new_block = result.blocks[1]
    assert new_block.cells == [1, 1, 1]
    assert new_block.grading_def == [1.0, 1.0, 1.0]


def test_vertex_only_three_vertices_raises():
    bmd = BlockMeshDict()
    for name, coords in [("a", [0, 0, 1]), ("b", [1, 0, 1]), ("c", [0, 1, 1])]:
        bmd.vertices.add(Vertex(name, coords, marker=""))
    # Kein Block → kein Face → NoMarkersFoundError
    with pytest.raises(NoMarkersFoundError):
        extrude(bmd, [(0.0, 0.0, 0.5)])


# ===========================================================================
# Cell counts taken from source block
# ===========================================================================

def test_cells_taken_from_source_block():
    bmd = _make_bmd_with_marked_top_face()
    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    new_block = result.blocks[1]
    # Source block cells = [5, 6, 7], extrusion along k (axis 2) -> new = [5, 6, 1]
    assert new_block.cells == [5, 6, 1]


# ===========================================================================
# Edges copied per layer
# ===========================================================================

def test_edges_copied_per_layer():
    bmd = _make_bmd_with_marked_top_face()
    # Add an arc edge between two marked vertices
    bmd.edges.add(
        Edge("arc", "v4", "v5", points=[[0.5, 0.0, 1.1]])
    )

    result = extrude(bmd, [(0.0, 0.0, 0.5), (0.0, 0.0, 0.5)])

    # Should have 2 new edges: one per layer
    edge_keys = {(e.v_start, e.v_end) for e in result.edges}
    assert ("v4e1", "v5e1") in edge_keys
    assert ("v4e2", "v5e2") in edge_keys


def test_edge_control_point_shifted():
    bmd = _make_bmd_with_marked_top_face()
    bmd.edges.add(Edge("arc", "v4", "v5", points=[[0.5, 0.0, 1.1]]))

    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    new_edge = result.edges.get("v4e1", "v5e1")
    assert new_edge.points[0][2] == pytest.approx(1.6)  # 1.1 + 0.5


# ===========================================================================
# Boundary patch update
# ===========================================================================

def test_boundary_patch_updated():
    bmd = _make_bmd_with_marked_top_face()
    patch = Patch("top_patch", type="wall", faces=[
        Face(["v4", "v5", "v6", "v7"])
    ])
    bmd.boundary.add(patch)

    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    top_patch = result.boundary.get("top_patch")
    face_verts = top_patch.faces[0].vertices
    # Face must now reference layer-1 vertices
    assert set(face_verts) == {"v4e1", "v5e1", "v6e1", "v7e1"}


def test_boundary_patch_non_marked_face_unchanged():
    bmd = _make_bmd_with_marked_top_face()
    patch = Patch("bottom_patch", type="wall", faces=[
        Face(["v0", "v1", "v2", "v3"])  # bottom, not marked
    ])
    bmd.boundary.add(patch)

    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    bottom_patch = result.boundary.get("bottom_patch")
    face_verts = bottom_patch.faces[0].vertices
    # Not marked -> should be unchanged
    assert set(face_verts) == {"v0", "v1", "v2", "v3"}


# ===========================================================================
# Marker stripping
# ===========================================================================

def test_markers_stripped():
    bmd = _make_bmd_with_marked_top_face()
    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    for v in result.vertices:
        assert v.marker is None, f"Vertex {v.name} still has marker: {v.marker!r}"
    for b in result.blocks:
        assert b.marker is None, f"Block {b.name} still has marker: {b.marker!r}"


# ===========================================================================
# No markers found
# ===========================================================================

def test_no_markers_raises():
    bmd = BlockMeshDict()
    bmd.vertices.add(Vertex("v0", [0, 0, 0]))
    with pytest.raises(NoMarkersFoundError):
        extrude(bmd, [(0.0, 0.0, 1.0)])


# ===========================================================================
# AmbiguousFaceError
# ===========================================================================

def test_block_no_matching_face_raises():
    bmd = BlockMeshDict()
    # Vertices: only 3 marked, not a full face
    for name, coords in [
        ("v0", [0, 0, 0]),
        ("v1", [1, 0, 0]),
        ("v2", [1, 1, 0]),
        ("v3", [0, 1, 0]),
    ]:
        bmd.vertices.add(Vertex(name, coords))
    for name, coords in [
        ("v4", [0, 0, 1]),
        ("v5", [1, 0, 1]),
        ("v6", [1, 1, 1]),
    ]:
        # Only 3 of the 4 top-face vertices are marked
        bmd.vertices.add(Vertex(name, coords, marker="top"))
    # v7 is NOT marked
    bmd.vertices.add(Vertex("v7", [0, 1, 1]))

    bmd.blocks.add(
        Block(
            "b0",
            vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            cells=[1, 1, 1],
            grading_type="simpleGrading",
            grading_def=[1.0, 1.0, 1.0],
            marker="extrudeMe",
        )
    )

    with pytest.raises((AmbiguousFaceError, NonCoplanarVerticesError)):
        extrude(bmd, [(0.0, 0.0, 0.5)])


# ===========================================================================
# VERTEX_ONLY mode — new tests
# ===========================================================================

def test_vertex_only_discovers_correct_face():
    bmd = _make_bmd_vertex_only()
    result = extrude(bmd, [(0.0, 0.0, 0.5)])

    new_block = result.blocks[1]
    assert "v4" in new_block.vertices
    assert "v7" in new_block.vertices
    assert "v4e1" in new_block.vertices
    assert "v7e1" in new_block.vertices


def test_vertex_only_cells_from_source_block():
    bmd = BlockMeshDict()
    for name, coords in [
        ("v0", [0, 0, 0]), ("v1", [1, 0, 0]),
        ("v2", [1, 1, 0]), ("v3", [0, 1, 0]),
    ]:
        bmd.vertices.add(Vertex(name, coords))
    for name, coords in [
        ("v4", [0, 0, 1]), ("v5", [1, 0, 1]),
        ("v6", [1, 1, 1]), ("v7", [0, 1, 1]),
    ]:
        bmd.vertices.add(Vertex(name, coords, marker=""))
    bmd.blocks.add(Block(
        "b0",
        vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        cells=[5, 6, 7],
        grading_type="simpleGrading",
        grading_def=[1.0, 2.0, 3.0],
        marker=None,
    ))
    result = extrude(bmd, [(0.0, 0.0, 0.5)])
    new_block = result.blocks[1]
    # Extrusion along k-axis (axis 2): cells[2] -> 1
    assert new_block.cells == [5, 6, 1]


def test_vertex_only_unused_marked_vertices_does_not_raise():
    bmd = _make_bmd_vertex_only()
    # Extra-Vertex: markiert, koplanar mit den anderen (z=1), aber kein Face-Eckpunkt
    bmd.vertices.add(Vertex("v_extra", [0.5, 0.5, 1.0], marker=""))
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = extrude(bmd, [(0.0, 0.0, 0.5)])
    assert len(result.blocks) == 2  # Extra-Vertex erzeugt keinen neuen Block
    assert any("v_extra" in str(warning.message) for warning in w)


def test_vertex_only_marker_without_label():
    """Exact scenario from example_blockMeshDict: //* without label."""
    bmd = _make_bmd_vertex_only()  # marker="" fuer v4..v7
    result = extrude(bmd, [(0.0, 0.0, 0.5)])
    assert len(result.blocks) == 2  # kein NoMarkersFoundError


def test_no_markers_raises_with_helpful_message():
    bmd = BlockMeshDict()
    bmd.vertices.add(Vertex("v0", [0, 0, 0]))
    with pytest.raises(NoMarkersFoundError, match=r"/\*"):
        extrude(bmd, [(0.0, 0.0, 1.0)])


def test_block_driven_takes_priority_over_vertex_only():
    """When both vertices and blocks are marked, BLOCK_DRIVEN mode is used."""
    bmd = _make_bmd_with_marked_top_face()  # Block b0 ist markiert
    result = extrude(bmd, [(0.0, 0.0, 0.5)])
    # Identisches Verhalten wie bestehende BLOCK_DRIVEN-Tests
    assert len(result.blocks) == 2
    new_block = result.blocks[1]
    assert "v4e1" in new_block.vertices


# ===========================================================================
# Right-handedness for all 6 face orientations
# ===========================================================================

# Standard unit hex: vertex names indexed as v0..v7 in OpenFOAM order.
_UNIT_HEX_VERTICES = [
    ("v0", [0.0, 0.0, 0.0]),
    ("v1", [1.0, 0.0, 0.0]),
    ("v2", [1.0, 1.0, 0.0]),
    ("v3", [0.0, 1.0, 0.0]),
    ("v4", [0.0, 0.0, 1.0]),
    ("v5", [1.0, 0.0, 1.0]),
    ("v6", [1.0, 1.0, 1.0]),
    ("v7", [0.0, 1.0, 1.0]),
]

# (face label, marked local indices, outward offset)
_FACE_CASES = [
    ("face0_-k", [0, 1, 2, 3], (0.0, 0.0, -0.5)),
    ("face1_+k", [4, 5, 6, 7], (0.0, 0.0, 0.5)),
    ("face2_-j", [0, 1, 5, 4], (0.0, -0.5, 0.0)),
    ("face3_+j", [2, 3, 7, 6], (0.0, 0.5, 0.0)),
    ("face4_+i", [1, 2, 6, 5], (0.5, 0.0, 0.0)),
    ("face5_-i", [0, 3, 7, 4], (-0.5, 0.0, 0.0)),
]


def _build_unit_hex_bmd(marked_local_indices, src_cells=(2, 3, 5),
                        src_grading=(1.5, 2.5, 3.5)) -> BlockMeshDict:
    """Build a BMD with one unit hex; the listed local vertex indices carry a
    marker. Block also carries a marker (BLOCK_DRIVEN mode)."""
    bmd = BlockMeshDict()
    marked = set(marked_local_indices)
    for idx, (name, coords) in enumerate(_UNIT_HEX_VERTICES):
        marker = "f" if idx in marked else None
        bmd.vertices.add(Vertex(name, coords, marker=marker))
    bmd.blocks.add(Block(
        "b0",
        vertices=[name for name, _ in _UNIT_HEX_VERTICES],
        cells=list(src_cells),
        grading_type="simpleGrading",
        grading_def=list(src_grading),
        marker="extrudeMe",
    ))
    return bmd


def _hex_signed_volume(block, vertex_lookup):
    """Return (v1-v0) x (v2-v1) . (v4-v0). Positive => right-handed hex."""
    coords = [vertex_lookup(name).coords for name in block.vertices]
    def sub(a, b): return [a[i] - b[i] for i in range(3)]
    def cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    def dot(a, b): return sum(a[i] * b[i] for i in range(3))
    return dot(cross(sub(coords[1], coords[0]), sub(coords[2], coords[1])),
               sub(coords[4], coords[0]))


@pytest.mark.parametrize(
    "label,marked,offset",
    _FACE_CASES,
    ids=[c[0] for c in _FACE_CASES],
)
def test_extruded_block_is_right_handed(label, marked, offset):
    bmd = _build_unit_hex_bmd(marked)
    result = extrude(bmd, [offset])

    new_block = result.blocks[1]
    sv = _hex_signed_volume(new_block, result.vertices.get)
    assert sv > 0, (
        f"Extruded block for {label} is inside-out "
        f"(signed volume = {sv})."
    )


# ===========================================================================
# Cells / grading axis mapping for non-trivial faces
# ===========================================================================

@pytest.mark.parametrize(
    "label,marked,offset,expected_cells,expected_grading",
    [
        # face 0: x1'=j, x2'=i  -> [nj, ni, 1], [gj, gi, 1]
        ("face0", [0, 1, 2, 3], (0.0, 0.0, -0.5),
         [3, 2, 1], [2.5, 1.5, 1.0]),
        # face 1: x1'=i, x2'=j
        ("face1", [4, 5, 6, 7], (0.0, 0.0, 0.5),
         [2, 3, 1], [1.5, 2.5, 1.0]),
        # face 2: x1'=i, x2'=k
        ("face2", [0, 1, 5, 4], (0.0, -0.5, 0.0),
         [2, 5, 1], [1.5, 3.5, 1.0]),
        # face 3: x1'=i, x2'=k (after (2,3,7,6) ordering: 2->3 is -i, axis i)
        ("face3", [2, 3, 7, 6], (0.0, 0.5, 0.0),
         [2, 5, 1], [1.5, 3.5, 1.0]),
        # face 4: x1'=j, x2'=k
        ("face4", [1, 2, 6, 5], (0.5, 0.0, 0.0),
         [3, 5, 1], [2.5, 3.5, 1.0]),
        # face 5: x1'=k, x2'=j (EXTRUSION_LOCAL_FACE_INDICES = (0,4,7,3))
        ("face5", [0, 3, 7, 4], (-0.5, 0.0, 0.0),
         [5, 3, 1], [3.5, 2.5, 1.0]),
    ],
)
def test_cells_and_grading_mapping_per_face(
    label, marked, offset, expected_cells, expected_grading
):
    bmd = _build_unit_hex_bmd(marked,
                              src_cells=(2, 3, 5),
                              src_grading=(1.5, 2.5, 3.5))
    result = extrude(bmd, [offset])
    new_block = result.blocks[1]
    assert new_block.cells == expected_cells, label
    assert new_block.grading_def == expected_grading, label


# ===========================================================================
# parse_layer_steps — Parser tests P1-P9
# ===========================================================================

def test_P1_single_normal_token():
    """P1: ((0 0 1)) -> 1 LayerStep, skip=(0,0,0), delta=(0,0,1)."""
    steps = parse_layer_steps("((0 0 1))")
    assert len(steps) == 1
    assert steps[0].skip_offset == pytest.approx((0.0, 0.0, 0.0))
    assert steps[0].block_delta == pytest.approx((0.0, 0.0, 1.0))


def test_P2_skip_then_normal():
    """P2: ([0 0 0.5] (0 0 1)) -> 1 LayerStep, skip=(0,0,0.5), delta=(0,0,1)."""
    steps = parse_layer_steps("([0 0 0.5] (0 0 1))")
    assert len(steps) == 1
    assert steps[0].skip_offset == pytest.approx((0.0, 0.0, 0.5))
    assert steps[0].block_delta == pytest.approx((0.0, 0.0, 1.0))


def test_P3_normal_skip_normal():
    """P3: ((0 0 0.5) [0 0 0.2] (0 0 0.5)) -> 2 LayerSteps."""
    steps = parse_layer_steps("((0 0 0.5) [0 0 0.2] (0 0 0.5))")
    assert len(steps) == 2
    # First step: no skip, delta=(0,0,0.5)
    assert steps[0].skip_offset == pytest.approx((0.0, 0.0, 0.0))
    assert steps[0].block_delta == pytest.approx((0.0, 0.0, 0.5))
    # Second step: skip=(0,0,0.2), delta=(0,0,0.5)
    assert steps[1].skip_offset == pytest.approx((0.0, 0.0, 0.2))
    assert steps[1].block_delta == pytest.approx((0.0, 0.0, 0.5))


def test_P4_consecutive_skips_aggregated():
    """P4: ([0 0 0.3] [0 0 0.6] (0 0 1)) -> 1 LayerStep, skip=(0,0,0.9), delta=(0,0,1)."""
    steps = parse_layer_steps("([0 0 0.3] [0 0 0.6] (0 0 1))")
    assert len(steps) == 1
    assert steps[0].skip_offset == pytest.approx((0.0, 0.0, 0.9))
    assert steps[0].block_delta == pytest.approx((0.0, 0.0, 1.0))


def test_P5_trailing_skip_raises():
    """P5: ((0 0 1) [0 0 0.5]) -> ParseError (last token must be NORMAL)."""
    with pytest.raises(ParseError):
        parse_layer_steps("((0 0 1) [0 0 0.5])")


def test_P6_only_skip_raises():
    """P6: ([0 0 1]) -> ParseError (pure SKIP, no NORMAL token)."""
    with pytest.raises(ParseError):
        parse_layer_steps("([0 0 1])")


def test_P7_skip_with_xyz_components():
    """P7: ([0.1 0.2 0.3] (1 1 1)) -> skip=(0.1,0.2,0.3), delta=(1,1,1)."""
    steps = parse_layer_steps("([0.1 0.2 0.3] (1 1 1))")
    assert len(steps) == 1
    assert steps[0].skip_offset == pytest.approx((0.1, 0.2, 0.3))
    assert steps[0].block_delta == pytest.approx((1.0, 1.0, 1.0))


def test_P8_empty_string_raises():
    """P8a: "" -> ParseError."""
    with pytest.raises(ParseError):
        parse_layer_steps("")


def test_P8_empty_parens_raises():
    """P8b: () -> ParseError."""
    with pytest.raises(ParseError):
        parse_layer_steps("()")


def test_P9_whitespace_robust():
    """P9: Whitespace-robust parsing."""
    steps = parse_layer_steps("(  [  0  0  0.5  ]   ( 0  0  1 )  )")
    assert len(steps) == 1
    assert steps[0].skip_offset == pytest.approx((0.0, 0.0, 0.5))
    assert steps[0].block_delta == pytest.approx((0.0, 0.0, 1.0))


# ===========================================================================
# Block-Generierungs-Tests B1-B9
# ===========================================================================

def test_B1_happy_path_no_skips():
    """B1: Happy Path ohne Skips (Regression) — extrude_with_steps verhaelt sich
    wie extrude bei reinen NORMAL-Vektoren."""
    bmd = _make_bmd_with_marked_top_face()
    steps = [
        LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5)),
        LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5)),
    ]
    result = extrude_with_steps(bmd, steps)
    # 8 original + 4*2 new vertices
    assert len(result.vertices) == 16
    # 1 original + 2 new blocks
    assert len(result.blocks) == 3


def test_B2_first_vector_is_skip():
    """B2: Erster Vektor ist Skip -> 1 Block, Bottom nicht auf Originalebene."""
    bmd = _make_bmd_with_marked_top_face()
    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.5), block_delta=(0.0, 0.0, 1.0))]
    result = extrude_with_steps(bmd, steps)

    # Only 1 new block
    assert len(result.blocks) == 2

    new_block = result.blocks[1]
    # Bottom of block is at z = 1 + 0.5 = 1.5, NOT at z = 1 (original)
    # Layer 1 = skip bottom at z=1.5, Layer 2 = top at z=2.5
    # The source vertices (z=1) should NOT appear in the new block's vertices
    # as they are skipped.
    for vname in new_block.vertices:
        assert vname not in {"v4", "v5", "v6", "v7"}, (
            f"Original source vertex {vname!r} should not be in skip-starting block."
        )


def test_B2_original_plane_not_in_vertex_list_when_skipped():
    """B2 (vertex check): Originalebene nicht in Vertex-Liste bei erstem Skip."""
    bmd = _make_bmd_with_marked_top_face()
    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.5), block_delta=(0.0, 0.0, 1.0))]
    result = extrude_with_steps(bmd, steps)

    # v4..v7 are the marked source vertices. They must NOT appear as extruded
    # intermediate vertices (no "v4e0" etc.) — only the actual block levels exist.
    {v.name for v in result.vertices}
    # v4..v7 themselves still exist (original block bottom), but should not have
    # been used as a block bottom/top in the extruded geometry.
    # What must NOT exist: a layer at the source offset (0,0,0), i.e. "v4e1" at z=1
    # when the first step has a skip. Instead, layer indices should skip the origin.
    # Layer 1 corresponds to z=1.5, layer 2 to z=2.5.
    v4e1 = result.vertices.get("v4e1")
    # v4e1 should be at z = 1 + 0.5 = 1.5 (skip bottom), NOT at z=1.5...
    # Actually the layer index assignment: skip bottom at z=1.5 gets index 1,
    # block top at z=2.5 gets index 2.
    assert v4e1 is not None
    assert v4e1.coords[2] == pytest.approx(1.5)


def test_B3_skip_in_middle():
    """B3: Skip in der Mitte -> N-1 Bloecke statt N."""
    bmd = _make_bmd_with_marked_top_face()
    steps = [
        LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5)),
        LayerStep(skip_offset=(0.0, 0.0, 0.3), block_delta=(0.0, 0.0, 0.5)),
    ]
    result = extrude_with_steps(bmd, steps)
    # 1 original + 2 new extrusion blocks
    assert len(result.blocks) == 3


def test_B4_consecutive_skips_structurally_identical_to_sum():
    """B4: Konsekutive Skips aggregiert -> strukturell identisch zu Summen-Skip."""
    bmd_a = _make_bmd_with_marked_top_face()
    bmd_b = _make_bmd_with_marked_top_face()

    [
        LayerStep(skip_offset=(0.0, 0.0, 0.3), block_delta=(0.0, 0.0, 0.0)),
    ]
    # Use a valid scenario: two consecutive SKIPs then a block
    parsed_two = parse_layer_steps("([0 0 0.3] [0 0 0.6] (0 0 1))")
    parsed_one = parse_layer_steps("([0 0 0.9] (0 0 1))")

    result_two = extrude_with_steps(bmd_a, parsed_two)
    result_one = extrude_with_steps(bmd_b, parsed_one)

    # Both should produce the same number of blocks and vertices
    assert len(result_two.blocks) == len(result_one.blocks)
    assert len(result_two.vertices) == len(result_one.vertices)

    # The extruded vertex at the block bottom should be at the same z
    # (z_source + 0.9 = 1 + 0.9 = 1.9)
    result_two.blocks[1]
    result_one.blocks[1]
    # Both should have the same bottom vertex z-coordinate
    # Bottom layer is layer 1 for both
    v4_two = result_two.vertices.get("v4e1")
    v4_one = result_one.vertices.get("v4e1")
    assert v4_two.coords[2] == pytest.approx(v4_one.coords[2])


def test_B5_bottom_patch_on_first_produced_layer():
    """B5: Bottom-Patch auf erstem erzeugten Layer (nicht Originalebene bei Skip)."""
    bmd = _make_bmd_with_marked_top_face()
    patch = Patch("top_patch", type="wall", faces=[
        Face(["v4", "v5", "v6", "v7"])
    ])
    bmd.boundary.add(patch)

    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5))]
    result = extrude_with_steps(bmd, steps)

    top_patch = result.boundary.get("top_patch")
    face_verts = top_patch.faces[0].vertices
    # Top patch should reference the top layer (layer 1)
    assert set(face_verts) == {"v4e1", "v5e1", "v6e1", "v7e1"}


def test_B6_top_patch_on_uppermost_layer():
    """B6: Top-Patch auf oberstem erzeugten Layer."""
    bmd = _make_bmd_with_marked_top_face()
    patch = Patch("top_patch", type="wall", faces=[
        Face(["v4", "v5", "v6", "v7"])
    ])
    bmd.boundary.add(patch)

    steps = [
        LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5)),
        LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=(0.0, 0.0, 0.5)),
    ]
    result = extrude_with_steps(bmd, steps)

    top_patch = result.boundary.get("top_patch")
    face_verts = top_patch.faces[0].vertices
    # Top patch should reference the topmost layer (layer 2)
    assert set(face_verts) == {"v4e2", "v5e2", "v6e2", "v7e2"}


def test_B7_no_orphaned_vertices_in_skip_gap():
    """B7: Keine verwaisten Vertices in Skip-Luecken."""
    bmd = _make_bmd_with_marked_top_face()
    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.5), block_delta=(0.0, 0.0, 1.0))]
    result = extrude_with_steps(bmd, steps)

    # The skip gap is from z=1.0 to z=1.5. No vertices should exist at z=1.0
    # (other than the original source vertices which are part of the base block).
    # Specifically, no extruded vertex at the "skipped" level (layer 0 = source).
    # All new vertices should be either at z=1.5 (layer 1) or z=2.5 (layer 2).
    {v.name for v in result.vertices}
    for v in result.vertices:
        if v.name in {"v4", "v5", "v6", "v7"}:
            # Original source vertices are always present
            continue
        if v.name.startswith("v4") or v.name.startswith("v5") or \
           v.name.startswith("v6") or v.name.startswith("v7"):
            # Extruded vertices must be at z=1.5 or z=2.5, not at z=1.0
            assert v.coords[2] != pytest.approx(1.0), (
                f"Orphaned vertex {v.name!r} at z=1.0 in skip gap."
            )


def test_B8_no_side_patches_in_skip_gaps():
    """B8: Keine Side-Patches in Skip-Luecken — Vertices in Skip-Luecken existieren nicht."""
    bmd = _make_bmd_with_marked_top_face()
    # Add a side patch referencing the source face vertices
    patch = Patch("bottom_patch", type="wall", faces=[
        Face(["v0", "v1", "v2", "v3"])  # bottom — not marked
    ])
    bmd.boundary.add(patch)

    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.5), block_delta=(0.0, 0.0, 1.0))]
    result = extrude_with_steps(bmd, steps)

    # Bottom patch references non-marked vertices -> should be unchanged
    bp = result.boundary.get("bottom_patch")
    assert set(bp.faces[0].vertices) == {"v0", "v1", "v2", "v3"}

    # No extra patches should have been added for the skip gap
    patch_names = {p.name for p in result.boundary}
    assert "skip" not in " ".join(patch_names).lower()


def test_B9_skip_with_horizontal_component():
    """B9: Skip mit nicht-vertikaler Komponente -> Block horizontal verschoben."""
    bmd = _make_bmd_with_marked_top_face()
    # Skip 0.5 in x, then block goes up 1.0 in z
    steps = [LayerStep(skip_offset=(0.5, 0.0, 0.0), block_delta=(0.0, 0.0, 1.0))]
    result = extrude_with_steps(bmd, steps)

    assert len(result.blocks) == 2
    # v4 is at (0, 0, 1). Bottom of new block = v4 + (0.5,0,0) = (0.5, 0, 1).
    # This is layer 1. v4e1 should be at x=0.5.
    v4e1 = result.vertices.get("v4e1")
    assert v4e1 is not None
    assert v4e1.coords[0] == pytest.approx(0.5)
    assert v4e1.coords[1] == pytest.approx(0.0)
    assert v4e1.coords[2] == pytest.approx(1.0)


# ===========================================================================
# Integration test I1: Snapshot comparison for a realistic skip scenario
# ===========================================================================

def test_I1_snapshot_skip_scenario():
    """I1: Snapshot-Vergleich mit Referenz-blockMeshDict fuer Skip-Szenario.

    Scenario: Source block with top face at z=1. Two NORMAL layers with a
    SKIP gap between them.
      - Step 1: no skip,  block z=1..1.5
      - Step 2: skip=0.3, block z=1.8..2.3

    Verifies counts and key z-coordinates of all layers.
    """
    bmd = _make_bmd_with_marked_top_face()
    steps = parse_layer_steps("((0 0 0.5) [0 0 0.3] (0 0 0.5))")
    result = extrude_with_steps(bmd, steps)

    # --- Block counts ---
    # 1 original + 2 extruded
    assert len(result.blocks) == 3, "Expected 3 blocks total (1 original + 2 extruded)."

    # --- Vertex counts ---
    # Source layer:   v0..v7 (8)
    # Layer 1 (z=1.5): v4e1..v7e1 (4) — top of block 1
    # Layer 2 (z=1.8): v4e2..v7e2 (4) — bottom of block 2 (after skip)
    # Layer 3 (z=2.3): v4e3..v7e3 (4) — top of block 2
    assert len(result.vertices) == 20, "Expected 20 vertices (8 + 4*3)."

    # --- Z-coordinates ---
    assert result.vertices.get("v4e1").coords[2] == pytest.approx(1.5)
    assert result.vertices.get("v4e2").coords[2] == pytest.approx(1.8)
    assert result.vertices.get("v4e3").coords[2] == pytest.approx(2.3)

    # --- No markers in output ---
    for v in result.vertices:
        assert v.marker is None
    for b in result.blocks:
        assert b.marker is None


# ---------------------------------------------------------------------------
# D.3: Outward-normal self-check tests for extruded blocks
# ---------------------------------------------------------------------------

class TestExtrusionBlockOutwardNormals:
    """D.3: assert_hex_outward_from_coords integration into extrusion pipeline."""

    def _make_bmd_with_marked_face(self, face_idx: int) -> BlockMeshDict:
        """Build a 1x1x1 unit cube BlockMeshDict with the face at *face_idx*
        marked (the top 4 vertices of that face get the //* marker).

        EXTRUSION_LOCAL_FACE_INDICES from extrusion.py defines the 6 faces; we mark the 4 vertices
        of the requested face and also mark the block.
        """
        bmd = BlockMeshDict()
        coords_map = {
            "v0": [0, 0, 0], "v1": [1, 0, 0], "v2": [1, 1, 0], "v3": [0, 1, 0],
            "v4": [0, 0, 1], "v5": [1, 0, 1], "v6": [1, 1, 1], "v7": [0, 1, 1],
        }
        all_names = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        face_local = EXTRUSION_LOCAL_FACE_INDICES[face_idx]
        marked_names = {all_names[i] for i in face_local}

        for name in all_names:
            marker = "face" if name in marked_names else None
            bmd.vertices.add(Vertex(name, coords_map[name], marker=marker))

        bmd.blocks.add(Block(
            "b0",
            vertices=all_names,
            cells=[1, 1, 1],
            grading_type="simpleGrading",
            grading_def=[1.0, 1.0, 1.0],
            marker="extrude",
        ))
        return bmd

    def test_extrusion_block_outward_normals_top_face(self):
        """Extruding the top face (face 1) must produce blocks with outward normals."""
        bmd = self._make_bmd_with_marked_face(face_idx=1)  # top face (kmax)
        result = extrude(bmd, [(0.0, 0.0, 1.0)])

        # Check the new extrusion block (blocks[1])
        assert len(result.blocks) >= 2
        new_block = result.blocks[1]
        name_to_coords = {v.name: tuple(v.coords) for v in result.vertices}
        coords = [name_to_coords[vn] for vn in new_block.vertices]
        # Must not raise
        assert_hex_outward_from_coords(coords, block_label=new_block.name)

    def test_extrusion_block_outward_normals_bottom_face(self):
        """Extruding the bottom face (face 0) must produce blocks with outward normals."""
        bmd = self._make_bmd_with_marked_face(face_idx=0)  # bottom face (kmin)
        result = extrude(bmd, [(0.0, 0.0, -1.0)])

        assert len(result.blocks) >= 2
        new_block = result.blocks[1]
        name_to_coords = {v.name: tuple(v.coords) for v in result.vertices}
        coords = [name_to_coords[vn] for vn in new_block.vertices]
        assert_hex_outward_from_coords(coords, block_label=new_block.name)

    def test_extrusion_self_check_catches_artificial_inversion(self):
        """assert_hex_outward_from_coords must raise for an artificially inverted block."""
        # Standard correct coords
        coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        # Invert: swap v4 and v7 (top face winding becomes inward)
        inv = list(coords)
        inv[4], inv[7] = inv[7], inv[4]
        inv[5], inv[6] = inv[6], inv[5]

        with pytest.raises(OrderingConsistencyError):
            assert_hex_outward_from_coords(inv, block_label="artificial_inversion")


# ---------------------------------------------------------------------------
# zone propagation through extrusion
# ---------------------------------------------------------------------------

class TestExtrusionZonePropagation:

    def test_extrusion_preserves_zone(self):
        """Extruded blocks must carry the same zone as the source block."""
        bmd = _make_bmd_with_marked_top_face()
        # Assign a zone to the source block
        bmd.blocks[0].zone = "fluid"
        result = extrude(bmd, [(0.0, 0.0, 1.0)])
        # result contains original block + at least one extruded block
        extruded_blocks = [b for b in result.blocks if b.name != "b0"]
        assert len(extruded_blocks) >= 1
        for blk in extruded_blocks:
            assert blk.zone == "fluid", (
                f"Extruded block {blk.name!r} has zone={blk.zone!r}, expected 'fluid'"
            )

    def test_extrusion_no_zone_stays_none(self):
        """Extruded blocks from a zone-less source must have zone=None."""
        bmd = _make_bmd_with_marked_top_face()
        assert bmd.blocks[0].zone is None
        result = extrude(bmd, [(0.0, 0.0, 1.0)])
        extruded_blocks = [b for b in result.blocks if b.name != "b0"]
        assert len(extruded_blocks) >= 1
        for blk in extruded_blocks:
            assert blk.zone is None
