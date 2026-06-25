"""Tests for meshing_utils.operations.combine."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshing_utils import (
    Block,
    BlockMeshDict,
    Edge,
    Face,
    Patch,
    Vertex,
)
from meshing_utils.operations.combine import (
    combine_blockmeshdicts,
    discover_source_files,
)
from meshing_utils.operations.split_by_zones import split_blockmeshdict_by_zones

# ---------------------------------------------------------------------------
# Helpers — programmatic BMD construction (mirrors test_split_by_zones.py)
# ---------------------------------------------------------------------------


def _cube_coords(x_offset: float = 0.0) -> list[tuple[float, float, float]]:
    """Return 8 corner coordinates of a unit cube shifted along X."""
    o = x_offset
    return [
        (o + 0.0, 0.0, 0.0),
        (o + 1.0, 0.0, 0.0),
        (o + 1.0, 1.0, 0.0),
        (o + 0.0, 1.0, 0.0),
        (o + 0.0, 0.0, 1.0),
        (o + 1.0, 0.0, 1.0),
        (o + 1.0, 1.0, 1.0),
        (o + 0.0, 1.0, 1.0),
    ]


def _add_cube_block(
    bmd: BlockMeshDict,
    block_name: str,
    vertex_prefix: str,
    zone: str | None = None,
    x_offset: float = 0.0,
) -> Block:
    """Add 8 vertices and one hex block to *bmd*. Returns the Block."""
    coords = _cube_coords(x_offset)
    v_names = [f"{vertex_prefix}{i}" for i in range(8)]
    for name, c in zip(v_names, coords, strict=False):
        bmd.vertices.add(Vertex(name, list(c)))
    block = Block(block_name, vertices=v_names, cells=[1, 1, 1], zone=zone)
    bmd.blocks.add(block)
    return block


def _make_bmd_a() -> BlockMeshDict:
    """Simple BMD with one block in zone 'fluid'."""
    bmd = BlockMeshDict()
    _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
    return bmd


def _make_bmd_b() -> BlockMeshDict:
    """Simple BMD with one block in zone 'solid'."""
    bmd = BlockMeshDict()
    _add_cube_block(bmd, "b_solid", "vs", zone="solid", x_offset=1.0)
    return bmd


# ---------------------------------------------------------------------------
# Happy Path: disjoint BMDs
# ---------------------------------------------------------------------------


class TestDisjointMerge:

    def test_two_disjoint_bmds_union_correct(self):
        """Combining two disjoint BMDs produces the union of their contents."""
        bmd_a = _make_bmd_a()
        bmd_b = _make_bmd_b()
        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        block_names = {b.name for b in combined.blocks}
        assert "b_fluid" in block_names
        assert "b_solid" in block_names
        assert len(combined.blocks) == 2

        vertex_names = {v.name for v in combined.vertices}
        # 8 vertices from each source
        assert len(vertex_names) == 16
        for i in range(8):
            assert f"vf{i}" in vertex_names
            assert f"vs{i}" in vertex_names

    def test_single_source_output_equals_input(self):
        """A single-source combine returns output equivalent to the input."""
        bmd_a = _make_bmd_a()
        combined = combine_blockmeshdicts([bmd_a])

        assert len(combined.blocks) == 1
        assert len(combined.vertices) == 8
        combined_block = next(iter(combined.blocks))
        original_block = next(iter(bmd_a.blocks))
        assert combined_block.name == original_block.name
        assert combined_block.zone == original_block.zone

    def test_sources_not_mutated(self):
        """Source BMDs are not modified by combine."""
        bmd_a = _make_bmd_a()
        bmd_b = _make_bmd_b()
        orig_a_block_names = [b.name for b in bmd_a.blocks]
        orig_b_vertex_names = [v.name for v in bmd_b.vertices]

        combine_blockmeshdicts([bmd_a, bmd_b])

        assert [b.name for b in bmd_a.blocks] == orig_a_block_names
        assert [v.name for v in bmd_b.vertices] == orig_b_vertex_names


# ---------------------------------------------------------------------------
# Shared vertices (deduplication)
# ---------------------------------------------------------------------------


class TestSharedVertices:

    def test_shared_vertices_appear_only_once(self):
        """Vertices with the same name and coordinates are deduplicated."""
        bmd_a = BlockMeshDict()
        # Both BMDs share vertices v0..v3 (the x=1 face)
        shared_coords = [
            (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0),
        ]
        a_coords = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)]
        for i, c in enumerate(a_coords):
            bmd_a.vertices.add(Vertex(f"va{i}", list(c)))
        for i, c in enumerate(shared_coords):
            bmd_a.vertices.add(Vertex(f"vs{i}", list(c)))
        bmd_a.blocks.add(
            Block("block_a", vertices=[f"va{i}" for i in range(4)] + [f"vs{i}" for i in range(4)])
        )

        bmd_b = BlockMeshDict()
        b_coords = [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 1.0, 1.0), (2.0, 0.0, 1.0)]
        for i, c in enumerate(shared_coords):
            bmd_b.vertices.add(Vertex(f"vs{i}", list(c)))  # Same name + same coords
        for i, c in enumerate(b_coords):
            bmd_b.vertices.add(Vertex(f"vb{i}", list(c)))
        bmd_b.blocks.add(
            Block("block_b", vertices=[f"vs{i}" for i in range(4)] + [f"vb{i}" for i in range(4)])
        )

        combined = combine_blockmeshdicts([bmd_a, bmd_b])
        vertex_names = [v.name for v in combined.vertices]
        # Shared vertices (vs0..vs3) must appear exactly once
        for i in range(4):
            assert vertex_names.count(f"vs{i}") == 1


# ---------------------------------------------------------------------------
# Round-trip: split → combine
# ---------------------------------------------------------------------------


class TestRoundTrip:

    def test_split_then_combine_preserves_blocks_and_vertices(self, tmp_path: Path):
        """split → combine round-trip preserves blocks and vertices."""
        original = BlockMeshDict()
        _add_cube_block(original, "b_fluid", "vf", zone="fluid", x_offset=0.0)
        _add_cube_block(original, "b_solid", "vs", zone="solid", x_offset=1.0)

        # Split into per-zone files
        split_blockmeshdict_by_zones(original, tmp_path)

        # Discover and reload
        from meshing_utils.operations.combine import discover_source_files
        source_paths = discover_source_files(tmp_path)
        sources = [BlockMeshDict(p) for p in source_paths]

        combined = combine_blockmeshdicts(
            sources, source_labels=[p.name for p in source_paths]
        )

        # Block names preserved
        orig_block_names = {b.name for b in original.blocks}
        combined_block_names = {b.name for b in combined.blocks}
        assert orig_block_names == combined_block_names

        # Vertex names preserved
        orig_vertex_names = {v.name for v in original.vertices}
        combined_vertex_names = {v.name for v in combined.vertices}
        assert orig_vertex_names == combined_vertex_names


# ---------------------------------------------------------------------------
# Sorting: source discovery is alphabetical and deterministic
# ---------------------------------------------------------------------------


class TestDiscovery:

    def test_discovery_alphabetical_order(self, tmp_path: Path):
        """discover_source_files returns paths sorted alphabetically."""
        for name in ["blockMeshDict_zone_c", "blockMeshDict_zone_a", "blockMeshDict_zone_b"]:
            (tmp_path / name).write_text("", encoding="utf-8")

        discovered = discover_source_files(tmp_path)
        names = [p.name for p in discovered]
        assert names == sorted(names)

    def test_discovery_substring_match(self, tmp_path: Path):
        """discover_source_files finds files containing 'blockMeshDict' in their name."""
        (tmp_path / "blockMeshDict_zone1").write_text("", encoding="utf-8")
        (tmp_path / "blockMeshDict_zone2").write_text("", encoding="utf-8")
        (tmp_path / "other_file.txt").write_text("", encoding="utf-8")

        discovered = discover_source_files(tmp_path)
        names = {p.name for p in discovered}
        assert "blockMeshDict_zone1" in names
        assert "blockMeshDict_zone2" in names
        assert "other_file.txt" not in names

    def test_discovery_excludes_main_file(self, tmp_path: Path):
        """The main 'blockMeshDict' file is always excluded."""
        (tmp_path / "blockMeshDict").write_text("", encoding="utf-8")
        (tmp_path / "blockMeshDict_zone1").write_text("", encoding="utf-8")

        discovered = discover_source_files(tmp_path)
        names = {p.name for p in discovered}
        assert "blockMeshDict" not in names
        assert "blockMeshDict_zone1" in names

    def test_discovery_exclude_list_filters(self, tmp_path: Path):
        """Files in the --exclude list are excluded from discovery."""
        (tmp_path / "blockMeshDict_zone1").write_text("", encoding="utf-8")
        (tmp_path / "blockMeshDict_zone2").write_text("", encoding="utf-8")

        discovered = discover_source_files(tmp_path, excludes=["blockMeshDict_zone2"])
        names = {p.name for p in discovered}
        assert "blockMeshDict_zone1" in names
        assert "blockMeshDict_zone2" not in names

    def test_discovery_no_matches_raises(self, tmp_path: Path):
        """ValueError is raised when no source files are discovered."""
        # Only place the main file (excluded) + an unrelated file
        (tmp_path / "blockMeshDict").write_text("", encoding="utf-8")
        (tmp_path / "unrelated.txt").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="No blockMeshDict fragment files found"):
            discover_source_files(tmp_path)


# ---------------------------------------------------------------------------
# combine_cell_zones override
# ---------------------------------------------------------------------------


class TestCombineCellZones:

    def test_combine_cell_zones_overrides_all_zones(self):
        """combine_cell_zones overrides ALL block zones including None."""
        bmd_a = _make_bmd_a()  # zone='fluid'
        bmd_b = _make_bmd_b()  # zone='solid'
        bmd_c = BlockMeshDict()
        _add_cube_block(bmd_c, "b_none", "vn", zone=None, x_offset=2.0)

        combined = combine_blockmeshdicts(
            [bmd_a, bmd_b, bmd_c],
            combine_cell_zones="rotor",
        )

        for block in combined.blocks:
            assert block.zone == "rotor", (
                f"Block {block.name!r} has zone {block.zone!r}, expected 'rotor'"
            )

    def test_without_combine_cell_zones_zones_preserved(self):
        """Without combine_cell_zones, each block retains its original zone."""
        bmd_a = _make_bmd_a()  # zone='fluid'
        bmd_b = _make_bmd_b()  # zone='solid'

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        zones = {b.name: b.zone for b in combined.blocks}
        assert zones["b_fluid"] == "fluid"
        assert zones["b_solid"] == "solid"


# ---------------------------------------------------------------------------
# Edge cases: conflicts
# ---------------------------------------------------------------------------


class TestVertexConflicts:

    def test_vertex_name_coord_conflict_renamed_on_new_coord(self):
        """Vertex name collision with genuinely new coord: renamed v0 -> v0_2, no error."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        # Give bmd_a a block that references v0 so we can verify remap in blocks
        bmd_a.blocks.add(Block("block_a", vertices=["v0"] * 8, cells=[1, 1, 1]))

        bmd_b = BlockMeshDict()
        bmd_b.vertices.add(Vertex("v0", [99.0, 0.0, 0.0]))  # Different coords
        bmd_b.blocks.add(Block("block_b", vertices=["v0"] * 8, cells=[1, 1, 1]))

        # Should NOT raise
        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        vertex_names = [v.name for v in combined.vertices]
        assert "v0" in vertex_names, "Original v0 must be present"
        assert "v0_2" in vertex_names, "Renamed vertex v0_2 must be present"

        # Coords map correctly
        assert combined.vertices.get("v0").coords == [0.0, 0.0, 0.0]
        assert combined.vertices.get("v0_2").coords == [99.0, 0.0, 0.0]

        # The block from bmd_b must reference v0_2, not v0
        block_b = combined.blocks.get("block_b")
        assert "v0_2" in block_b.vertices, "block_b must reference renamed vertex v0_2"
        assert "v0" not in block_b.vertices, "block_b must not reference original v0"

    def test_vertex_name_collision_collapses_onto_existing_coord(self):
        """Name collision where incoming coord matches an existing vertex under a different name."""
        bmd_a = BlockMeshDict()
        # vx at [5,5,5] and v0 at [0,0,0]
        bmd_a.vertices.add(Vertex("vx", [5.0, 5.0, 5.0]))
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        block_a_verts = ["v0", "vx", "v0", "vx", "v0", "vx", "v0", "vx"]
        bmd_a.blocks.add(Block("block_a", vertices=block_a_verts, cells=[1, 1, 1]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b has coords [5,5,5] — same as bmd_a's vx, but name is v0
        bmd_b.vertices.add(Vertex("v0", [5.0, 5.0, 5.0]))
        bmd_b.blocks.add(Block("block_b", vertices=["v0"] * 8, cells=[1, 1, 1]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        vertex_names = [v.name for v in combined.vertices]
        # No v0_2 should be added — it collapsed onto vx
        assert "v0_2" not in vertex_names, "No renamed v0_2 expected (should collapse)"
        assert vertex_names.count("v0") == 1, "v0 from bmd_a must still exist"
        assert "vx" in vertex_names, "vx must still exist"

        # block_b references should be remapped from v0 -> vx
        block_b = combined.blocks.get("block_b")
        assert all(n == "vx" for n in block_b.vertices), (
            "block_b vertices must all reference 'vx' after collapse"
        )

    def test_vertex_within_tol_is_ok(self):
        """Vertices within vertex_tol are treated as identical (no error)."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))

        bmd_b = BlockMeshDict()
        # Difference of 1e-10 which is < default tol 1e-9
        bmd_b.vertices.add(Vertex("v0", [1e-10, 0.0, 0.0]))

        # Should not raise
        combined = combine_blockmeshdicts([bmd_a, bmd_b], vertex_tol=1e-9)
        vertex_names = [v.name for v in combined.vertices]
        assert vertex_names.count("v0") == 1


class TestVertexRemapPropagation:
    """Verify that vertex remap from _merge_vertices propagates to blocks, edges, and faces."""

    def test_renamed_vertex_propagates_to_blocks(self):
        """After vertex rename collision, merged block references the new name."""
        bmd_a = BlockMeshDict()
        _add_cube_block(bmd_a, "block_a", "va", x_offset=0.0)

        bmd_b = BlockMeshDict()
        # Reuse vertex name "va0" (same as bmd_a's va0) but with different coords
        bmd_b.vertices.add(Vertex("va0", [100.0, 0.0, 0.0]))
        v_names = ["va0"] + [f"vb{i}" for i in range(7)]
        for i, c in enumerate(_cube_coords(x_offset=100.0)[1:], start=1):
            bmd_b.vertices.add(Vertex(f"vb{i - 1}", list(c)))
        bmd_b.blocks.add(Block("block_b", vertices=v_names, cells=[1, 1, 1]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        block_b = combined.blocks.get("block_b")
        # va0 in bmd_b had a different coord -> renamed to va0_2
        assert "va0_2" in block_b.vertices, "block_b should reference renamed vertex va0_2"
        assert "va0" not in block_b.vertices, "block_b should not reference original va0"

    def test_renamed_vertex_propagates_to_edges(self):
        """After vertex rename collision, edge endpoints are remapped."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd_a.vertices.add(Vertex("v1", [1.0, 0.0, 0.0]))
        bmd_a.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b at different coords -> will be renamed v0_2
        bmd_b.vertices.add(Vertex("v0", [10.0, 0.0, 0.0]))
        bmd_b.vertices.add(Vertex("v2", [11.0, 0.0, 0.0]))
        # Edge from v0 (bmd_b) to v2 — should be remapped to v0_2 -> v2
        bmd_b.edges.add(Edge("arc", "v0", "v2", coords=[10.5, 0.1, 0.0]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        edge_endpoints = {frozenset({e.v_start, e.v_end}) for e in combined.edges}
        assert frozenset({"v0_2", "v2"}) in edge_endpoints, (
            "Edge from bmd_b must use remapped endpoint v0_2"
        )
        # Original edge still present
        assert frozenset({"v0", "v1"}) in edge_endpoints

    def test_renamed_vertex_propagates_to_patches(self):
        """After vertex rename collision, patch face vertices are remapped."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd_a.boundary.add(Patch("walls", type="wall", faces=[Face(["v0", "v1", "v2", "v3"])]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b at different coords -> renamed v0_2
        bmd_b.vertices.add(Vertex("v0", [99.0, 0.0, 0.0]))
        bmd_b.boundary.add(Patch("outlet", type="patch", faces=[Face(["v0", "v4", "v5", "v6"])]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        outlet_patch = None
        for p in combined.boundary:
            if p.name == "outlet":
                outlet_patch = p
                break
        assert outlet_patch is not None
        face_verts = outlet_patch.faces[0].vertices
        assert "v0_2" in face_verts, "Face in outlet patch must use remapped v0_2"
        assert "v0" not in face_verts, "Face in outlet patch must not reference original v0"

    def test_sources_not_mutated_on_collision(self):
        """Source BMD vertices/blocks/edges/faces are unchanged after a collision combine."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd_a.blocks.add(Block("block_a", vertices=["v0"] * 8, cells=[1, 1, 1]))

        bmd_b = BlockMeshDict()
        bmd_b.vertices.add(Vertex("v0", [99.0, 0.0, 0.0]))
        bmd_b.blocks.add(Block("block_b", vertices=["v0"] * 8, cells=[1, 1, 1]))
        bmd_b.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))
        bmd_b.boundary.add(Patch("walls", type="wall", faces=[Face(["v0", "v1", "v2", "v3"])]))

        # Remember original state of bmd_b
        first_v = next(iter(bmd_b.vertices))
        orig_b_vertex_name = first_v.name
        orig_b_vertex_coords = list(first_v.coords)
        orig_b_block_verts = list(next(iter(bmd_b.blocks)).vertices)
        orig_b_edge_start = next(iter(bmd_b.edges)).v_start
        orig_b_face_verts = list(next(iter(bmd_b.boundary)).faces[0].vertices)

        combine_blockmeshdicts([bmd_a, bmd_b])

        # bmd_b must be completely unchanged
        first_v_after = next(iter(bmd_b.vertices))
        assert first_v_after.name == orig_b_vertex_name
        assert first_v_after.coords == orig_b_vertex_coords
        assert list(next(iter(bmd_b.blocks)).vertices) == orig_b_block_verts
        assert next(iter(bmd_b.edges)).v_start == orig_b_edge_start
        assert list(next(iter(bmd_b.boundary)).faces[0].vertices) == orig_b_face_verts


    def test_collapsed_vertex_propagates_to_edges(self):
        """COLLAPSE path: edge endpoints are remapped to the collapsed-onto name."""
        bmd_a = BlockMeshDict()
        # vx at [5,5,5] and v0 at [0,0,0]
        bmd_a.vertices.add(Vertex("vx", [5.0, 5.0, 5.0]))
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        # An edge in bmd_a from vx to v0
        bmd_a.edges.add(Edge("arc", "vx", "v0", coords=[2.5, 2.5, 0.0]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b has coords [5,5,5] — same as bmd_a's vx (collapse R2)
        bmd_b.vertices.add(Vertex("v0", [5.0, 5.0, 5.0]))
        bmd_b.vertices.add(Vertex("vb1", [10.0, 0.0, 0.0]))
        # Edge in bmd_b from v0 to vb1 — v0 must be remapped to vx
        bmd_b.edges.add(Edge("arc", "v0", "vb1", coords=[7.5, 0.1, 0.0]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        edge_endpoints = {frozenset({e.v_start, e.v_end}) for e in combined.edges}
        # bmd_b's edge endpoint v0 should be remapped to vx (the collapse target)
        assert frozenset({"vx", "vb1"}) in edge_endpoints, (
            "bmd_b edge must use collapsed-onto name 'vx', not 'v0'"
        )
        # No edge with the original v0 name for bmd_b's edge
        assert frozenset({"v0", "vb1"}) not in edge_endpoints, (
            "Edge must not reference original 'v0' after collapse"
        )

    def test_collapsed_vertex_propagates_to_patches(self):
        """COLLAPSE path: patch face vertices are remapped to the collapsed-onto name."""
        bmd_a = BlockMeshDict()
        # vx at [5,5,5] and v0 at [0,0,0]
        bmd_a.vertices.add(Vertex("vx", [5.0, 5.0, 5.0]))
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd_a.boundary.add(Patch("inlet", type="patch", faces=[Face(["v0", "v1", "v2", "v3"])]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b has coords [5,5,5] — same as bmd_a's vx (collapse R2)
        bmd_b.vertices.add(Vertex("v0", [5.0, 5.0, 5.0]))
        # Face in bmd_b references v0 (and unique vertices) — v0 must collapse to vx
        bmd_b.boundary.add(Patch("outlet", type="patch", faces=[Face(["v0", "v4", "v5", "v6"])]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        outlet_patch = None
        for p in combined.boundary:
            if p.name == "outlet":
                outlet_patch = p
                break
        assert outlet_patch is not None, "outlet patch must be present"
        face_verts = outlet_patch.faces[0].vertices
        assert "vx" in face_verts, (
            "Face in outlet patch must reference collapsed-onto name 'vx'"
        )
        assert "v0" not in face_verts, (
            "Face in outlet patch must not reference original 'v0' after collapse"
        )

    def test_resolve_vertex_name_falls_back_to_suffix_3(self):
        """resolve_vertex_name: when v0_2 is already taken, the next name is v0_3."""
        bmd_a = BlockMeshDict()
        # Pre-occupy both v0 and v0_2 in bmd_a
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))
        bmd_a.vertices.add(Vertex("v0_2", [1.0, 0.0, 0.0]))

        bmd_b = BlockMeshDict()
        # v0 in bmd_b: coords differ from v0 ([0,0,0]) AND from v0_2 ([1,0,0])
        # and do not match any other existing vertex -> must be renamed v0_3
        bmd_b.vertices.add(Vertex("v0", [99.0, 0.0, 0.0]))
        bmd_b.blocks.add(Block("block_b", vertices=["v0"] * 8, cells=[1, 1, 1]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])

        vertex_names = [v.name for v in combined.vertices]
        assert "v0_3" in vertex_names, (
            "When v0_2 is already occupied, the renamed vertex must be v0_3"
        )
        assert combined.vertices.get("v0_3").coords == [99.0, 0.0, 0.0], (
            "v0_3 must carry the new coordinates"
        )

        # block_b must reference v0_3, not v0 or v0_2
        block_b = combined.blocks.get("block_b")
        assert all(n == "v0_3" for n in block_b.vertices), (
            "block_b must reference v0_3 after double suffix collision"
        )


class TestBlockNameConflicts:

    def test_block_name_collision_auto_renamed(self):
        """Conflicting block names are auto-renamed via resolve_block_name."""
        bmd_a = BlockMeshDict()
        _add_cube_block(bmd_a, "my_block", "va", x_offset=0.0)

        bmd_b = BlockMeshDict()
        _add_cube_block(bmd_b, "my_block", "vb", x_offset=1.0)  # Same block name

        combined = combine_blockmeshdicts([bmd_a, bmd_b])
        block_names = [b.name for b in combined.blocks]

        # Both blocks present, but with distinct names
        assert len(block_names) == 2
        assert "my_block" in block_names
        # The second one must have been renamed (e.g. my_block_2)
        assert block_names[0] != block_names[1]


class TestEdgeConflicts:

    def test_edge_type_conflict_warning(self):
        """Edge type conflict does not raise when strict=False; first edge is kept."""
        bmd_a = BlockMeshDict()
        bmd_a.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))

        bmd_b = BlockMeshDict()
        bmd_b.edges.add(Edge("BSpline", "v0", "v1", coords=[0.5, 0.2, 0.0]))

        # No exception expected — conflict is only logged as a warning
        combined = combine_blockmeshdicts([bmd_a, bmd_b], strict=False)

        # Only the first (original) edge should remain
        edges = list(combined.edges)
        assert len(edges) == 1
        assert edges[0].type == "arc"

    def test_edge_type_conflict_strict_raises(self):
        """Edge type conflict raises ValueError when strict=True."""
        bmd_a = BlockMeshDict()
        bmd_a.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))

        bmd_b = BlockMeshDict()
        bmd_b.edges.add(Edge("BSpline", "v0", "v1", coords=[0.5, 0.2, 0.0]))

        with pytest.raises(ValueError, match="Edge conflict"):
            combine_blockmeshdicts([bmd_a, bmd_b], strict=True)

    def test_duplicate_edge_skipped(self):
        """Edges with same endpoints, type, and geometry appear only once."""
        bmd_a = BlockMeshDict()
        bmd_a.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))

        bmd_b = BlockMeshDict()
        bmd_b.edges.add(Edge("arc", "v0", "v1", coords=[0.5, 0.1, 0.0]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])
        assert len(combined.edges) == 1


class TestPatchConflicts:

    def test_patch_type_conflict_warning(self):
        """Patch type conflict logs a warning when strict=False."""
        bmd_a = BlockMeshDict()
        bmd_a.boundary.add(Patch("walls", type="wall", faces=[]))

        bmd_b = BlockMeshDict()
        bmd_b.boundary.add(Patch("walls", type="symmetry", faces=[]))

        # No exception expected
        combined = combine_blockmeshdicts([bmd_a, bmd_b], strict=False)
        # First type wins
        for p in combined.boundary:
            if p.name == "walls":
                assert p.type == "wall"

    def test_patch_type_conflict_strict_raises(self):
        """Patch type conflict raises ValueError when strict=True."""
        bmd_a = BlockMeshDict()
        bmd_a.boundary.add(Patch("walls", type="wall", faces=[]))

        bmd_b = BlockMeshDict()
        bmd_b.boundary.add(Patch("walls", type="symmetry", faces=[]))

        with pytest.raises(ValueError, match="Patch type conflict"):
            combine_blockmeshdicts([bmd_a, bmd_b], strict=True)

    def test_patch_merging_same_name_faces_unioned(self):
        """Patches with same name have their faces merged and deduplicated."""
        face_a = Face(["v0", "v1", "v2", "v3"])
        face_b = Face(["v4", "v5", "v6", "v7"])
        face_dup = Face(["v0", "v1", "v2", "v3"])  # Duplicate of face_a

        bmd_a = BlockMeshDict()
        bmd_a.boundary.add(Patch("walls", type="wall", faces=[face_a]))

        bmd_b = BlockMeshDict()
        bmd_b.boundary.add(Patch("walls", type="wall", faces=[face_dup, face_b]))

        combined = combine_blockmeshdicts([bmd_a, bmd_b])
        walls_patch = None
        for p in combined.boundary:
            if p.name == "walls":
                walls_patch = p
                break
        assert walls_patch is not None

        face_keys = {tuple(sorted(f.vertices)) for f in walls_patch.faces}
        # face_a and face_b both present, but face_dup deduplicated
        assert tuple(sorted(["v0", "v1", "v2", "v3"])) in face_keys
        assert tuple(sorted(["v4", "v5", "v6", "v7"])) in face_keys
        assert len(walls_patch.faces) == 2

    def test_face_in_two_patches_warns(self):
        """A face present in two different patches triggers a warning."""
        shared_face_a = Face(["v0", "v1", "v2", "v3"])
        shared_face_b = Face(["v0", "v1", "v2", "v3"])

        bmd_a = BlockMeshDict()
        bmd_a.boundary.add(Patch("patch_a", type="wall", faces=[shared_face_a]))

        bmd_b = BlockMeshDict()
        bmd_b.boundary.add(Patch("patch_b", type="wall", faces=[shared_face_b]))

        # No exception; but the face in patch_b should be skipped
        combined = combine_blockmeshdicts([bmd_a, bmd_b], strict=False)

        # The face should appear in exactly one patch
        face_key = tuple(sorted(["v0", "v1", "v2", "v3"]))
        count = 0
        for p in combined.boundary:
            for f in p.faces:
                if tuple(sorted(f.vertices)) == face_key:
                    count += 1
        assert count == 1

    def test_face_in_two_patches_strict_raises(self):
        """A face in two different patches raises ValueError when strict=True."""
        shared_face_a = Face(["v0", "v1", "v2", "v3"])
        shared_face_b = Face(["v0", "v1", "v2", "v3"])

        bmd_a = BlockMeshDict()
        bmd_a.boundary.add(Patch("patch_a", type="wall", faces=[shared_face_a]))

        bmd_b = BlockMeshDict()
        bmd_b.boundary.add(Patch("patch_b", type="wall", faces=[shared_face_b]))

        with pytest.raises(ValueError, match="already present in patch"):
            combine_blockmeshdicts([bmd_a, bmd_b], strict=True)


class TestScalarConflicts:

    def test_convertTometers_mismatch_warns(self):
        """convertToMeters mismatch logs a warning."""
        bmd_a = BlockMeshDict()
        bmd_a.convertToMeters = 1.0

        bmd_b = BlockMeshDict()
        bmd_b.convertToMeters = 0.001

        # Should not raise, just warn
        combined = combine_blockmeshdicts([bmd_a, bmd_b], strict=False)
        # First value wins
        assert abs(combined.convertToMeters - 1.0) < 1e-12

    def test_convertTometers_mismatch_strict_raises(self):
        """convertToMeters mismatch raises ValueError when strict=True."""
        bmd_a = BlockMeshDict()
        bmd_a.convertToMeters = 1.0

        bmd_b = BlockMeshDict()
        bmd_b.convertToMeters = 0.001

        with pytest.raises(ValueError, match="convertToMeters conflict"):
            combine_blockmeshdicts([bmd_a, bmd_b], strict=True)


class TestEmptySourceList:

    def test_empty_source_list_raises_valueerror(self):
        """Combining an empty list of sources raises ValueError."""
        with pytest.raises(ValueError, match="at least one source"):
            combine_blockmeshdicts([])
