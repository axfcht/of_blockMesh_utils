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

    def test_vertex_name_coord_conflict_raises_valueerror(self):
        """Vertex with same name but different coordinates always raises."""
        bmd_a = BlockMeshDict()
        bmd_a.vertices.add(Vertex("v0", [0.0, 0.0, 0.0]))

        bmd_b = BlockMeshDict()
        bmd_b.vertices.add(Vertex("v0", [99.0, 0.0, 0.0]))  # Different coords

        with pytest.raises(ValueError, match="Vertex name conflict"):
            combine_blockmeshdicts([bmd_a, bmd_b])

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
