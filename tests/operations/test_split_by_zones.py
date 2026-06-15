"""Tests for meshing_utils.operations.split_by_zones."""

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
from meshing_utils.operations.split_by_zones import split_blockmeshdict_by_zones

# ---------------------------------------------------------------------------
# Helpers — programmatic BMD construction
# ---------------------------------------------------------------------------


def _make_vertex_row(prefix: str, offsets: list[tuple[float, float, float]]) -> list[Vertex]:
    """Return a list of named Vertex objects."""
    return [Vertex(f"{prefix}{i}", list(coords)) for i, coords in enumerate(offsets)]


def _cube_coords(x_offset: float = 0.0) -> list[tuple[float, float, float]]:
    """Return the 8 corner coordinates of a unit cube shifted along X."""
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


def _make_simple_bmd() -> BlockMeshDict:
    """Return a BMD with three blocks: fluid (x=0), solid (x=1), no zone (x=2)."""
    bmd = BlockMeshDict()
    _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
    _add_cube_block(bmd, "b_solid", "vs", zone="solid", x_offset=1.0)
    _add_cube_block(bmd, "b_none", "vn", zone=None, x_offset=2.0)
    return bmd


def _make_two_zone_bmd_with_shared_face() -> tuple[BlockMeshDict, str, str]:
    """Return a BMD with two adjacent blocks that share an interface face.

    The shared face uses vertices at x=1 (the common face between block A at
    x=0..1 and block B at x=1..2).  Returns (bmd, patch_left_name, patch_right_name).
    """
    bmd = BlockMeshDict()

    # Block A: x=0..1
    va_names = [f"va{i}" for i in range(8)]
    for name, c in zip(va_names, _cube_coords(0.0), strict=False):
        bmd.vertices.add(Vertex(name, list(c)))
    block_a = Block("b_a", vertices=va_names, cells=[1, 1, 1], zone="zoneA")
    bmd.blocks.add(block_a)

    # Block B: x=1..2  — shares the x=1 face with block A
    # va1, va2, va5, va6 are the shared vertices (x=1 face of block A)
    # For block B we reuse those same vertex names in the appropriate positions
    vb_new_names = ["vb0", "vb1", "vb2", "vb3"]  # x=2 far face of block B
    for name, c in zip(vb_new_names, [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
                                       (2.0, 1.0, 1.0), (2.0, 0.0, 1.0)], strict=False):
        bmd.vertices.add(Vertex(name, list(c)))
    # blockB vertex order: [shared_bottom_front, shared_top_front, far_top, far_bottom, ...]
    # Use: va1(1,0,0), vb0(2,0,0), vb1(2,1,0), va2(1,1,0) — bottom face
    #      va5(1,0,1), vb3(2,0,1), vb2(2,1,1), va6(1,1,1) — top face
    block_b_verts = ["va1", "vb0", "vb1", "va2", "va5", "vb3", "vb2", "va6"]
    block_b = Block("b_b", vertices=block_b_verts, cells=[1, 1, 1], zone="zoneB")
    bmd.blocks.add(block_b)

    # Add a patch that contains the shared interface face
    # This face (va1, va2, va6, va5) is on the right of block A and left of block B
    shared_face = Face(["va1", "va2", "va6", "va5"])
    patch = Patch("interface", type="wall", faces=[shared_face])
    bmd.boundary.add(patch)

    return bmd


# ---------------------------------------------------------------------------
# Test 1: Default mode — one file per zone bucket
# ---------------------------------------------------------------------------


class TestDefaultMode:

    def test_group_blocks_by_zone_default(self, tmp_path: Path):
        """Default mode produces one file per zone bucket."""
        bmd = _make_simple_bmd()
        written = split_blockmeshdict_by_zones(bmd, tmp_path)
        names = {p.name for p in written}
        assert "blockMeshDict_fluid" in names
        assert "blockMeshDict_solid" in names
        assert "blockMeshDict_no_zone" in names
        assert len(written) == 3

    def test_blocks_without_zone_become_no_zone_file(self, tmp_path: Path):
        """Un-zoned blocks land in blockMeshDict_no_zone."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b0", "v", zone=None)
        written = split_blockmeshdict_by_zones(bmd, tmp_path)
        assert len(written) == 1
        assert written[0].name == "blockMeshDict_no_zone"


# ---------------------------------------------------------------------------
# Test 3 & 5: --include mode
# ---------------------------------------------------------------------------


class TestIncludeMode:

    def test_include_produces_rest_file(self, tmp_path: Path):
        """--include fluid → _fluid + _rest; _rest contains solid + no_zone blocks."""
        bmd = _make_simple_bmd()
        written = split_blockmeshdict_by_zones(bmd, tmp_path, include=["fluid"])
        names = {p.name for p in written}
        assert "blockMeshDict_fluid" in names
        assert "blockMeshDict_rest" in names
        assert "blockMeshDict_solid" not in names
        assert "blockMeshDict_no_zone" not in names
        assert len(written) == 2

    def test_rest_file_contains_solid_and_no_zone_blocks(self, tmp_path: Path):
        """_rest file content contains blocks from solid and no_zone buckets."""
        bmd = _make_simple_bmd()
        split_blockmeshdict_by_zones(bmd, tmp_path, include=["fluid"])
        rest_bmd = BlockMeshDict(tmp_path / "blockMeshDict_rest")
        block_names = {b.name for b in rest_bmd.blocks}
        assert "b_solid" in block_names
        assert "b_none" in block_names
        assert "b_fluid" not in block_names

    def test_no_rest_when_include_covers_all_zones(self, tmp_path: Path):
        """No _rest file when --include covers every zone (including no_zone implicitly)."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid")
        # No un-zoned blocks — include all explicit zones
        written = split_blockmeshdict_by_zones(bmd, tmp_path, include=["fluid"])
        names = {p.name for p in written}
        assert "blockMeshDict_rest" not in names
        assert "blockMeshDict_fluid" in names


# ---------------------------------------------------------------------------
# Test 4: --exclude mode
# ---------------------------------------------------------------------------


class TestExcludeMode:

    def test_exclude_produces_rest_file(self, tmp_path: Path):
        """--exclude solid → _fluid, _no_zone (own files), _rest (with solid blocks)."""
        bmd = _make_simple_bmd()
        written = split_blockmeshdict_by_zones(bmd, tmp_path, exclude=["solid"])
        names = {p.name for p in written}
        assert "blockMeshDict_fluid" in names
        assert "blockMeshDict_no_zone" in names
        assert "blockMeshDict_rest" in names
        assert "blockMeshDict_solid" not in names

    def test_exclude_rest_contains_excluded_blocks(self, tmp_path: Path):
        """_rest file in --exclude mode contains the excluded zone's blocks."""
        bmd = _make_simple_bmd()
        split_blockmeshdict_by_zones(bmd, tmp_path, exclude=["solid"])
        rest_bmd = BlockMeshDict(tmp_path / "blockMeshDict_rest")
        block_names = {b.name for b in rest_bmd.blocks}
        assert "b_solid" in block_names
        assert "b_fluid" not in block_names


# ---------------------------------------------------------------------------
# Tests 6-10: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_unknown_zone_in_include_raises(self, tmp_path: Path):
        """ValueError when --include names a zone not present in the BMD."""
        bmd = _make_simple_bmd()
        with pytest.raises(ValueError, match="Unknown zone"):
            split_blockmeshdict_by_zones(bmd, tmp_path, include=["ghost_zone"])

    def test_unknown_zone_in_exclude_raises(self, tmp_path: Path):
        """ValueError when --exclude names a zone not present in the BMD."""
        bmd = _make_simple_bmd()
        with pytest.raises(ValueError, match="Unknown zone"):
            split_blockmeshdict_by_zones(bmd, tmp_path, exclude=["ghost_zone"])

    def test_include_and_exclude_together_raises(self, tmp_path: Path):
        """ValueError when both --include and --exclude are given."""
        bmd = _make_simple_bmd()
        with pytest.raises(ValueError, match="mutually exclusive"):
            split_blockmeshdict_by_zones(
                bmd, tmp_path, include=["fluid"], exclude=["solid"]
            )

    def test_reserved_zone_name_no_zone_raises(self, tmp_path: Path):
        """A block with zone 'no_zone' raises ValueError (reserved name)."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b0", "v", zone="no_zone")
        with pytest.raises(ValueError, match="reserved bucket name"):
            split_blockmeshdict_by_zones(bmd, tmp_path)

    def test_reserved_zone_name_rest_raises(self, tmp_path: Path):
        """A block with zone 'rest' raises ValueError (reserved name)."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b0", "v", zone="rest")
        with pytest.raises(ValueError, match="reserved bucket name"):
            split_blockmeshdict_by_zones(bmd, tmp_path)


# ---------------------------------------------------------------------------
# Test 11: zone attribute preserved in _rest
# ---------------------------------------------------------------------------


class TestZonePreservationInRest:

    def test_rest_preserves_original_zone_names_per_block(self, tmp_path: Path):
        """Blocks in _rest retain their original zone attribute."""
        bmd = _make_simple_bmd()
        split_blockmeshdict_by_zones(bmd, tmp_path, include=["fluid"])
        rest_bmd = BlockMeshDict(tmp_path / "blockMeshDict_rest")
        zones_in_rest = {b.name: b.zone for b in rest_bmd.blocks}
        assert zones_in_rest["b_solid"] == "solid"
        # b_none had zone=None originally; after write/read it may be None or absent
        assert zones_in_rest.get("b_none") is None or zones_in_rest.get("b_none") == ""


# ---------------------------------------------------------------------------
# Tests 12-14: Subset construction (vertices, edges, faces)
# ---------------------------------------------------------------------------


class TestSubsetConstruction:

    def test_subset_vertices_only_referenced_ones(self, tmp_path: Path):
        """Output BMD contains only vertices referenced by its blocks."""
        bmd = _make_simple_bmd()
        split_blockmeshdict_by_zones(bmd, tmp_path)
        fluid_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")
        # fluid block uses vertices vf0..vf7 — none of vs* or vn* should appear
        vertex_names = {v.name for v in fluid_bmd.vertices}
        for vname in vertex_names:
            assert vname.startswith("vf"), (
                f"Unexpected vertex {vname!r} in fluid subset"
            )
        assert len(vertex_names) == 8

    def test_subset_edges_filtered_correctly(self, tmp_path: Path):
        """Edges whose endpoints lie outside the vertex subset are dropped."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
        _add_cube_block(bmd, "b_solid", "vs", zone="solid", x_offset=1.0)
        # Edge between the two zones (cross-zone edge — should be excluded from
        # fluid-only output)
        bmd.edges.add(Edge("arc", "vf1", "vs0", coords=[1.5, 0.5, 0.0]))
        # Edge entirely within fluid zone (should be kept)
        bmd.edges.add(Edge("arc", "vf0", "vf1", coords=[0.5, -0.1, 0.0]))

        split_blockmeshdict_by_zones(bmd, tmp_path)
        fluid_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")
        edge_pairs = [(e.v_start, e.v_end) for e in fluid_bmd.edges]
        # Cross-zone edge must not appear
        assert ("vf1", "vs0") not in edge_pairs
        # Intra-fluid edge must appear
        assert ("vf0", "vf1") in edge_pairs

    def test_subset_faces_all_four_vertices_rule(self, tmp_path: Path):
        """A face with a vertex outside the subset is excluded."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
        _add_cube_block(bmd, "b_solid", "vs", zone="solid", x_offset=1.0)
        # Pure-fluid face (all vf*)
        fluid_face = Face(["vf0", "vf1", "vf2", "vf3"])
        # Mixed face (3 fluid + 1 solid vertex)
        mixed_face = Face(["vf1", "vs0", "vf2", "vf3"])
        patch = Patch("walls", type="wall", faces=[fluid_face, mixed_face])
        bmd.boundary.add(patch)

        split_blockmeshdict_by_zones(bmd, tmp_path)
        fluid_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")
        walls_patch = fluid_bmd.boundary.get("walls")
        face_vertex_sets = [frozenset(f.vertices) for f in walls_patch.faces]
        assert frozenset(["vf0", "vf1", "vf2", "vf3"]) in face_vertex_sets
        assert frozenset(["vf1", "vs0", "vf2", "vf3"]) not in face_vertex_sets


# ---------------------------------------------------------------------------
# Test 15: Shared interface face appears in both outputs
# ---------------------------------------------------------------------------


class TestSharedFace:

    def test_shared_interface_face_appears_in_both_outputs(self, tmp_path: Path):
        """A face shared between two zones appears in both output files."""
        bmd = _make_two_zone_bmd_with_shared_face()
        split_blockmeshdict_by_zones(bmd, tmp_path)

        zone_a_bmd = BlockMeshDict(tmp_path / "blockMeshDict_zoneA")
        zone_b_bmd = BlockMeshDict(tmp_path / "blockMeshDict_zoneB")

        shared_verts = frozenset(["va1", "va2", "va6", "va5"])

        def _has_shared_face(b: BlockMeshDict) -> bool:
            for patch in b.boundary:
                for face in patch.faces:
                    if frozenset(face.vertices) == shared_verts:
                        return True
            return False

        assert _has_shared_face(zone_a_bmd), "shared face missing from zoneA output"
        assert _has_shared_face(zone_b_bmd), "shared face missing from zoneB output"


# ---------------------------------------------------------------------------
# Tests 16-17: Empty patches
# ---------------------------------------------------------------------------


class TestEmptyPatches:

    def _bmd_with_patch_outside_zone(self) -> BlockMeshDict:
        """BMD with one fluid block and a patch whose face uses solid vertices."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
        _add_cube_block(bmd, "b_solid", "vs", zone="solid", x_offset=1.0)
        # Patch with only solid-zone vertices → empty after fluid-zone split
        solid_face = Face(["vs0", "vs1", "vs2", "vs3"])
        patch = Patch("solid_wall", type="wall", faces=[solid_face])
        bmd.boundary.add(patch)
        return bmd

    def test_empty_patches_dropped_by_default(self, tmp_path: Path):
        """Patches with no surviving faces are dropped by default."""
        bmd = self._bmd_with_patch_outside_zone()
        split_blockmeshdict_by_zones(bmd, tmp_path)
        fluid_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")
        patch_names = {p.name for p in fluid_bmd.boundary}
        assert "solid_wall" not in patch_names

    def test_keep_empty_patches_flag_retains_them(self, tmp_path: Path):
        """With keep_empty_patches=True, empty patches are retained."""
        bmd = self._bmd_with_patch_outside_zone()
        split_blockmeshdict_by_zones(bmd, tmp_path, keep_empty_patches=True)
        fluid_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")
        patch_names = {p.name for p in fluid_bmd.boundary}
        assert "solid_wall" in patch_names


# ---------------------------------------------------------------------------
# Test 18: reindex_vertices
# ---------------------------------------------------------------------------


class TestReindexVertices:

    def test_reindex_vertices_remaps_blocks_and_faces_and_edges(self, tmp_path: Path):
        """reindex_vertices=True produces compact v0..vN-1 names."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid", x_offset=0.0)
        # Add an intra-fluid edge and a patch face
        bmd.edges.add(Edge("arc", "vf0", "vf1", coords=[0.5, -0.1, 0.0]))
        fluid_face = Face(["vf0", "vf1", "vf2", "vf3"])
        bmd.boundary.add(Patch("walls", type="wall", faces=[fluid_face]))

        split_blockmeshdict_by_zones(bmd, tmp_path, reindex_vertices=True)
        out_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")

        # All vertex names must match v<int>
        import re
        v_pattern = re.compile(r"^v\d+$")
        for v in out_bmd.vertices:
            assert v_pattern.match(v.name), f"Vertex {v.name!r} not in vN form"

        # Blocks must reference only known vertex names
        known_names = {v.name for v in out_bmd.vertices}
        for block in out_bmd.blocks:
            for vname in block.vertices:
                assert vname in known_names, f"Block references unknown vertex {vname!r}"

        # Edges must reference only known vertex names
        for edge in out_bmd.edges:
            assert edge.v_start in known_names
            assert edge.v_end in known_names

        # Face vertices must reference only known vertex names
        for patch in out_bmd.boundary:
            for face in patch.faces:
                for vname in face.vertices:
                    assert vname in known_names

        # Vertices are compact 0..N-1
        indices = sorted(int(v.name[1:]) for v in out_bmd.vertices)
        assert indices == list(range(len(indices)))


# ---------------------------------------------------------------------------
# Test 19: defaultPatch and geometry copied verbatim
# ---------------------------------------------------------------------------


class TestVerbatimCopy:

    def test_default_patch_and_geometry_copied_verbatim(self, tmp_path: Path):
        """defaultPatch name/type and geometry_body are identical to source."""
        bmd = BlockMeshDict()
        bmd.default_patch.name = "myDefault"
        bmd.default_patch.type = "symmetryPlane"
        bmd.geometry_body = "    sphere s1 { type sphere; radius 1; }"
        _add_cube_block(bmd, "b_fluid", "vf", zone="fluid")

        split_blockmeshdict_by_zones(bmd, tmp_path)
        out_bmd = BlockMeshDict(tmp_path / "blockMeshDict_fluid")

        assert out_bmd.default_patch.name == "myDefault"
        assert out_bmd.default_patch.type == "symmetryPlane"
        assert "sphere s1" in out_bmd.geometry_body


# ---------------------------------------------------------------------------
# Test: Path-traversal containment guard
# ---------------------------------------------------------------------------


class TestPathTraversalGuard:

    def test_zone_name_with_traversal_chars_stays_inside_output_dir(
        self, tmp_path: Path
    ):
        """A zone name containing path separators is sanitised; output stays inside output_dir."""
        bmd = BlockMeshDict()
        # Zone name with path-traversal characters
        _add_cube_block(bmd, "b_evil", "ve", zone="../evil_zone")
        written = split_blockmeshdict_by_zones(bmd, tmp_path)
        assert len(written) == 1
        resolved = written[0].resolve()
        assert tmp_path.resolve() in resolved.parents, (
            f"Output file {resolved} escaped the output directory {tmp_path.resolve()}"
        )

    def test_zone_name_with_backslash_stays_inside_output_dir(
        self, tmp_path: Path
    ):
        """A zone name with backslash-like chars is sanitised; output stays inside output_dir."""
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b_bs", "vb", zone="zone\\name")
        written = split_blockmeshdict_by_zones(bmd, tmp_path)
        assert len(written) == 1
        resolved = written[0].resolve()
        assert tmp_path.resolve() in resolved.parents, (
            f"Output file {resolved} escaped the output directory {tmp_path.resolve()}"
        )

    def test_distinct_zones_sanitising_to_same_name_do_not_overwrite(
        self, tmp_path: Path
    ):
        """Two zones differing only in an illegal char must not collide.

        ``inlet-1`` and ``inlet.1`` both sanitise to ``inlet_1``.  Without a
        uniqueness backstop the second write would silently overwrite the
        first; this test guards against that data-loss regression.
        """
        bmd = BlockMeshDict()
        _add_cube_block(bmd, "b1", "va", zone="inlet-1", x_offset=0.0)
        _add_cube_block(bmd, "b2", "vb", zone="inlet.1", x_offset=1.0)

        written = split_blockmeshdict_by_zones(bmd, tmp_path)

        # Two distinct output files — no overwrite, no duplicate paths.
        assert len(written) == 2
        assert len(set(written)) == 2
        assert len({p.name for p in written}) == 2

        # Both files physically exist inside the output dir.
        on_disk = {p.name for p in tmp_path.iterdir() if p.is_file()}
        for p in written:
            assert p.name in on_disk
            assert tmp_path.resolve() in p.resolve().parents


# ---------------------------------------------------------------------------
# Test 20: Source BMD is not mutated
# ---------------------------------------------------------------------------


class TestSourceImmutability:

    def test_source_bmd_is_not_mutated(self, tmp_path: Path):
        """The source BlockMeshDict is not modified by the split operation."""
        bmd = _make_simple_bmd()
        # Capture snapshot before split
        orig_block_zones = {b.name: b.zone for b in bmd.blocks}
        orig_vertex_names = [v.name for v in bmd.vertices]
        orig_n_patches = len(list(bmd.boundary))

        split_blockmeshdict_by_zones(bmd, tmp_path)

        # Verify zone attributes unchanged
        for block in bmd.blocks:
            assert block.zone == orig_block_zones[block.name], (
                f"Block {block.name!r} zone was mutated"
            )
        # Verify vertex list unchanged
        assert [v.name for v in bmd.vertices] == orig_vertex_names
        # Verify boundary unchanged
        assert len(list(bmd.boundary)) == orig_n_patches
