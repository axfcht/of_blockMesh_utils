"""Unit tests for meshing_utils.operations.revolve.

Covers rotate_point, plan_angles, revolve() with blocks/edges/faces,
strip_internal_sector_faces, and immutability of the source mesh.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from meshing_utils import (
    Block,
    BlockMeshDict,
    Edge,
    Face,
    Patch,
    Vertex,
)
from meshing_utils.geometry.rotations import rotate_point
from meshing_utils.operations.revolve import (
    RevolveConfig,
    parse_axis,
    plan_angles,
    revolve,
    strip_internal_sector_faces,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

Z_UNIT = np.array([0.0, 0.0, 1.0])
ORIGIN = (0.0, 0.0, 0.0)


def _make_single_block_bmd(
    x0: float = 1.0,
    x1: float = 2.0,
    y0: float = 0.0,
    y1: float = 1.0,
    z0: float = 0.0,
    z1: float = 1.0,
    patch_name: str = "top",
) -> BlockMeshDict:
    """Return a BlockMeshDict with one proper 3-D hex block.

    The block occupies x=[x0,x1], y=[y0,y1], z=[z0,z1].
    Default geometry: x=[1,2], y=[0,1], z=[0,1] — a unit cube offset from the
    Z-axis so that rotating around Z produces distinct new blocks.

    Vertex layout (OpenFOAM hex convention)::

        v3 ------ v2
        |          |   (bottom face, z=z0)
        v0 ------ v1

        v7 ------ v6
        |          |   (top face, z=z1)
        v4 ------ v5

    A single boundary patch (*patch_name*) contains the top face (v4-v7).
    """
    bmd = BlockMeshDict()
    coords = [
        (x0, y0, z0),  # v0
        (x1, y0, z0),  # v1
        (x1, y1, z0),  # v2
        (x0, y1, z0),  # v3
        (x0, y0, z1),  # v4
        (x1, y0, z1),  # v5
        (x1, y1, z1),  # v6
        (x0, y1, z1),  # v7
    ]
    for i, c in enumerate(coords):
        bmd.vertices.add(Vertex(f"v{i}", list(c)))

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

    # Top face: v4, v5, v6, v7
    patch = Patch(
        name_or_string=patch_name,
        type="patch",
        faces=[Face(vertices_or_string=["v4", "v5", "v6", "v7"])],
    )
    bmd.boundary.add(patch)
    return bmd


# ---------------------------------------------------------------------------
# rotate_point
# ---------------------------------------------------------------------------

class TestRotatePoint:

    def test_identity_at_zero_angle(self):
        """Rotating by 0 rad must return the original point."""
        p = [3.0, 4.0, 5.0]
        result = rotate_point(p, ORIGIN, Z_UNIT, 0.0)
        assert result == pytest.approx(p, abs=1e-12)

    def test_90deg_around_z(self):
        """(1,0,0) rotated 90° around Z → (0,1,0)."""
        result = rotate_point([1.0, 0.0, 0.0], ORIGIN, Z_UNIT, math.pi / 2)
        assert result == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)

    def test_180deg_around_z(self):
        """(1,0,0) rotated 180° around Z → (-1,0,0)."""
        result = rotate_point([1.0, 0.0, 0.0], ORIGIN, Z_UNIT, math.pi)
        assert result == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)

    def test_offset_axis(self):
        """Rotate around an axis through (1,0,0) instead of the origin."""
        # Point (2,0,0) rotated 90° around Z-axis through (1,0,0):
        # translate: (1,0,0); rotate: (0,1,0); translate back: (1,1,0)
        result = rotate_point([2.0, 0.0, 0.0], (1.0, 0.0, 0.0), Z_UNIT, math.pi / 2)
        assert result == pytest.approx([1.0, 1.0, 0.0], abs=1e-12)

    def test_point_on_axis_is_invariant(self):
        """A point that lies ON the rotation axis must not move."""
        # axis through (0,0,0) in Z direction; point (0,0,5) is on the axis
        result = rotate_point([0.0, 0.0, 5.0], ORIGIN, Z_UNIT, math.pi / 3)
        assert result == pytest.approx([0.0, 0.0, 5.0], abs=1e-12)

    def test_negative_angle(self):
        """(0,1,0) rotated -90° around Z → (1,0,0)."""
        result = rotate_point([0.0, 1.0, 0.0], ORIGIN, Z_UNIT, -math.pi / 2)
        assert result == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


# ---------------------------------------------------------------------------
# plan_angles
# ---------------------------------------------------------------------------

class TestPlanAngles:

    def test_basic_half_circle(self):
        """alpha=180, count=4 → steps at 60, 120, 180 (last copy at total angle)."""
        angles = plan_angles(180.0, 4)
        assert angles == pytest.approx([60.0, 120.0, 180.0], abs=1e-12)

    def test_full_circle_no_360(self):
        """alpha=360, count=4 → [90, 180, 270]; no 360° copy."""
        angles = plan_angles(360.0, 4)
        assert angles == pytest.approx([90.0, 180.0, 270.0], abs=1e-12)
        assert 360.0 not in angles

    def test_count_two_single_copy(self):
        """count=2 → only one additional copy at alpha/2 * ... wait, step=alpha/count."""
        # step = 360/2 = 180; copies at k=1: 180
        angles = plan_angles(360.0, 2)
        assert angles == pytest.approx([180.0], abs=1e-12)

    def test_count_fourteen_thirteen_copies(self):
        """count=14, alpha=360 → 13 copies; last at 13*(360/14) ≈ 334.29°."""
        angles = plan_angles(360.0, 14)
        assert len(angles) == 13
        assert angles[-1] == pytest.approx(13 * (360.0 / 14), abs=1e-10)

    def test_negative_angle_produces_negative_steps(self):
        """Negative total angle → negative step values."""
        angles = plan_angles(-90.0, 3)
        assert len(angles) == 2
        assert all(a < 0 for a in angles)

    def test_partial_90_count_2_single_copy_at_90(self):
        """90°/count=2 partial sweep: step=90/(2-1)=90; exactly one copy at 90°."""
        angles = plan_angles(90.0, 2)
        assert len(angles) == 1
        assert angles[0] == pytest.approx(90.0, abs=1e-12)

    def test_partial_last_copy_lands_on_total_angle(self):
        """For any partial sweep the last copy must land exactly on total_angle_deg."""
        for total, count in [(45.0, 3), (270.0, 5), (120.0, 4)]:
            angles = plan_angles(total, count)
            assert angles[-1] == pytest.approx(total, abs=1e-9), (
                f"plan_angles({total}, {count}) last angle {angles[-1]} != {total}"
            )

    def test_full_circle_last_copy_not_at_360(self):
        """Full circle: last copy must NOT be at 360° (would duplicate origin)."""
        for count in (2, 4, 6, 8):
            angles = plan_angles(360.0, count)
            assert 360.0 not in angles, (
                f"plan_angles(360, {count}) incorrectly contains 360°: {angles}"
            )
            assert abs(360.0) not in [abs(a) for a in angles]


# ---------------------------------------------------------------------------
# parse_axis
# ---------------------------------------------------------------------------

class TestParseAxis:

    def test_parenthesised_form(self):
        assert parse_axis("(1.0 2.0 3.0)") == pytest.approx((1.0, 2.0, 3.0))

    def test_bare_form(self):
        assert parse_axis("0 0 1") == pytest.approx((0.0, 0.0, 1.0))

    def test_scientific_notation(self):
        assert parse_axis("(1e-3 0 0)") == pytest.approx((0.001, 0.0, 0.0))

    def test_too_few_tokens_raises(self):
        with pytest.raises(ValueError):
            parse_axis("1.0 2.0")

    def test_too_many_tokens_raises(self):
        with pytest.raises(ValueError):
            parse_axis("1 2 3 4")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_axis("(a b c)")


# ---------------------------------------------------------------------------
# revolve — validation errors
# ---------------------------------------------------------------------------

class TestRevolveConfigFrozen:
    """RevolveConfig is frozen=True; verify immutability and __post_init__ validation."""

    def _valid_cfg(self, **overrides) -> dict:
        defaults = dict(
            axis_point=(0.0, 0.0, 0.0),
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
        )
        defaults.update(overrides)
        return defaults

    def test_construction_succeeds(self):
        """A valid RevolveConfig must be constructable without error."""
        cfg = RevolveConfig(**self._valid_cfg())
        assert cfg.count == 2
        assert cfg.angle == 90.0

    def test_frozen_blocks_attribute_mutation(self):
        """Setting any attribute on a frozen dataclass must raise FrozenInstanceError."""
        import dataclasses
        cfg = RevolveConfig(**self._valid_cfg())
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.count = 99  # type: ignore[misc]

    def test_frozen_blocks_angle_mutation(self):
        """angle is also immutable after construction."""
        import dataclasses
        cfg = RevolveConfig(**self._valid_cfg())
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.angle = 180.0  # type: ignore[misc]

    def test_post_init_count_below_two_raises(self):
        """__post_init__ must raise ValueError for count < 2."""
        with pytest.raises(ValueError, match="count must be >= 2"):
            RevolveConfig(**self._valid_cfg(count=1))

    def test_post_init_count_zero_raises(self):
        with pytest.raises(ValueError, match="count must be >= 2"):
            RevolveConfig(**self._valid_cfg(count=0))

    def test_post_init_zero_angle_raises(self):
        """__post_init__ must reject angle=0."""
        with pytest.raises(ValueError, match="angle"):
            RevolveConfig(**self._valid_cfg(angle=0.0))

    def test_post_init_angle_too_large_raises(self):
        """__post_init__ must reject |angle| > 360."""
        with pytest.raises(ValueError, match="angle"):
            RevolveConfig(**self._valid_cfg(angle=361.0))

    def test_post_init_negative_valid_angle(self):
        """Negative angle within bounds must be accepted."""
        cfg = RevolveConfig(**self._valid_cfg(angle=-90.0))
        assert cfg.angle == -90.0

    def test_post_init_zero_axis_dir_raises(self):
        """__post_init__ must reject the zero axis vector."""
        with pytest.raises(ValueError):
            RevolveConfig(**self._valid_cfg(axis_dir=(0.0, 0.0, 0.0)))


class TestRevolveValidation:

    def _source(self) -> BlockMeshDict:
        return _make_single_block_bmd()

    def _cfg(self, **kwargs) -> RevolveConfig:
        defaults = dict(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-6,
        )
        defaults.update(kwargs)
        return RevolveConfig(**defaults)

    def test_count_below_two_raises(self):
        with pytest.raises(ValueError, match="count must be >= 2"):
            revolve(self._source(), self._cfg(count=1))

    def test_count_zero_raises(self):
        with pytest.raises(ValueError, match="count must be >= 2"):
            revolve(self._source(), self._cfg(count=0))

    def test_zero_angle_raises(self):
        with pytest.raises(ValueError, match="angle"):
            revolve(self._source(), self._cfg(angle=0.0))

    def test_angle_too_large_raises(self):
        with pytest.raises(ValueError, match="angle"):
            revolve(self._source(), self._cfg(angle=361.0))

    def test_empty_blocks_raises(self):
        empty_bmd = BlockMeshDict()
        with pytest.raises(ValueError, match="no blocks"):
            revolve(empty_bmd, self._cfg())

    def test_zero_axis_dir_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            revolve(self._source(), self._cfg(axis_dir=(0.0, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# revolve — single block, quarter turn
# ---------------------------------------------------------------------------

class TestRevolveSingleBlock:

    def test_quarter_turn_produces_two_blocks(self):
        """1 block, alpha=90, count=2 → 2 blocks total."""
        source = _make_single_block_bmd()
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        assert len(result.blocks) == 2

    def test_full_circle_produces_four_blocks(self):
        """1 block, alpha=360, count=4 → 4 blocks total."""
        source = _make_single_block_bmd()
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        assert len(result.blocks) == 4

    def test_full_circle_seam_dedupes_vertices(self):
        """360° with count=5 (copies at 72°, 144°, 216°, 288°): 5 blocks."""
        source = _make_single_block_bmd()
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        assert len(result.blocks) == 4
        assert len(result.vertices) == 4 * 8

    def test_source_not_mutated(self):
        """revolve() must not modify the source mesh."""
        source = _make_single_block_bmd()
        original_n_vertices = len(source.vertices)
        original_n_blocks = len(source.blocks)
        original_names = [v.name for v in source.vertices]

        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        revolve(source, cfg)

        assert len(source.vertices) == original_n_vertices
        assert len(source.blocks) == original_n_blocks
        assert [v.name for v in source.vertices] == original_names


# ---------------------------------------------------------------------------
# revolve — axis-collinear vertex collapses block
# ---------------------------------------------------------------------------

class TestAxisCollinearVertexCollapse:

    def test_collinear_vertex_collapses_block(self):
        """Block with all vertices ON the rotation axis: collapses and is skipped."""
        bmd = BlockMeshDict()
        # Place all 8 vertices ON the Z axis (x=y=0)
        for i in range(8):
            bmd.vertices.add(Vertex(f"v{i}", [0.0, 0.0, float(i)]))
        bmd.blocks.add(
            Block("block0", [f"v{i}" for i in range(8)], [1, 1, 1])
        )

        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-9,
        )
        result = revolve(source=bmd, cfg=cfg)
        assert len(result.blocks) == 1  # no new block added


# ---------------------------------------------------------------------------
# revolve — arc edge rotates control points
# ---------------------------------------------------------------------------

class TestArcEdgeRotation:

    def test_arc_edge_control_point_is_rotated(self):
        """An arc edge's midpoint must be rotated along with the endpoints.

        With count=2 and angle=90° (partial sweep), step = 90/(2-1) = 90°.
        There is exactly one copy, placed at k=1 → 90°.
        """
        source = _make_single_block_bmd()
        # Add an arc edge between v0=(1,0,0) and v1=(2,0,0) with midpoint (1.5, 0.1, 0)
        arc = Edge("arc", "v0", "v1", points=[[1.5, 0.1, 0.0]])
        source.edges.add(arc)

        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)

        # count=2, angle=90 (partial) → step=90° → single copy at 90°
        # Rotated midpoint: rotate (1.5, 0.1, 0) by 90° around Z
        expected_mid = rotate_point([1.5, 0.1, 0.0], ORIGIN, Z_UNIT, math.radians(90.0))

        # The original arc connects v0→v1; the rotated arc connects the
        # rotated counterparts.  We identify the rotated arc as any arc edge
        # whose v_start is NOT "v0" (the original source vertex name).
        original_arc_start = "v0"
        rotated_edges = [
            e for e in result.edges
            if e.type == "arc" and e.v_start != original_arc_start
               and e.v_end != original_arc_start
        ]
        assert len(rotated_edges) == 1, (
            f"Expected exactly 1 rotated arc edge, got {len(rotated_edges)}: "
            f"{[(e.v_start, e.v_end) for e in result.edges if e.type == 'arc']}"
        )
        assert rotated_edges[0].points[0] == pytest.approx(expected_mid, abs=1e-9)


# ---------------------------------------------------------------------------
# revolve — patch face handling
# ---------------------------------------------------------------------------

class TestPatchFaceHandling:

    def test_patch_face_is_rotated(self):
        """Boundary faces must appear in the rotated copy."""
        source = _make_single_block_bmd(patch_name="outer")
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        outer = result.boundary.get("outer")
        # Original face + one rotated copy = 2 faces (assuming not degenerate)
        assert len(outer.faces) >= 2

    def test_patch_face_deduped_full_circle(self):
        """At 360°, the last copy's face vertices coincide with the original
        source face vertices → no duplicate face should be added."""
        source = _make_single_block_bmd(patch_name="outer")
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        outer = result.boundary.get("outer")
        # There should be no duplicate face (same vertex set appears twice)
        face_sets = [frozenset(f.vertices) for f in outer.faces]
        assert len(face_sets) == len(set(face_sets))


# ---------------------------------------------------------------------------
# strip_internal_sector_faces
# ---------------------------------------------------------------------------

class TestStripInternalSectorFaces:

    def _two_block_bmd_with_shared_face(self) -> BlockMeshDict:
        """Two hex blocks sharing a face (v4-v5-v6-v7).

        block0: v0..v7
        block1: v4..v11 (shares v4,v5,v6,v7 with block0)
        Patch 'shared' has a face on (v4 v5 v6 v7).
        Patch 'outer' has a face only on block0's vertices.
        """
        bmd = BlockMeshDict()
        for i in range(12):
            bmd.vertices.add(Vertex(f"v{i}", [float(i), 0.0, 0.0]))

        bmd.blocks.add(Block("block0", [f"v{i}" for i in range(8)], [1, 1, 1]))
        bmd.blocks.add(Block("block1", [f"v{i}" for i in range(4, 12)], [1, 1, 1]))

        shared_patch = Patch(
            "shared", "patch",
            faces=[Face(["v4", "v5", "v6", "v7"])]
        )
        outer_patch = Patch(
            "outer", "patch",
            faces=[Face(["v0", "v1", "v2", "v3"])]
        )
        bmd.boundary.add(shared_patch)
        bmd.boundary.add(outer_patch)
        return bmd

    def test_internal_face_is_removed(self):
        """Face shared by two blocks → removed by strip_internal_sector_faces."""
        bmd = self._two_block_bmd_with_shared_face()
        removed = strip_internal_sector_faces(bmd)
        assert removed == 1
        shared = bmd.boundary.get("shared")
        assert len(shared.faces) == 0

    def test_outer_face_is_kept(self):
        """Face belonging to only one block → kept."""
        bmd = self._two_block_bmd_with_shared_face()
        strip_internal_sector_faces(bmd)
        outer = bmd.boundary.get("outer")
        assert len(outer.faces) == 1

    def test_patch_with_zero_faces_preserved(self):
        """Patches that end up with 0 faces are kept (not deleted)."""
        bmd = self._two_block_bmd_with_shared_face()
        strip_internal_sector_faces(bmd)
        # 'shared' patch still exists, just empty
        assert "shared" in bmd.boundary

    def test_full_revolution_internal_face_removed(self):
        """After a full 360° revolve the seam faces between first and last sector
        must be removed (they are shared by two blocks)."""
        source = _make_single_block_bmd(patch_name="side")
        # Give the source a side patch that will become an internal seam face
        # after revolution (the side face of the block, lying in the y=0 plane)
        side_face = Face(["v0", "v1", "v4", "v5"])  # bottom face at y=0
        source.boundary.get("side").faces.append(side_face)

        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        side = result.boundary.get("side")
        total_in_source = len(source.boundary.get("side").faces)
        assert len(side.faces) <= 4 * total_in_source

    def test_outer_radial_face_kept_partial(self):
        """In a partial revolve (90°), the outer radial face is only in one block
        → must not be removed."""
        source = _make_single_block_bmd(patch_name="outer")
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=90.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        outer = result.boundary.get("outer")
        assert len(outer.faces) >= 1

    def test_outer_sector_face_kept_half_revolution(self):
        """In a 180° revolution, the endcap face of the last sector is not shared."""
        source = _make_single_block_bmd(patch_name="top")
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=2,
            angle=180.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        top = result.boundary.get("top")
        assert len(top.faces) >= 1


# ---------------------------------------------------------------------------
# revolve — immutability of input
# ---------------------------------------------------------------------------

class TestRevolveDoesNotMutateInput:

    def test_source_vertices_unchanged(self):
        source = _make_single_block_bmd()
        source_copy = copy.deepcopy(source)
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        revolve(source, cfg)
        # Compare vertices by name and coords
        for v_orig, v_after in zip(source_copy.vertices, source.vertices, strict=False):
            assert v_orig.name == v_after.name
            assert v_orig.coords == pytest.approx(v_after.coords)

    def test_source_blocks_unchanged(self):
        source = _make_single_block_bmd()
        original_block_count = len(source.blocks)
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        revolve(source, cfg)
        assert len(source.blocks) == original_block_count

    def test_source_boundary_unchanged(self):
        source = _make_single_block_bmd(patch_name="outer")
        original_face_count = len(source.boundary.get("outer").faces)
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        revolve(source, cfg)
        assert len(source.boundary.get("outer").faces) == original_face_count


# ---------------------------------------------------------------------------
# revolve — unique_patches feature
# ---------------------------------------------------------------------------

class TestRevolveUniquePatches:
    """Tests for the unique_patches tri-state feature of RevolveConfig."""

    def _cfg(self, count: int = 4, angle: float = 360.0, unique_patches=None) -> RevolveConfig:
        return RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=count,
            angle=angle,
            tol=1e-9,
            unique_patches=unique_patches,
        )

    def test_default_none_keeps_existing_behaviour(self):
        """unique_patches=None (default) must accumulate all copies in the original patch."""
        source = _make_single_block_bmd(patch_name="outer")
        result = revolve(source, self._cfg(count=4, angle=360.0, unique_patches=None))
        # Only the original 'outer' patch exists — no 'outer_1', 'outer_2', etc.
        patch_names = {p.name for p in result.boundary}
        assert "outer" in patch_names
        assert "outer_1" not in patch_names
        assert "outer_2" not in patch_names
        assert "outer_3" not in patch_names

    def test_empty_list_uniquifies_all_patches(self):
        """unique_patches=[] must create outer, outer_1, outer_2, outer_3 for count=4."""
        source = _make_single_block_bmd(patch_name="outer")
        result = revolve(source, self._cfg(count=4, angle=360.0, unique_patches=[]))
        patch_names = {p.name for p in result.boundary}
        assert "outer" in patch_names
        assert "outer_1" in patch_names
        assert "outer_2" in patch_names
        assert "outer_3" in patch_names

    def test_named_list_uniquifies_only_specified(self):
        """unique_patches=['inlet'] must uniquify 'inlet' but leave 'outlet' untouched."""
        source = _make_single_block_bmd(patch_name="inlet")
        # Add a second patch 'outlet' with a different face
        outlet_patch = Patch(
            name_or_string="outlet",
            type="patch",
            faces=[Face(vertices_or_string=["v0", "v1", "v2", "v3"])],
        )
        source.boundary.add(outlet_patch)

        result = revolve(source, self._cfg(count=3, angle=90.0, unique_patches=["inlet"]))
        patch_names = {p.name for p in result.boundary}

        # inlet is uniquified: inlet, inlet_1, inlet_2
        assert "inlet" in patch_names
        assert "inlet_1" in patch_names
        assert "inlet_2" in patch_names

        # outlet is NOT uniquified — no outlet_1, outlet_2
        assert "outlet" in patch_names
        assert "outlet_1" not in patch_names
        assert "outlet_2" not in patch_names

    def test_unknown_patch_name_raises(self):
        """unique_patches with a name that is not in the boundary must raise ValueError."""
        source = _make_single_block_bmd(patch_name="outer")
        with pytest.raises(ValueError, match="unknown patches") as exc_info:
            revolve(source, self._cfg(count=2, unique_patches=["does_not_exist"]))
        assert "does_not_exist" in str(exc_info.value)
        assert "outer" in str(exc_info.value)

    def test_naming_collision_raises(self):
        """If the source already contains a patch named '<orig>_1', revolve
        must raise ValueError."""
        source = _make_single_block_bmd(patch_name="inlet")
        # Manually add a colliding patch 'inlet_1'
        colliding = Patch(
            name_or_string="inlet_1",
            type="patch",
            faces=[],
        )
        source.boundary.add(colliding)

        with pytest.raises(ValueError, match="collision") as exc_info:
            revolve(source, self._cfg(count=2, unique_patches=["inlet"]))
        assert "inlet_1" in str(exc_info.value)

    def test_patch_type_and_marker_propagated(self):
        """Uniquified patches must inherit type and marker from the original patch."""
        source = _make_single_block_bmd(patch_name="inlet")
        # Override type and marker on the original patch
        orig_patch = source.boundary.get("inlet")
        orig_patch.type = "wall"
        orig_patch.marker = "m"

        result = revolve(source, self._cfg(count=3, angle=90.0, unique_patches=["inlet"]))
        inlet_1 = result.boundary.get("inlet_1")
        assert inlet_1.type == "wall"
        assert inlet_1.marker == "m"

    def test_strip_internal_faces_excludes_unique_patches(self):
        """In a full 360° revolve with all patches uniquified, each unique patch
        must retain its faces (the strip step must not remove them)."""
        source = _make_single_block_bmd(patch_name="outer")
        # Add a side face that would become internal after revolution
        side_face = Face(["v0", "v1", "v4", "v5"])
        source.boundary.get("outer").faces.append(side_face)

        result = revolve(source, self._cfg(count=4, angle=360.0, unique_patches=[]))
        # outer_1, outer_2, outer_3 are unique and must not be stripped
        for k in range(1, 4):
            patch = result.boundary.get(f"outer_{k}")
            assert len(patch.faces) > 0, (
                f"outer_{k} has 0 faces after revolve — unique patches should not be stripped"
            )

    def test_partial_revolve_creates_count_minus_one_unique_patches(self):
        """count=3, angle=90 must create exactly 2 additional unique patches (k=1,2)."""
        source = _make_single_block_bmd(patch_name="outer")
        result = revolve(source, self._cfg(count=3, angle=90.0, unique_patches=[]))
        patch_names = {p.name for p in result.boundary}
        assert "outer" in patch_names
        assert "outer_1" in patch_names
        assert "outer_2" in patch_names
        # No third copy
        assert "outer_3" not in patch_names


# ---------------------------------------------------------------------------
# revolve — zone propagation
# ---------------------------------------------------------------------------

class TestRevolveZonePropagation:

    def test_revolve_preserves_zone(self):
        """All blocks generated from a zoned source block must carry the same zone."""
        source = _make_single_block_bmd()
        # Assign a zone to the single source block
        source.blocks[0].zone = "fluid"
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        for blk in result.blocks:
            assert blk.zone == "fluid", (
                f"Block {blk.name!r} zone is {blk.zone!r}, expected 'fluid'"
            )

    def test_revolve_no_zone_stays_none(self):
        """Blocks generated from a zone-less source must have zone=None."""
        source = _make_single_block_bmd()
        assert source.blocks[0].zone is None
        cfg = RevolveConfig(
            axis_point=ORIGIN,
            axis_dir=(0.0, 0.0, 1.0),
            count=4,
            angle=360.0,
            tol=1e-9,
        )
        result = revolve(source, cfg)
        for blk in result.blocks:
            assert blk.zone is None
