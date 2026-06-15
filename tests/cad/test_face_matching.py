"""Unit tests for meshing_utils.cad.face_matching.

Migrated from tests/common/test_face_geometry.py.
OCP-dependent tests are guarded by the ``requires_ocp`` skip marker.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from meshing_utils import (
    BlockFace,
    compute_outward_normal,
    effective_normal_tolerance,
    extract_block_faces,
    find_dominant_face,
    nearest_face_within_tol,
    normals_consistent,
    surface_type_of,
)
from meshing_utils.cad.face_matching import local_surface_normal
from meshing_utils.cad.face_matching.matching import (
    _build_face_ancestor_maps,
    _recover_faces,
)
from meshing_utils.geometry.hex_topology import HEX_FACE_INDICES

# Skip marker for tests that build real OCP (cadquery-ocp) shapes. These are
# skipped when OCP is not importable rather than erroring.
requires_ocp = pytest.mark.skipif(
    importlib.util.find_spec("OCP") is None,
    reason="requires OCP (cadquery-ocp); not installed in this environment",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < 1e-15:
        return v
    return (v[0] / length, v[1] / length, v[2] / length)


def _make_bmd_with_cube(tol: float = 1e-6):
    """Return a BlockMeshDict with a single unit-cube hex block."""
    from meshing_utils import Block, BlockMeshDict

    bmd = BlockMeshDict()
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    names = []
    for c in coords:
        n = bmd.find_or_add_vertex(c, tol=tol)
        names.append(n)

    block = Block(
        name_or_string="testBlock",
        vertices=names,
        cells=[1, 1, 1],
    )
    bmd.blocks.add(block)
    return bmd, names


# ---------------------------------------------------------------------------
# compute_outward_normal
# ---------------------------------------------------------------------------

class TestComputeOutwardNormal:
    def test_xy_plane_points_in_z(self):
        """A quad in the XY plane should produce a normal along Z."""
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        n = compute_outward_normal(coords)
        u = _unit(n)
        assert abs(abs(u[2]) - 1.0) < 1e-9

    def test_xz_plane(self):
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ]
        n = compute_outward_normal(coords)
        u = _unit(n)
        assert abs(abs(u[1]) - 1.0) < 1e-9

    def test_degenerate_triangle_not_crash(self):
        """Degenerate (zero-area) polygon should not raise."""
        coords = [(0.0, 0.0, 0.0)] * 4
        n = compute_outward_normal(coords)
        assert isinstance(n, tuple)
        assert len(n) == 3

    def test_reversed_winding_flips_sign(self):
        """Reversing the vertex order should negate the normal."""
        coords = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        n_fwd = compute_outward_normal(coords)
        n_rev = compute_outward_normal(list(reversed(coords)))
        dot = n_fwd[0] * n_rev[0] + n_fwd[1] * n_rev[1] + n_fwd[2] * n_rev[2]
        assert dot < 0.0


# ---------------------------------------------------------------------------
# normals_consistent
# ---------------------------------------------------------------------------

class TestNormalsConsistent:
    def test_identical_normals(self):
        n = (0.0, 0.0, 1.0)
        assert normals_consistent(n, n) is True

    def test_anti_parallel_normals(self):
        n1 = (0.0, 0.0, 1.0)
        n2 = (0.0, 0.0, -1.0)
        assert normals_consistent(n1, n2) is True

    def test_perpendicular_normals(self):
        n1 = (1.0, 0.0, 0.0)
        n2 = (0.0, 1.0, 0.0)
        assert normals_consistent(n1, n2) is False

    def test_within_tol(self):
        angle_rad = math.radians(3.0)
        n1 = (0.0, 0.0, 1.0)
        n2 = (math.sin(angle_rad), 0.0, math.cos(angle_rad))
        assert normals_consistent(n1, n2, angle_tol_deg=5.0) is True

    def test_outside_tol(self):
        angle_rad = math.radians(10.0)
        n1 = (0.0, 0.0, 1.0)
        n2 = (math.sin(angle_rad), 0.0, math.cos(angle_rad))
        assert normals_consistent(n1, n2, angle_tol_deg=5.0) is False

    def test_near_180_within_tol(self):
        """Slightly off 180° should be consistent."""
        angle_rad = math.radians(178.0)
        n1 = (0.0, 0.0, 1.0)
        n2 = (math.sin(angle_rad), 0.0, math.cos(angle_rad))
        assert normals_consistent(n1, n2, angle_tol_deg=5.0) is True

    def test_zero_vector_returns_false(self):
        assert normals_consistent((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is False

    def test_non_unit_vectors(self):
        """Should work with non-unit vectors."""
        n1 = (0.0, 0.0, 5.0)
        n2 = (0.0, 0.0, 3.0)
        assert normals_consistent(n1, n2) is True


# ---------------------------------------------------------------------------
# HEX_FACE_INDICES
# ---------------------------------------------------------------------------

class TestHexFaceLocalIndices:
    def test_six_faces(self):
        assert len(HEX_FACE_INDICES) == 6

    def test_each_face_four_vertices(self):
        for face in HEX_FACE_INDICES:
            assert len(face) == 4

    def test_indices_in_range(self):
        for face in HEX_FACE_INDICES:
            for idx in face:
                assert 0 <= idx <= 7


# ---------------------------------------------------------------------------
# extract_block_faces
# ---------------------------------------------------------------------------

class TestExtractBlockFaces:
    def test_cube_produces_six_faces(self):
        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)
        assert len(faces) == 6

    def test_face_names(self):
        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)
        face_names = {f.face_name for f in faces}
        assert face_names == {"imin", "imax", "jmin", "jmax", "kmin", "kmax"}

    def test_each_face_four_vertices(self):
        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)
        for f in faces:
            assert len(f.vertex_names) == 4
            assert len(f.vertex_coords) == 4

    def test_block_name_propagated(self):
        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)
        for f in faces:
            assert f.block_name == "testBlock"

    def test_block_with_less_than_8_vertices_returns_empty(self):
        from meshing_utils import Block, BlockMeshDict

        bmd = BlockMeshDict()
        block = Block(name_or_string="shortBlock", vertices=["v0", "v1", "v2"])
        result = extract_block_faces(block, bmd)
        assert result == []

    def test_support_points_empty_without_edges(self):
        """When no curved edges exist, support_points should be empty."""
        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)
        for f in faces:
            assert f.support_points == []


# ---------------------------------------------------------------------------
# OCC-dependent tests
# ---------------------------------------------------------------------------

@requires_ocp
class TestNearestFaceWithinTol:
    def test_cube_face_matches_box_solid(self):
        """All 6 faces of a unit-cube block should match the corresponding
        box solid faces within a reasonable tolerance."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        bmd, _names = _make_bmd_with_cube()
        block = next(iter(bmd.blocks))
        faces = extract_block_faces(block, bmd)

        box_solid = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        matched = 0
        for bf in faces:
            result = nearest_face_within_tol(bf, box_solid, tol=1e-3)
            if result is not None:
                matched += 1
        assert matched == 6

    def test_face_outside_tol_returns_none(self):
        """A block face shifted far away should not match."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box_solid = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        far_face = BlockFace(
            block_name="farBlock",
            face_index=0,
            face_name="kmin",
            vertex_names=["a", "b", "c", "d"],
            vertex_coords=[
                (10.0, 10.0, 10.0),
                (11.0, 10.0, 10.0),
                (11.0, 11.0, 10.0),
                (10.0, 11.0, 10.0),
            ],
        )
        result = nearest_face_within_tol(far_face, box_solid, tol=1e-3)
        assert result is None

    def test_cylinder_face_not_matched_as_planar(self):
        """A planar block face should NOT match the curved lateral surface of a
        cylinder if the sample points are outside tolerance."""

        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        cyl_solid = BRepPrimAPI_MakeCylinder(1.0, 1.0).Shape()

        flat_face = BlockFace(
            block_name="b",
            face_index=0,
            face_name="kmin",
            vertex_names=["a", "b", "c", "d"],
            vertex_coords=[
                (5.0, 5.0, 0.5),
                (6.0, 5.0, 0.5),
                (6.0, 6.0, 0.5),
                (5.0, 6.0, 0.5),
            ],
        )
        result = nearest_face_within_tol(flat_face, cyl_solid, tol=1e-3)
        assert result is None

    def test_thin_curved_face_centroid_outside_returns_face(self):
        """Regression: thin curved block faces whose arithmetic centroid lies
        slightly outside the surface annulus (sagitta effect) must still
        return a TopoDS_Face via the majority-vote fallback.

        Reproduces the bug observed for Solid937/Solid1526 in
        blockMeshDict_mistake matched against patches_lager_komplett.stp:
        all four vertices and arc supports of the kmax face lie exactly on
        the bottom annulus of seite_1, but the centroid drifts into the
        inner hole, causing the centroid lookup to return an edge rather
        than a face.
        """
        import math as _math

        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

        # Hollow disk (annulus): outer R=12, inner R=9.9, thickness in z = 1.
        # The bottom face at z=0 is a wide annulus; we deliberately make it
        # much wider than the block-face wall so two of the block-face
        # vertices lie INTERIOR to the bottom annulus face (their support
        # shape is the face itself), while the other two sit on the inner
        # edge (their support shape is the shared edge). This mirrors the
        # real-world Solid937 geometry vs. seite_1_1.
        axis = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
        outer = BRepPrimAPI_MakeCylinder(axis, 12.0, 1.0).Shape()
        inner = BRepPrimAPI_MakeCylinder(axis, 9.9, 1.0).Shape()
        annulus = BRepAlgoAPI_Cut(outer, inner).Shape()

        # Block-face wall at R=9.9 (inner, on edge) to R=10.0 (interior of
        # bottom face). 20-degree sector: the sagitta is large enough that
        # the arithmetic centroid lies at R ~ 9.798 — inside the hole, i.e.
        # outside the bottom annulus.
        half_angle = _math.radians(10.0)
        r_in = 9.9
        r_out = 10.0
        # Order vertices so that the FIRST one lies on the inner shared edge
        # (R = annulus inner radius). This rules out the old v0-fallback as
        # the rescue mechanism — only the new per-point majority vote can
        # produce a face.
        p0 = (r_in * _math.cos(half_angle), -r_in * _math.sin(half_angle), 0.0)
        p1 = (r_in * _math.cos(half_angle), +r_in * _math.sin(half_angle), 0.0)
        p2 = (r_out * _math.cos(half_angle), +r_out * _math.sin(half_angle), 0.0)
        p3 = (r_out * _math.cos(half_angle), -r_out * _math.sin(half_angle), 0.0)
        # Sanity check our setup: centroid must lie strictly inside the hole.
        cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4
        cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4
        centroid_r = _math.sqrt(cx * cx + cy * cy)
        assert centroid_r < 9.9, f"test setup broken: centroid R={centroid_r}"
        block_face = BlockFace(
            block_name="thinCurved",
            face_index=4,
            face_name="kmin",
            vertex_names=["v0", "v1", "v2", "v3"],
            vertex_coords=[p0, p1, p2, p3],
            support_points=[],
        )

        result = nearest_face_within_tol(block_face, annulus, tol=1e-3)
        assert result is not None, (
            "centroid-edge regression: all 4 vertices sit on the annulus, "
            "function must not return None"
        )

    def test_centroid_inside_face_fast_path_unchanged(self):
        """Sanity check: when the centroid clearly lies on the interior of a
        face, the fast path returns a face and behavior is unchanged."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box_solid = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        # kmin face of unit cube: 4 vertices on z=0, centroid clearly on the
        # bottom face interior at (0.5, 0.5, 0).
        bottom_face = BlockFace(
            block_name="b",
            face_index=4,
            face_name="kmin",
            vertex_names=["v0", "v1", "v2", "v3"],
            vertex_coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            support_points=[],
        )
        result = nearest_face_within_tol(bottom_face, box_solid, tol=1e-6)
        assert result is not None

    def test_all_vertices_on_shared_edges_does_not_crash(self):
        """Edge case: 4 vertices all sit exactly on a shared edge. Function
        must not crash; it may return either a face or None."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box_solid = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        # Degenerate face: all 4 vertices on the bottom edge x=0 (shared
        # between the xmin face and the kmin face of the box).
        degenerate = BlockFace(
            block_name="b",
            face_index=0,
            face_name="imin",
            vertex_names=["v0", "v1", "v2", "v3"],
            vertex_coords=[
                (0.0, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.75, 0.0),
            ],
            support_points=[],
        )
        # Just verify it doesn't crash — result may be None or a face.
        nearest_face_within_tol(degenerate, box_solid, tol=1e-6)


class TestLocalSurfaceNormal:
    @requires_ocp
    def test_box_top_face_normal_points_up(self):
        """The top face of a unit box should have a normal close to +Z."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        exp = TopExp_Explorer(box, TopAbs_FACE)
        found_any = False
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            n = local_surface_normal(face, (0.5, 0.5, 1.0))
            if n is not None and abs(abs(n[2]) - 1.0) < 0.2:
                found_any = True
                break
            exp.Next()

        assert found_any

    def test_returns_none_on_bad_input(self):
        """local_surface_normal should return None when given a face that
        cannot be evaluated (simulated by passing None)."""
        result = local_surface_normal(None, (0.0, 0.0, 0.0))
        assert result is None


# ---------------------------------------------------------------------------
# surface_type_of
# ---------------------------------------------------------------------------

class TestSurfaceTypeOf:
    @requires_ocp
    def test_box_face_is_plane(self):
        """All faces of a MakeBox solid should be of type 'plane'."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        exp = TopExp_Explorer(box, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            assert surface_type_of(face) == "plane"
            exp.Next()

    @requires_ocp
    def test_cylinder_lateral_face_is_cylinder(self):
        """The lateral face of a MakeCylinder solid should be of type 'cylinder'."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        cyl = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
        exp = TopExp_Explorer(cyl, TopAbs_FACE)
        found_cylinder = False
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            stype = surface_type_of(face)
            if stype == "cylinder":
                found_cylinder = True
            exp.Next()
        assert found_cylinder, "Expected at least one face of type 'cylinder'"

    @requires_ocp
    def test_torus_face_is_torus(self):
        """The face of a MakeTorus solid should be of type 'torus'."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        torus = BRepPrimAPI_MakeTorus(5.0, 1.0).Shape()
        exp = TopExp_Explorer(torus, TopAbs_FACE)
        found_torus = False
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            stype = surface_type_of(face)
            if stype == "torus":
                found_torus = True
            exp.Next()
        assert found_torus, "Expected at least one face of type 'torus'"

    @requires_ocp
    def test_sphere_face_is_sphere(self):
        """The face of a MakeSphere solid should be of type 'sphere'."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        sphere = BRepPrimAPI_MakeSphere(2.0).Shape()
        exp = TopExp_Explorer(sphere, TopAbs_FACE)
        found_sphere = False
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            stype = surface_type_of(face)
            if stype == "sphere":
                found_sphere = True
            exp.Next()
        assert found_sphere, "Expected at least one face of type 'sphere'"

    def test_bad_input_returns_other(self):
        """surface_type_of should return 'other' for invalid input."""
        result = surface_type_of(None)
        assert result == "other"


# ---------------------------------------------------------------------------
# effective_normal_tolerance
# ---------------------------------------------------------------------------

class TestEffectiveNormalTolerance:
    def test_plane_returns_planar_tol(self):
        assert effective_normal_tolerance("plane", 5.0, 30.0) == 5.0

    def test_cylinder_returns_double_planar_tol(self):
        assert effective_normal_tolerance("cylinder", 5.0, 30.0) == 10.0

    def test_cone_returns_double_planar_tol(self):
        assert effective_normal_tolerance("cone", 5.0, 30.0) == 10.0

    def test_torus_returns_curved_tol(self):
        assert effective_normal_tolerance("torus", 5.0, 30.0) == 30.0

    def test_bspline_returns_curved_tol(self):
        assert effective_normal_tolerance("bspline", 5.0, 30.0) == 30.0

    def test_bezier_returns_curved_tol(self):
        assert effective_normal_tolerance("bezier", 5.0, 30.0) == 30.0

    def test_sphere_returns_curved_tol(self):
        assert effective_normal_tolerance("sphere", 5.0, 30.0) == 30.0

    def test_other_returns_curved_tol(self):
        assert effective_normal_tolerance("other", 5.0, 30.0) == 30.0

    def test_revolution_returns_curved_tol(self):
        assert effective_normal_tolerance("revolution", 5.0, 30.0) == 30.0

    def test_custom_values(self):
        assert effective_normal_tolerance("plane", 3.0, 45.0) == 3.0
        assert effective_normal_tolerance("cylinder", 3.0, 45.0) == 6.0
        assert effective_normal_tolerance("torus", 3.0, 45.0) == 45.0


# ---------------------------------------------------------------------------
# find_dominant_face
# ---------------------------------------------------------------------------

@requires_ocp
class TestFindDominantFace:
    def test_returns_none_for_empty_points(self):
        """find_dominant_face with no points should return None."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        result = find_dominant_face(box, [], [], tol=1e-3)
        assert result is None

    def test_box_top_face_dominates(self):
        """Points clearly on the top face (z=1) of a unit box should select that face."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()

        sample_points = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.5, 1.0),  # support point
        ]
        support_flags = [False, False, False, False, True]

        result = find_dominant_face(box, sample_points, support_flags, tol=1e-3)
        assert result is not None

        dominant_face, _contributing = result
        assert dominant_face is not None
        n = local_surface_normal(dominant_face, (0.5, 0.5, 1.0))
        assert n is not None
        assert abs(abs(n[2]) - 1.0) < 0.1

    def test_support_point_breaks_tie_in_favour_of_cylinder(self):
        """When vertices lie on a shared edge between a plane and cylinder, the support
        point in the cylinder interior should make the cylinder win the vote."""
        import math as _math

        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        cyl = BRepPrimAPI_MakeCylinder(1.0, 1.0).Shape()

        vertex_points = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ]
        # Support point on the cylindrical mantle, intentionally off the seam
        # edge (at theta=0, i.e. the +X axis) so the distance solver returns
        # the mantle face rather than the seam edge.
        support_pt = (_math.cos(0.5), _math.sin(0.5), 0.5)  # r=1, theta=0.5 rad, z=0.5

        all_pts = [*vertex_points, support_pt]
        flags = [False] * len(vertex_points) + [True]

        result = find_dominant_face(cyl, all_pts, flags, tol=1e-3)
        assert result is not None

        dominant_face, _contributing = result
        stype = surface_type_of(dominant_face)
        assert stype == "cylinder", (
            f"Expected 'cylinder', got '{stype}' — support point weighting failed"
        )


# ---------------------------------------------------------------------------
# Edge/vertex face recovery
# ---------------------------------------------------------------------------
#
# Regression for the bug where a block face that lies *exactly* on a STEP
# patch surface fails to match because the patch is tessellated along the
# block-face boundaries: every sample point's nearest support on the shell is
# a shared EDGE or VERTEX rather than a FACE. The pre-fix ``_try_get_face``
# discarded those supports, leaving the perfectly-coincident face unmatched
# (observed for the fluid / Aussenring / Innenring patches of
# patches_fuer_bauraum.stp). The ancestor-map recovery resolves the incident
# faces from the edge / vertex so the majority vote and normal check proceed.


def _outer_shell(solid):
    """Return the first ``TopoDS_Shell`` of *solid* (or *solid* itself)."""
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    exp = TopExp_Explorer(solid, TopAbs_SHELL)
    if exp.More():
        return TopoDS.Shell_s(exp.Current())
    return solid


def _two_box_coplanar_solid():
    """Two unit boxes fused along x=1.

    The fused solid's top (z=1) consists of TWO coplanar faces — [0,1]x[0,1]
    and [1,2]x[0,1] — meeting at the shared edge x=1. This mirrors a STEP
    patch whose surface is split along the block-face boundaries.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    box_a = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 1.0, 1.0)).Shape()
    box_b = BRepPrimAPI_MakeBox(gp_Pnt(1.0, 0.0, 0.0), gp_Pnt(2.0, 1.0, 1.0)).Shape()
    return BRepAlgoAPI_Fuse(box_a, box_b).Shape()


@requires_ocp
class TestRecoverFaces:
    def test_face_support_returns_that_single_face(self):
        """A FACE support shape is returned as the sole candidate."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        shell = _outer_shell(box)
        edge_map, vertex_map = _build_face_ancestor_maps(shell)

        face = TopoDS.Face_s(TopExp_Explorer(shell, TopAbs_FACE).Current())
        recovered = _recover_faces(face, edge_map, vertex_map)
        assert len(recovered) == 1

    def test_edge_support_returns_both_incident_faces(self):
        """An EDGE of a closed box shell is shared by exactly two faces."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        shell = _outer_shell(box)
        edge_map, vertex_map = _build_face_ancestor_maps(shell)

        edge = TopoDS.Edge_s(TopExp_Explorer(shell, TopAbs_EDGE).Current())
        recovered = _recover_faces(edge, edge_map, vertex_map)
        assert len(recovered) == 2

    def test_vertex_support_returns_three_incident_faces(self):
        """A corner VERTEX of a box is shared by exactly three faces."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_VERTEX
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        shell = _outer_shell(box)
        edge_map, vertex_map = _build_face_ancestor_maps(shell)

        vertex = TopoDS.Vertex_s(TopExp_Explorer(shell, TopAbs_VERTEX).Current())
        recovered = _recover_faces(vertex, edge_map, vertex_map)
        assert len(recovered) == 3


@requires_ocp
class TestEdgeSeamRecovery:
    def test_nearest_face_on_coplanar_seam_returns_face(self):
        """Block face spanning two coplanar patch sub-faces must match.

        All four corners are box vertices (VERTEX support) and the centroid
        (1, 0.5, 1) sits on the shared edge between the two top faces (EDGE
        support). No sample point lands in a face interior, so the pre-fix
        code returned ``None``; with recovery the function returns a face.
        """
        solid = _two_box_coplanar_solid()
        block_face = BlockFace(
            block_name="seam",
            face_index=5,
            face_name="kmax",
            vertex_names=["v0", "v1", "v2", "v3"],
            vertex_coords=[
                (0.0, 0.0, 1.0),
                (2.0, 0.0, 1.0),
                (2.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ],
            support_points=[],
        )
        result = nearest_face_within_tol(block_face, solid, tol=1e-6)
        assert result is not None, (
            "block face lies exactly on two coplanar patch sub-faces; "
            "edge/vertex recovery must return a face instead of None"
        )

    def test_dominant_face_on_coplanar_seam_is_planar_top(self):
        """The recovered dominant face is the planar top with a +Z normal."""
        solid = _two_box_coplanar_solid()
        sample_points = [
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.5, 1.0),  # support point on the shared seam edge
        ]
        flags = [False, False, False, False, True]
        result = find_dominant_face(solid, sample_points, flags, tol=1e-6)
        assert result is not None

        dominant_face, _contrib = result
        assert surface_type_of(dominant_face) == "plane"
        n = local_surface_normal(dominant_face, (1.0, 0.5, 1.0))
        assert n is not None
        assert abs(abs(n[2]) - 1.0) < 1e-6

    def test_offset_face_still_returns_none(self):
        """Recovery must not create false positives: a face parallel to the
        seam top but lifted out of tolerance still does not match."""
        solid = _two_box_coplanar_solid()
        lifted = BlockFace(
            block_name="lifted",
            face_index=5,
            face_name="kmax",
            vertex_names=["v0", "v1", "v2", "v3"],
            vertex_coords=[
                (0.0, 0.0, 1.5),
                (2.0, 0.0, 1.5),
                (2.0, 1.0, 1.5),
                (0.0, 1.0, 1.5),
            ],
            support_points=[],
        )
        assert nearest_face_within_tol(lifted, solid, tol=1e-6) is None
