"""Tests for meshing_utils.geometry.containment."""

from __future__ import annotations

import math

import pytest

from meshing_utils import Block, BlockMeshDict, Vertex
from meshing_utils.geometry.containment import (
    STATE_IN,
    STATE_ON,
    STATE_OUT,
    aabbs_overlap,
    bmd_bbox_diagonal,
    classify_point_in_solid,
    classify_point_with_classifier,
    compute_block_centroid,
    compute_block_sample_sets,
    compute_inset_sample_points,
    compute_points_aabb,
    compute_solid_aabb,
    make_solid_classifier,
    point_inside_solid,
    resolve_block_coords,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bmd_with_named_vertices() -> BlockMeshDict:
    """Return a BlockMeshDict with 8 named vertices forming a unit cube."""
    bmd = BlockMeshDict()
    coords = [
        ("v0", [0.0, 0.0, 0.0]),
        ("v1", [1.0, 0.0, 0.0]),
        ("v2", [1.0, 1.0, 0.0]),
        ("v3", [0.0, 1.0, 0.0]),
        ("v4", [0.0, 0.0, 1.0]),
        ("v5", [1.0, 0.0, 1.0]),
        ("v6", [1.0, 1.0, 1.0]),
        ("v7", [0.0, 1.0, 1.0]),
    ]
    for name, c in coords:
        bmd.vertices.add(Vertex(name, c))
    return bmd


def _make_block_named_verts(bmd: BlockMeshDict) -> Block:
    return Block(
        "b0",
        vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        cells=[1, 1, 1],
    )


def _make_bmd_with_indexed_vertices() -> BlockMeshDict:
    """Return a BlockMeshDict with 8 unnamed (index-accessible) vertices."""
    bmd = BlockMeshDict()
    coords = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
    ]
    for c in coords:
        bmd.vertices.add(Vertex("", c))
    return bmd


# ---------------------------------------------------------------------------
# compute_block_centroid
# ---------------------------------------------------------------------------

def test_compute_block_centroid_unit_cube():
    unit_cube = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    cx, cy, cz = compute_block_centroid(unit_cube)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert cz == pytest.approx(0.5)


def test_compute_block_centroid_assertion_on_wrong_count():
    with pytest.raises(AssertionError):
        compute_block_centroid([(0.0, 0.0, 0.0)] * 4)


# ---------------------------------------------------------------------------
# resolve_block_coords — named vertices
# ---------------------------------------------------------------------------

def test_resolve_block_coords_named_vertices():
    bmd = _make_bmd_with_named_vertices()
    block = _make_block_named_verts(bmd)
    coords = resolve_block_coords(block, bmd)
    assert len(coords) == 8
    assert coords[0] == pytest.approx((0.0, 0.0, 0.0))
    assert coords[6] == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# resolve_block_coords — indexed vertices
# ---------------------------------------------------------------------------

def test_resolve_block_coords_indexed_vertices():
    bmd = _make_bmd_with_indexed_vertices()
    block = Block("b0", vertices=["0", "1", "2", "3", "4", "5", "6", "7"], cells=[1, 1, 1])
    coords = resolve_block_coords(block, bmd)
    assert len(coords) == 8
    assert coords[0] == pytest.approx((0.0, 0.0, 0.0))
    assert coords[2] == pytest.approx((2.0, 2.0, 0.0))


# ---------------------------------------------------------------------------
# resolve_block_coords — wrong vertex count
# ---------------------------------------------------------------------------

def test_resolve_block_coords_wrong_vertex_count():
    bmd = _make_bmd_with_named_vertices()
    # Block with only 4 references
    block = Block("b0", vertices=["v0", "v1", "v2", "v3"], cells=[1, 1, 1])
    with pytest.raises(ValueError, match="expected 8"):
        resolve_block_coords(block, bmd)


# ---------------------------------------------------------------------------
# OCP-based tests — skipped via importorskip in their fixtures when OCP is absent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def box_solid():
    """Create a 2x2x2 box solid centred at (1,1,1) using OCP."""
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    maker = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(2.0, 2.0, 2.0))
    return maker.Solid()


def test_classify_point_in_solid_states(box_solid):
    # Interior point
    assert classify_point_in_solid(box_solid, (1.0, 1.0, 1.0), 1e-7) == STATE_IN
    # Exterior point
    assert classify_point_in_solid(box_solid, (5.0, 5.0, 5.0), 1e-7) == STATE_OUT


def test_point_inside_solid_box(box_solid):
    assert point_inside_solid(box_solid, (1.0, 1.0, 1.0), 1e-7) is True
    assert point_inside_solid(box_solid, (5.0, 5.0, 5.0), 1e-7) is False


# ---------------------------------------------------------------------------
# bmd_bbox_diagonal
# ---------------------------------------------------------------------------

def test_bmd_bbox_diagonal_unit_cube():
    bmd = _make_bmd_with_named_vertices()
    diag = bmd_bbox_diagonal(bmd)
    assert diag == pytest.approx(math.sqrt(3.0))


def test_bmd_bbox_diagonal_empty():
    bmd = BlockMeshDict()
    assert bmd_bbox_diagonal(bmd) == 0.0


# ---------------------------------------------------------------------------
# Shared unit-cube coordinates for inset-sampling tests
# ---------------------------------------------------------------------------

_UNIT_CUBE_COORDS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]


# ---------------------------------------------------------------------------
# compute_inset_sample_points
# ---------------------------------------------------------------------------

class TestComputeInsetSamplePoints:

    def test_inset_samples_unit_cube_default_factor(self):
        """Unit cube with f=0.5: centroid=(0.5,0.5,0.5), V0-inset=(0.25,0.25,0.25)."""
        samples = compute_inset_sample_points(_UNIT_CUBE_COORDS)
        # Index 0 is the centroid
        assert samples[0] == pytest.approx((0.5, 0.5, 0.5))
        # V0 = (0,0,0) → inset = centroid + 0.5*(V0 - centroid) = (0.25,0.25,0.25)
        assert samples[1] == pytest.approx((0.25, 0.25, 0.25))
        # V1 = (1,0,0) → inset = (0.5 + 0.5*(1-0.5), 0.5+0.5*(0-0.5), 0.5+0.5*(0-0.5))
        #                       = (0.75, 0.25, 0.25)
        assert samples[2] == pytest.approx((0.75, 0.25, 0.25))

    def test_inset_samples_length_is_nine(self):
        samples = compute_inset_sample_points(_UNIT_CUBE_COORDS)
        assert len(samples) == 9

    def test_inset_samples_centroid_at_index_zero(self):
        samples = compute_inset_sample_points(_UNIT_CUBE_COORDS)
        expected_centroid = compute_block_centroid(_UNIT_CUBE_COORDS)
        assert samples[0] == pytest.approx(expected_centroid)

    def test_inset_samples_all_strictly_inside_aabb(self):
        """All 9 sample points must lie strictly inside the unit-cube AABB."""
        samples = compute_inset_sample_points(_UNIT_CUBE_COORDS)
        xs = [c[0] for c in _UNIT_CUBE_COORDS]
        ys = [c[1] for c in _UNIT_CUBE_COORDS]
        zs = [c[2] for c in _UNIT_CUBE_COORDS]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        for pt in samples:
            assert x_min < pt[0] < x_max
            assert y_min < pt[1] < y_max
            assert z_min < pt[2] < z_max

    def test_inset_factor_invalid_zero_raises(self):
        with pytest.raises(ValueError, match="inset_factor"):
            compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=0.0)

    def test_inset_factor_invalid_one_raises(self):
        with pytest.raises(ValueError, match="inset_factor"):
            compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=1.0)

    def test_inset_factor_invalid_negative_raises(self):
        with pytest.raises(ValueError, match="inset_factor"):
            compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=-0.1)

    def test_inset_factor_invalid_above_one_raises(self):
        with pytest.raises(ValueError, match="inset_factor"):
            compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=1.5)

    def test_wrong_coord_count_raises_seven(self):
        with pytest.raises(ValueError, match="Expected 8 coords"):
            compute_inset_sample_points(_UNIT_CUBE_COORDS[:7])

    def test_wrong_coord_count_raises_nine(self):
        extra = [*_UNIT_CUBE_COORDS, (2.0, 2.0, 2.0)]
        with pytest.raises(ValueError, match="Expected 8 coords"):
            compute_inset_sample_points(extra)

    def test_inset_samples_arbitrary_factor(self):
        """With f=0.1, inset samples are very close to the centroid."""
        f = 0.1
        samples = compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=f)
        M = compute_block_centroid(_UNIT_CUBE_COORDS)
        for i, V in enumerate(_UNIT_CUBE_COORDS):
            expected = (
                M[0] + f * (V[0] - M[0]),
                M[1] + f * (V[1] - M[1]),
                M[2] + f * (V[2] - M[2]),
            )
            assert samples[i + 1] == pytest.approx(expected)

    def test_inset_samples_skewed_hex(self):
        """Annulus-sector-like vertices: inset samples must differ from centroid."""
        import math
        # Thin annulus sector: r_inner=1, r_outer=2, dz=0.1, angle 0..30°
        angles = [0.0, math.radians(30)]
        radii = [1.0, 2.0]
        z_vals = [0.0, 0.1]
        coords_skewed: list[tuple[float, float, float]] = []
        for z in z_vals:
            for r in radii:
                for a in angles:
                    coords_skewed.append((r * math.cos(a), r * math.sin(a), z))
        # coords_skewed now has 8 points
        assert len(coords_skewed) == 8
        samples = compute_inset_sample_points(coords_skewed)
        assert len(samples) == 9
        M = samples[0]
        # All inset samples must differ from the centroid
        for pt in samples[1:]:
            dist = math.sqrt(
                (pt[0] - M[0]) ** 2 + (pt[1] - M[1]) ** 2 + (pt[2] - M[2]) ** 2
            )
            assert dist > 0.0


# ---------------------------------------------------------------------------
# TestComputeBlockSampleSets
# ---------------------------------------------------------------------------

class TestComputeBlockSampleSets:
    """Tests for compute_block_sample_sets — the two-pass sample dispatcher."""

    def test_returns_tuple_of_two_lists(self):
        result = compute_block_sample_sets(_UNIT_CUBE_COORDS)
        assert isinstance(result, tuple)
        assert len(result) == 2
        interior, vertices = result
        assert isinstance(interior, list)
        assert isinstance(vertices, list)

    def test_interior_has_nine_points(self):
        interior, _ = compute_block_sample_sets(_UNIT_CUBE_COORDS)
        assert len(interior) == 9

    def test_vertices_has_eight_points(self):
        _, vertices = compute_block_sample_sets(_UNIT_CUBE_COORDS)
        assert len(vertices) == 8

    def test_interior_is_identical_to_compute_inset_sample_points(self):
        interior, _ = compute_block_sample_sets(_UNIT_CUBE_COORDS)
        expected = compute_inset_sample_points(_UNIT_CUBE_COORDS)
        assert interior == expected

    def test_vertex_list_preserves_order(self):
        _, vertices = compute_block_sample_sets(_UNIT_CUBE_COORDS)
        for i, coord in enumerate(_UNIT_CUBE_COORDS):
            assert vertices[i] == pytest.approx(coord)

    def test_inset_factor_propagated_to_interior_samples(self):
        f = 0.3
        interior_custom, _ = compute_block_sample_sets(_UNIT_CUBE_COORDS, inset_factor=f)
        expected = compute_inset_sample_points(_UNIT_CUBE_COORDS, inset_factor=f)
        assert interior_custom == expected

    def test_wrong_coord_count_raises_via_inset(self):
        with pytest.raises(ValueError, match="Expected 8 coords"):
            compute_block_sample_sets(_UNIT_CUBE_COORDS[:7])


# ---------------------------------------------------------------------------
# TestComputeSolidAABB (OCC-marked)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unit_box_solid():
    """Create a 1x1x1 box solid from (0,0,0) to (1,1,1) using OCP."""
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    maker = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 1.0, 1.0))
    return maker.Solid()


class TestComputeSolidAABB:
    """Tests for compute_solid_aabb — requires OCC."""

    def test_compute_solid_aabb_unit_box(self, unit_box_solid):
        """1x1x1 box → AABB contains the unit box with padding."""
        tol = 1e-7
        aabb = compute_solid_aabb(unit_box_solid, tol)
        xmin, ymin, zmin, xmax, ymax, zmax = aabb
        # Padded min must be slightly below 0, padded max slightly above 1
        assert xmin < 0.0
        assert ymin < 0.0
        assert zmin < 0.0
        assert xmax > 1.0
        assert ymax > 1.0
        assert zmax > 1.0
        # The unpadded bounds must be inside the padded bounds
        assert xmin <= 0.0 <= xmax
        assert ymin <= 0.0 <= ymax
        assert zmin <= 0.0 <= zmax
        assert xmin <= 1.0 <= xmax
        assert ymin <= 1.0 <= ymax
        assert zmin <= 1.0 <= zmax

    def test_compute_solid_aabb_pad_uses_diag_when_tol_small(self, unit_box_solid):
        """tol=0 → pad = diag * 1e-9 (diagonal of unit cube is sqrt(3))."""
        import math

        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        # Compute the raw OCC bbox first — OCC's BRepBndLib.Add_s applies its
        # own internal padding which can be orders of magnitude larger than
        # our diag*1e-9, so we must compare against the raw bounds rather
        # than against the nominal unit-cube extents.
        raw_bbox = Bnd_Box()
        BRepBndLib.Add_s(unit_box_solid, raw_bbox)
        raw_xmin, raw_ymin, raw_zmin, raw_xmax, raw_ymax, raw_zmax = raw_bbox.Get()

        dx = raw_xmax - raw_xmin
        dy = raw_ymax - raw_ymin
        dz = raw_zmax - raw_zmin
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)
        expected_pad = diag * 1e-9

        aabb = compute_solid_aabb(unit_box_solid, tol=0.0)
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = aabb
        assert xmin == pytest.approx(raw_xmin - expected_pad, abs=1e-15)
        assert xmax == pytest.approx(raw_xmax + expected_pad, abs=1e-15)

    def test_compute_solid_aabb_pad_uses_tol_when_tol_large(self, unit_box_solid):
        """tol=1.0 → pad = tol = 1.0 (larger than diag*1e-9)."""
        tol = 1.0
        aabb = compute_solid_aabb(unit_box_solid, tol=tol)
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = aabb
        assert xmin == pytest.approx(0.0 - tol, rel=1e-6)
        assert xmax == pytest.approx(1.0 + tol, rel=1e-6)

    def test_compute_solid_aabb_raises_on_void(self):
        """A void bounding box must raise ValueError."""
        from unittest.mock import MagicMock, patch

        mock_bbox = MagicMock()
        mock_bbox.IsVoid.return_value = True
        mock_BRepBndLib = MagicMock()
        mock_BRepBndLib.Add_s = MagicMock()

        # Patch both ``Bnd_Box`` and ``BRepBndLib`` to prevent real OCC code
        # from inspecting our MagicMock arguments — touching a MagicMock from
        # inside an OCC C extension triggers unbounded Mock auto-spec
        # recursion and hangs the test process on newer Python versions.
        with patch.dict(
            "sys.modules",
            {
                "OCP.Bnd": MagicMock(Bnd_Box=lambda: mock_bbox),
                "OCP.BRepBndLib": MagicMock(BRepBndLib=mock_BRepBndLib),
            },
        ), pytest.raises(ValueError, match="void"):
            compute_solid_aabb(MagicMock(), tol=1e-7)


# ---------------------------------------------------------------------------
# TestComputePointsAABB (pure Python)
# ---------------------------------------------------------------------------

class TestComputePointsAABB:
    """Tests for compute_points_aabb — no OCC required."""

    def test_compute_points_aabb_simple_cube(self):
        """8 corners of the unit cube → (0,0,0,1,1,1)."""
        pts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        aabb = compute_points_aabb(pts)
        assert aabb == pytest.approx((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))

    def test_compute_points_aabb_single_point(self):
        """Single point → degenerate AABB where min == max."""
        aabb = compute_points_aabb([(3.0, 4.0, 5.0)])
        assert aabb == pytest.approx((3.0, 4.0, 5.0, 3.0, 4.0, 5.0))

    def test_compute_points_aabb_empty_raises(self):
        """Empty input must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_points_aabb([])


# ---------------------------------------------------------------------------
# TestAABBsOverlap (pure Python)
# ---------------------------------------------------------------------------

class TestAABBsOverlap:
    """Tests for aabbs_overlap — no OCC required."""

    def test_aabbs_overlap_disjoint_x(self):
        """Boxes separated along x → no overlap."""
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (2.0, 0.0, 0.0, 3.0, 1.0, 1.0)
        assert aabbs_overlap(a, b) is False

    def test_aabbs_overlap_disjoint_y(self):
        """Boxes separated along y → no overlap."""
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (0.0, 2.0, 0.0, 1.0, 3.0, 1.0)
        assert aabbs_overlap(a, b) is False

    def test_aabbs_overlap_disjoint_z(self):
        """Boxes separated along z → no overlap."""
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (0.0, 0.0, 2.0, 1.0, 1.0, 3.0)
        assert aabbs_overlap(a, b) is False

    def test_aabbs_overlap_touching(self):
        """Boxes share a face (touching) → edge-inclusive → True."""
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (1.0, 0.0, 0.0, 2.0, 1.0, 1.0)
        assert aabbs_overlap(a, b) is True

    def test_aabbs_overlap_contained(self):
        """Box A fully contained in box B → True."""
        a = (0.2, 0.2, 0.2, 0.8, 0.8, 0.8)
        b = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert aabbs_overlap(a, b) is True

    def test_aabbs_overlap_partial(self):
        """Partial overlap in all three axes → True."""
        a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = (0.5, 0.5, 0.5, 1.5, 1.5, 1.5)
        assert aabbs_overlap(a, b) is True


# ---------------------------------------------------------------------------
# TestSolidClassifierReuse (OCC-marked)
# ---------------------------------------------------------------------------

class TestSolidClassifierReuse:
    """Tests for make_solid_classifier and classify_point_with_classifier."""

    def test_make_solid_classifier_returns_instance(self, unit_box_solid):
        """make_solid_classifier must return a BRepClass3d_SolidClassifier."""
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        clf = make_solid_classifier(unit_box_solid)
        assert isinstance(clf, BRepClass3d_SolidClassifier)

    def test_classify_point_with_classifier_inside_box(self, unit_box_solid):
        """Point inside the unit box → STATE_IN."""
        clf = make_solid_classifier(unit_box_solid)
        result = classify_point_with_classifier(clf, (0.5, 0.5, 0.5), 1e-7)
        assert result == STATE_IN

    def test_classify_point_with_classifier_outside_box(self, unit_box_solid):
        """Point clearly outside the unit box → STATE_OUT."""
        clf = make_solid_classifier(unit_box_solid)
        result = classify_point_with_classifier(clf, (5.0, 5.0, 5.0), 1e-7)
        assert result == STATE_OUT

    def test_classify_point_with_classifier_on_boundary(self, unit_box_solid):
        """Point on the boundary face → STATE_ON."""
        clf = make_solid_classifier(unit_box_solid)
        # Face centre of the top face (z=1)
        result = classify_point_with_classifier(clf, (0.5, 0.5, 1.0), 1e-7)
        assert result == STATE_ON

    def test_classifier_reused_for_multiple_points(self, unit_box_solid):
        """A single classifier must produce results consistent with classify_point_in_solid."""
        clf = make_solid_classifier(unit_box_solid)
        test_points = [
            (0.5, 0.5, 0.5),   # inside
            (5.0, 5.0, 5.0),   # outside
            (-1.0, 0.0, 0.0),  # outside
            (0.1, 0.1, 0.1),   # inside
        ]
        for pt in test_points:
            expected = classify_point_in_solid(unit_box_solid, pt, 1e-7)
            actual = classify_point_with_classifier(clf, pt, 1e-7)
            assert actual == expected, (
                f"Mismatch at {pt}: classifier={actual!r}, per-point={expected!r}"
            )
