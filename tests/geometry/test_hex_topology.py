"""Unit tests for meshing_utils.geometry.hex_topology."""

import pytest

from meshing_utils.geometry.hex_topology import (
    HexCandidate,
    HexValidationError,
    OrderingConsistencyError,
    PointPool,
    assert_block_face_normals_outward,
    assert_hex_outward_from_coords,
    enforce_openfoam_face_convention,
    ensure_right_handed,
    order_hex_vertices,
    validate_hex,
)

# ---------------------------------------------------------------------------
# Helpers: build a unit-cube HexCandidate
# ---------------------------------------------------------------------------

def _unit_cube_candidate(label=None):
    """Return a valid HexCandidate for the unit cube [0,1]^3."""
    vertex_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (1, 2, 6, 5),
        (0, 4, 7, 3),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return HexCandidate(
        vertex_indices=vertex_indices,
        faces=faces,
        edges=edges,
        edge_curves={},
        label=label,
    )


def _unit_cube_pool():
    """PointPool containing the 8 vertices of the unit cube."""
    pool = PointPool(tol=1e-6)
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for c in coords:
        pool.add_or_get(c)
    return pool


# ---------------------------------------------------------------------------
# PointPool tests
# ---------------------------------------------------------------------------

class TestPointPool:
    def test_identical_point_returns_same_index(self):
        pool = PointPool(tol=1e-6)
        idx_a = pool.add_or_get((1.0, 2.0, 3.0))
        idx_b = pool.add_or_get((1.0, 2.0, 3.0))
        assert idx_a == idx_b

    def test_point_just_inside_tolerance_is_same(self):
        tol = 1e-4
        pool = PointPool(tol=tol)
        idx_a = pool.add_or_get((0.0, 0.0, 0.0))
        delta = 0.4 * tol
        idx_b = pool.add_or_get((delta, delta, delta))
        assert idx_a == idx_b

    def test_point_just_outside_tolerance_is_new(self):
        tol = 1e-4
        pool = PointPool(tol=tol)
        idx_a = pool.add_or_get((0.0, 0.0, 0.0))
        delta = 1.5 * tol
        idx_b = pool.add_or_get((delta, delta, delta))
        assert idx_a != idx_b

    def test_coord_at_returns_correct_coord(self):
        pool = PointPool(tol=1e-6)
        c = (3.5, -1.0, 0.25)
        idx = pool.add_or_get(c)
        assert pool.coord_at(idx) == c

    def test_coord_at_invalid_index_raises(self):
        pool = PointPool(tol=1e-6)
        with pytest.raises(IndexError):
            pool.coord_at(99)


# ---------------------------------------------------------------------------
# validate_hex tests
# ---------------------------------------------------------------------------

class TestValidateHex:
    def test_valid_unit_cube_passes(self):
        cand = _unit_cube_candidate()
        validate_hex(cand)  # must not raise

    def test_wrong_vertex_count_raises(self):
        cand = _unit_cube_candidate(label="bad_solid")
        cand.vertex_indices = cand.vertex_indices[:7]
        with pytest.raises(HexValidationError, match="8 vertices"):
            validate_hex(cand)

    def test_wrong_edge_count_raises(self):
        cand = _unit_cube_candidate(label="edge_bad")
        cand.edges = cand.edges[:11]
        with pytest.raises(HexValidationError, match="12 edges"):
            validate_hex(cand)

    def test_wrong_face_count_raises(self):
        cand = _unit_cube_candidate(label="face_bad")
        cand.faces = cand.faces[:5]
        with pytest.raises(HexValidationError, match="6 faces"):
            validate_hex(cand)

    def test_label_in_error_message(self):
        cand = _unit_cube_candidate(label="MySolid")
        cand.vertex_indices = cand.vertex_indices[:7]
        with pytest.raises(HexValidationError, match="MySolid"):
            validate_hex(cand)


# ---------------------------------------------------------------------------
# order_hex_vertices tests
# ---------------------------------------------------------------------------

class TestOrderHexVertices:
    def test_axis_aligned_cube_bottom_face_recognized(self):
        cand = _unit_cube_candidate()
        pool = _unit_cube_pool()
        ordering = order_hex_vertices(cand, pool)
        assert len(ordering) == 8
        bottom_coords = [pool.coord_at(ordering[i]) for i in range(4)]
        for coord in bottom_coords:
            assert coord[2] == pytest.approx(0.0), f"Expected z=0, got {coord}"

    def test_axis_aligned_cube_top_face_recognized(self):
        cand = _unit_cube_candidate()
        pool = _unit_cube_pool()
        ordering = order_hex_vertices(cand, pool)
        top_coords = [pool.coord_at(ordering[i]) for i in range(4, 8)]
        for coord in top_coords:
            assert coord[2] == pytest.approx(1.0), f"Expected z=1, got {coord}"

    def test_axis_aligned_cube_v0_is_lex_min(self):
        """v0 must be the bottom vertex with lex-min (z, y, x)."""
        cand = _unit_cube_candidate()
        pool = _unit_cube_pool()
        ordering = order_hex_vertices(cand, pool)
        v0_coord = pool.coord_at(ordering[0])
        assert v0_coord == pytest.approx((0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# ensure_right_handed tests
# ---------------------------------------------------------------------------

class TestEnsureRightHanded:
    def test_already_right_handed_unchanged(self):
        pool = _unit_cube_pool()
        cand = _unit_cube_candidate()
        ordering = order_hex_vertices(cand, pool)
        corrected = ensure_right_handed(ordering, pool)
        v = [pool.coord_at(i) for i in corrected]
        e1 = tuple(v[1][k] - v[0][k] for k in range(3))
        e3 = tuple(v[3][k] - v[0][k] for k in range(3))
        e4 = tuple(v[4][k] - v[0][k] for k in range(3))
        cross = (
            e1[1] * e3[2] - e1[2] * e3[1],
            e1[2] * e3[0] - e1[0] * e3[2],
            e1[0] * e3[1] - e1[1] * e3[0],
        )
        triple = sum(cross[k] * e4[k] for k in range(3))
        assert triple > 0


# ---------------------------------------------------------------------------
# Helpers for enforce / assert tests
# ---------------------------------------------------------------------------

def _standard_unit_cube_ordering():
    pool = PointPool(tol=1e-6)
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for c in coords:
        pool.add_or_get(c)
    ordering = [0, 1, 2, 3, 4, 5, 6, 7]
    return pool, ordering


def _hex_with_inward_kmin_winding():
    pool = PointPool(tol=1e-6)
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
    ]
    for c in coords:
        pool.add_or_get(c)

    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 3, 7, 4), (1, 5, 6, 2),
        (0, 4, 5, 1), (3, 2, 6, 7),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    cand = HexCandidate(
        vertex_indices=list(range(8)),
        faces=faces,
        edges=edges,
        edge_curves={},
        label="block77_like_symmetric",
    )
    ordering = order_hex_vertices(cand, pool)
    ordering = ensure_right_handed(ordering, pool)
    return pool, ordering


# ---------------------------------------------------------------------------
# TestEnforceOpenFOAMFaceConvention
# ---------------------------------------------------------------------------

class TestEnforceOpenFOAMFaceConvention:

    def test_already_correct_unit_cube(self):
        pool, ordering = _standard_unit_cube_ordering()
        result = enforce_openfoam_face_convention(ordering, pool)
        assert set(result[:4]) == {0, 1, 2, 3}
        assert set(result[4:]) == {4, 5, 6, 7}
        assert_block_face_normals_outward(result, pool, "unit_cube_correct")

    def test_idempotent(self):
        pool, ordering = _standard_unit_cube_ordering()
        once = enforce_openfoam_face_convention(ordering, pool)
        twice = enforce_openfoam_face_convention(once, pool)
        assert once == twice

    def test_degenerate_planar_hex_raises(self):
        pool = PointPool(tol=1e-6)
        coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.2, 0.2, 0.0), (0.8, 0.2, 0.0), (0.8, 0.8, 0.0), (0.2, 0.8, 0.0),
        ]
        for c in coords:
            pool.add_or_get(c)
        ordering = [0, 1, 2, 3, 4, 5, 6, 7]
        with pytest.raises(OrderingConsistencyError, match=r"[Dd]egenerate"):
            enforce_openfoam_face_convention(ordering, pool)


# ---------------------------------------------------------------------------
# TestAssertBlockFaceNormalsOutward
# ---------------------------------------------------------------------------

class TestAssertBlockFaceNormalsOutward:

    def test_correct_block_passes(self):
        pool, ordering = _standard_unit_cube_ordering()
        assert_block_face_normals_outward(ordering, pool, "standard_cube")

    def test_inverted_block_raises(self):
        pool = PointPool(tol=1e-6)
        coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        for c in coords:
            pool.add_or_get(c)
        ordering = [0, 1, 2, 3, 4, 7, 6, 5]
        with pytest.raises(OrderingConsistencyError):
            assert_block_face_normals_outward(ordering, pool, "inverted_top")


# ---------------------------------------------------------------------------
# TestAssertHexOutwardFromCoords
# ---------------------------------------------------------------------------

class TestAssertHexOutwardFromCoords:

    def test_correct_coords_passes(self):
        coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        assert_hex_outward_from_coords(coords, block_label="standard_cube")

    def test_inverted_coords_raises(self):
        coords = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        coords_inv = list(coords)
        coords_inv[4], coords_inv[7] = coords_inv[7], coords_inv[4]
        coords_inv[5], coords_inv[6] = coords_inv[6], coords_inv[5]
        with pytest.raises(OrderingConsistencyError):
            assert_hex_outward_from_coords(coords_inv, block_label="inverted")
