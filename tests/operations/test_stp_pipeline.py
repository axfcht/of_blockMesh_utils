"""Unit tests for meshing_utils.operations.stp_pipeline.

These tests are adapted from the three existing test files in tests/tools/.
OCP-dependent tests are guarded by the ``requires_ocp`` skip marker.
Most tests mock load_step and work with prepared HexCandidate stubs.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import meshing_utils.operations.stp_pipeline as stp_mod
from meshing_utils import (
    BlockMeshDict,
    CurveInfo,
    Edge,
    HexCandidate,
    PointPool,
    Vertex,
)
from meshing_utils.operations.stp_pipeline import (
    add_edge_with_conflict_check,
    parse_fractions,
    parse_origin,
)

# Skip marker for tests that build real OCP (cadquery-ocp) shapes or call the
# real load_step. Skipped when OCP is not importable rather than erroring.
requires_ocp = pytest.mark.skipif(
    importlib.util.find_spec("OCP") is None,
    reason="requires OCP (cadquery-ocp); not installed in this environment",
)


def _run(*, case_dir, **kwargs):
    """Phase 2.3 shim: build a StpPipelineConfig from kwargs and dispatch.

    Lets the rest of this test file keep its readable kwargs-style call
    pattern after :func:`stp_mod.run` switched to the
    ``run(config, *, case_dir)`` signature.
    """
    stp_mod.run(stp_mod.StpPipelineConfig(**kwargs), case_dir=case_dir)


# ---------------------------------------------------------------------------
# parse_origin
# ---------------------------------------------------------------------------

class TestParseOrigin:
    def test_zero_origin(self):
        result = parse_origin("(0 0 0)")
        assert result == (0.0, 0.0, 0.0)

    def test_float_origin(self):
        result = parse_origin("(-1.5 2 -3.14)")
        assert result == pytest.approx((-1.5, 2.0, -3.14))

    def test_extra_spaces_valid(self):
        result = parse_origin("(  1.0  2.0  3.0  )")
        assert result == pytest.approx((1.0, 2.0, 2.0 + 1.0))  # (1,2,3)

    def test_missing_component_raises(self):
        with pytest.raises(ValueError, match="origin"):
            parse_origin("(0 0)")

    def test_no_parens_raises(self):
        with pytest.raises(ValueError, match="origin"):
            parse_origin("0 0 0")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="origin"):
            parse_origin("")

    def test_letters_raise(self):
        with pytest.raises(ValueError, match="origin"):
            parse_origin("(a b c)")


# ---------------------------------------------------------------------------
# Helpers: build a minimal mock for a valid single-solid case
# ---------------------------------------------------------------------------

def _make_mock_case_dir(tmp_path: Path) -> Path:
    """Create a minimal OpenFOAM case directory layout."""
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant" / "geometry").mkdir(parents=True)
    return case_dir


def _place_stub_stp(case_dir: Path, name: str = "part.stp") -> Path:
    """Create a dummy STEP file (not real STEP, just to satisfy path checks)."""
    p = case_dir / "constant" / "geometry" / name
    p.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    return p


def _make_hex_candidate_stub(pool):
    """Return a minimal HexCandidate+pool pair for a unit cube."""
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for c in coords:
        pool.add_or_get(c)

    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (3, 7, 6, 2),
        (1, 2, 6, 5), (0, 4, 7, 3),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return HexCandidate(
        vertex_indices=list(range(8)),
        faces=faces,
        edges=edges,
        edge_curves={},
        label="TestSolid",
    )


def _make_unit_cube_candidate(
    pool: PointPool, offset: float = 0.0, label: str = "TestSolid"
) -> HexCandidate:
    """Return a HexCandidate for a unit cube, optionally offset in X by *offset*.

    All vertices are registered into *pool*.
    """
    coords = [
        (offset + 0.0, 0.0, 0.0),
        (offset + 1.0, 0.0, 0.0),
        (offset + 1.0, 1.0, 0.0),
        (offset + 0.0, 1.0, 0.0),
        (offset + 0.0, 0.0, 1.0),
        (offset + 1.0, 0.0, 1.0),
        (offset + 1.0, 1.0, 1.0),
        (offset + 0.0, 1.0, 1.0),
    ]
    base = len(pool)
    for c in coords:
        pool.add_or_get(c)

    indices = list(range(base, base + 8))
    faces = [
        (indices[0], indices[3], indices[2], indices[1]),
        (indices[4], indices[5], indices[6], indices[7]),
        (indices[0], indices[1], indices[5], indices[4]),
        (indices[3], indices[7], indices[6], indices[2]),
        (indices[1], indices[2], indices[6], indices[5]),
        (indices[0], indices[4], indices[7], indices[3]),
    ]
    edges = [
        (indices[0], indices[1]),
        (indices[1], indices[2]),
        (indices[2], indices[3]),
        (indices[3], indices[0]),
        (indices[4], indices[5]),
        (indices[5], indices[6]),
        (indices[6], indices[7]),
        (indices[7], indices[4]),
        (indices[0], indices[4]),
        (indices[1], indices[5]),
        (indices[2], indices[6]),
        (indices[3], indices[7]),
    ]
    return HexCandidate(
        vertex_indices=indices,
        faces=faces,
        edges=edges,
        edge_curves={},
        label=label,
    )


def _run_tool(case_dir: Path, candidates, pool, overwrite=False, strict=False,
              name_collision="suffix", default_patch_name="defaultFaces",
              default_patch_name_explicit=False, origin=(0.0, 0.0, 0.0)):
    """Convenience wrapper around _run() with mocked load_step."""
    with patch.object(stp_mod, "load_step", return_value=(candidates, pool, "mm")):
        _run(
            case_dir=case_dir,
            origin=origin,
            tol=1e-6,
            n_samples=20,
            name_collision=name_collision,
            strict=strict,
            overwrite=overwrite,
            default_patch_name=default_patch_name,
            default_patch_name_explicit=default_patch_name_explicit,
        )


# ---------------------------------------------------------------------------
# File-system error cases
# ---------------------------------------------------------------------------

class TestRunFileSystemErrors:
    def test_no_stp_file_raises(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

    def test_multiple_stp_files_raises(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir, "part1.stp")
        _place_stub_stp(case_dir, "part2.stp")
        with pytest.raises(ValueError):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

    def test_missing_system_dir_raises(self, tmp_path):
        case_dir = tmp_path / "case"
        (case_dir / "constant" / "geometry").mkdir(parents=True)
        _place_stub_stp(case_dir)
        with pytest.raises(FileNotFoundError):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------

class TestOverwriteProtection:
    def test_existing_file_without_overwrite_flag_enters_append_mode(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)
        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        bmd_path = case_dir / "system" / "blockMeshDict"
        assert bmd_path.exists(), "blockMeshDict must exist after first run"

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_hex_candidate_stub(pool2)
        with patch.object(stp_mod, "load_step", return_value=([stub2], pool2, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        bak_path = bmd_path.with_suffix(".bak")
        assert bak_path.exists(), "Backup .bak must be created in append mode"

    def test_existing_file_with_overwrite_creates_backup(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)
        bmd_path = case_dir / "system" / "blockMeshDict"
        bmd_path.write_text("existing content")

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=True,
            )

        bak_path = bmd_path.with_suffix(".bak")
        assert bak_path.exists(), "Backup file .bak must be created"
        assert bak_path.read_text() == "existing content"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_single_solid_produces_blockMeshDict(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        bmd_path = case_dir / "system" / "blockMeshDict"
        assert bmd_path.exists(), "blockMeshDict must be created"
        content = bmd_path.read_text()
        assert "vertices" in content
        assert "blocks" in content
        assert "hex" in content

    def test_origin_shift_applied(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        origin = (1.0, 2.0, 3.0)
        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=origin,
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        bmd_path = case_dir / "system" / "blockMeshDict"
        content = bmd_path.read_text()
        assert "-1" in content
        assert "-2" in content
        assert "-3" in content


# ---------------------------------------------------------------------------
# Name-collision policy
# ---------------------------------------------------------------------------

class TestNameCollision:
    def test_suffix_policy_renames_duplicate_block(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub1 = _make_hex_candidate_stub(pool)

        PointPool(tol=1e-6)
        coords2 = [
            (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0), (3.0, 0.0, 1.0), (3.0, 1.0, 1.0), (2.0, 1.0, 1.0),
        ]
        for c in coords2:
            pool.add_or_get(c)

        faces2 = [
            (8, 11, 10, 9), (12, 13, 14, 15),
            (8, 9, 13, 12), (11, 15, 14, 10),
            (9, 10, 14, 13), (8, 12, 15, 11),
        ]
        edges2 = [
            (8, 9), (9, 10), (10, 11), (11, 8),
            (12, 13), (13, 14), (14, 15), (15, 12),
            (8, 12), (9, 13), (10, 14), (11, 15),
        ]
        stub2 = HexCandidate(
            vertex_indices=list(range(8, 16)),
            faces=faces2,
            edges=edges2,
            edge_curves={},
            label="TestSolid",  # same label as stub1
        )

        with patch.object(stp_mod, "load_step", return_value=([stub1, stub2], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        bmd_path = case_dir / "system" / "blockMeshDict"
        content = bmd_path.read_text()
        assert "TestSolid" in content
        assert "TestSolid_2" in content


# ---------------------------------------------------------------------------
# Strict mode edge conflict
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_edge_conflict_raises(self, tmp_path):
        bmd = BlockMeshDict()
        logger_obj = logging.getLogger("test")
        edge_origin = {}

        arc_curve = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.5, 0.5))
        stp_mod.add_edge_with_conflict_check(
            bmd, "v0", "v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=arc_curve, block_name="blockA",
            edge_origin=edge_origin, strict=True, logger=logger_obj,
        )

        arc_curve2 = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.6, 0.5))
        with pytest.raises((ValueError, SystemExit)):
            stp_mod.add_edge_with_conflict_check(
                bmd, "v0", "v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=arc_curve2, block_name="blockB",
                edge_origin=edge_origin, strict=True, logger=logger_obj,
            )

    def test_non_strict_edge_conflict_logs_warning(self, tmp_path, caplog):
        bmd = BlockMeshDict()
        logger_obj = logging.getLogger("test_non_strict")
        edge_origin = {}

        arc_curve = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.5, 0.5))
        stp_mod.add_edge_with_conflict_check(
            bmd, "v0", "v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=arc_curve, block_name="blockA",
            edge_origin=edge_origin, strict=False, logger=logger_obj,
        )

        arc_curve2 = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.6, 0.5))
        with caplog.at_level(logging.WARNING):
            stp_mod.add_edge_with_conflict_check(
                bmd, "v0", "v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=arc_curve2, block_name="blockB",
                edge_origin=edge_origin, strict=False, logger=logger_obj,
            )

        assert any("conflict" in r.message.lower() or "warn" in r.message.lower()
                   for r in caplog.records), \
            "Expected a warning log entry for edge conflict"


# ---------------------------------------------------------------------------
# BSpline / arc classification (via classify_edge mock)
# ---------------------------------------------------------------------------

@requires_ocp
class TestClassifyEdge:
    def test_bspline_support_points_exclude_endpoints(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.Geom import Geom_BSplineCurve
        from OCP.gp import gp_Pnt
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

        poles = TColgp_Array1OfPnt(1, 4)
        poles.SetValue(1, gp_Pnt(0, 0, 0))
        poles.SetValue(2, gp_Pnt(0.3, 0.5, 0))
        poles.SetValue(3, gp_Pnt(0.7, 0.5, 0))
        poles.SetValue(4, gp_Pnt(1, 0, 0))

        knots = TColStd_Array1OfReal(1, 2)
        knots.SetValue(1, 0.0)
        knots.SetValue(2, 1.0)

        mults = TColStd_Array1OfInteger(1, 2)
        mults.SetValue(1, 4)
        mults.SetValue(2, 4)

        bspline = Geom_BSplineCurve(poles, knots, mults, 3)
        edge = BRepBuilderAPI_MakeEdge(bspline, 0.0, 1.0).Edge()

        curve_info = stp_mod.classify_edge(edge, n_samples=5)
        assert curve_info.kind == "bspline"

        if curve_info.support_points:
            first = curve_info.support_points[0]
            last = curve_info.support_points[-1]
            assert first != pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
            assert last != pytest.approx((1.0, 0.0, 0.0), abs=1e-6)

    def test_arc_over_half_pi_becomes_bspline(self, caplog):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.Geom import Geom_Circle
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

        ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        circle = Geom_Circle(ax, 1.0)
        import math
        edge = BRepBuilderAPI_MakeEdge(circle, 0.0, math.radians(200)).Edge()

        with caplog.at_level(logging.WARNING):
            curve_info = stp_mod.classify_edge(edge, n_samples=10)

        assert curve_info.kind == "bspline"
        assert any("arc" in r.message.lower() or "warn" in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# OCP-based load_step tests
# ---------------------------------------------------------------------------

def _write_box_step(tmp_path: Path, dx=1.0, dy=1.0, dz=1.0) -> Path:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    box = BRepPrimAPI_MakeBox(dx, dy, dz).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(box, STEPControl_AsIs)
    out_path = tmp_path / "box.stp"
    status = writer.Write(str(out_path))
    assert status == IFSelect_RetDone, "STEPControl_Writer failed"
    return out_path


def _write_two_boxes_step(tmp_path: Path) -> Path:
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    box1 = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(1.0, 0.0, 0.0))
    box2 = BRepBuilderAPI_Transform(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(), trsf).Shape()

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, box1)
    builder.Add(compound, box2)

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    out_path = tmp_path / "two_boxes.stp"
    status = writer.Write(str(out_path))
    assert status == IFSelect_RetDone, "STEPControl_Writer failed"
    return out_path


@requires_ocp
class TestIsHexTopology:
    def test_box_is_hex(self, tmp_path):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        exp = TopExp_Explorer(box, TopAbs_SOLID)
        assert exp.More()
        solid = TopoDS.Solid_s(exp.Current())
        assert stp_mod._is_hex_topology(solid)

    def test_cylinder_is_not_hex(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        cyl = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
        exp = TopExp_Explorer(cyl, TopAbs_SOLID)
        assert exp.More()
        solid = TopoDS.Solid_s(exp.Current())
        assert not stp_mod._is_hex_topology(solid)


@requires_ocp
class TestLoadStepOneBox:
    def test_returns_one_hex_candidate(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        assert len(candidates) == 1

    def test_candidate_has_8_vertices(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        assert len(candidates[0].vertex_indices) == 8

    def test_candidate_has_12_edges(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        assert len(candidates[0].edges) == 12

    def test_candidate_has_6_faces(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        assert len(candidates[0].faces) == 6

    def test_validate_hex_passes(self, tmp_path):
        from meshing_utils import validate_hex
        stp_path = _write_box_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        validate_hex(candidates[0])

    def test_pool_has_8_unique_points(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        _candidates, pool, _unit = stp_mod.load_step(stp_path)
        assert len(pool) == 8

    def test_unit_returned(self, tmp_path):
        stp_path = _write_box_step(tmp_path)
        _candidates, _pool, unit = stp_mod.load_step(stp_path)
        assert isinstance(unit, str)


@requires_ocp
class TestLoadStepTwoBoxes:
    def test_returns_two_candidates(self, tmp_path):
        stp_path = _write_two_boxes_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        assert len(candidates) == 2

    def test_both_validate_hex(self, tmp_path):
        from meshing_utils import validate_hex
        stp_path = _write_two_boxes_step(tmp_path)
        candidates, _pool, _unit = stp_mod.load_step(stp_path)
        for c in candidates:
            validate_hex(c)

    def test_shared_face_consistency(self, tmp_path):
        from meshing_utils import (
            check_global_face_consistency,
            ensure_right_handed,
            order_hex_vertices,
            validate_hex,
        )
        stp_path = _write_two_boxes_step(tmp_path)
        candidates, pool, _unit = stp_mod.load_step(stp_path)
        orderings = []
        faces_per_block = []
        for c in candidates:
            validate_hex(c)
            o = order_hex_vertices(c, pool)
            o = ensure_right_handed(o, pool)
            orderings.append(o)
            faces_per_block.append(c.faces)
        check_global_face_consistency(orderings, faces_per_block)


@requires_ocp
class TestLoadStepErrorPaths:
    def test_nonexistent_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to read"):
            stp_mod.load_step(tmp_path / "does_not_exist.stp")

    def test_empty_file_raises_runtime_error(self, tmp_path):
        p = tmp_path / "empty.stp"
        p.write_text("")
        with pytest.raises(RuntimeError):
            stp_mod.load_step(p)

    def test_no_hex_solids_raises_runtime_error(self, tmp_path):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

        cyl = BRepPrimAPI_MakeCylinder(0.5, 2.0).Shape()
        writer = STEPControl_Writer()
        writer.Transfer(cyl, STEPControl_AsIs)
        out_path = tmp_path / "only_cyl.stp"
        status = writer.Write(str(out_path))
        assert status == IFSelect_RetDone

        with pytest.raises(RuntimeError, match="No hexahedral"):
            stp_mod.load_step(out_path)


class TestReadStepSolidNames:
    def test_extracts_block_names(self, tmp_path):
        content = (
            "#1=MANIFOLD_SOLID_BREP('blockA',#10);\n"
            "#2=MANIFOLD_SOLID_BREP('blockB',#20);\n"
            "#3=MANIFOLD_SOLID_BREP('blockC',#30);\n"
        )
        p = tmp_path / "test.stp"
        p.write_text(content, encoding="utf-8")
        result = stp_mod._read_step_solid_names(p)
        assert result == ["blockA", "blockB", "blockC"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.stp"
        p.write_text("", encoding="utf-8")
        result = stp_mod._read_step_solid_names(p)
        assert result == []


class TestDefaultPatchNameCli:
    def test_default_patch_name_written_to_file(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
                default_patch_name="walls",
            )

        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "name walls;" in content

    def test_default_patch_name_default_value(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )

        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "name defaultFaces;" in content


class TestParseFractions:
    def test_valid_fractions(self):
        result = parse_fractions("(0.1 0.1 0.1)")
        assert result == pytest.approx((0.1, 0.1, 0.1))

    def test_valid_fractions_with_spaces(self):
        result = parse_fractions("( 0.5  0.25  0.1 )")
        assert result == pytest.approx((0.5, 0.25, 0.1))

    def test_zero_fraction_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_fractions("(0.0 0.1 0.1)")

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError, match="fractions"):
            parse_fractions("(-0.1 0.1 0.1)")

    def test_missing_component_raises(self):
        with pytest.raises(ValueError, match="fractions"):
            parse_fractions("(0.1 0.1)")

    def test_no_parens_raises(self):
        with pytest.raises(ValueError, match="fractions"):
            parse_fractions("0.1 0.1 0.1")


class TestRunWithFractions:
    def test_fractions_produces_non_unit_cell_counts(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        coords = [
            (0.0, 0.0,  0.0), (10.0, 0.0,  0.0), (10.0, 10.0,  0.0), (0.0, 10.0,  0.0),
            (0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (10.0, 10.0, 10.0), (0.0, 10.0, 10.0),
        ]
        for c in coords:
            pool.add_or_get(c)
        faces = [
            (0, 3, 2, 1), (4, 5, 6, 7),
            (0, 1, 5, 4), (3, 7, 6, 2),
            (1, 2, 6, 5), (0, 4, 7, 3),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        stub = HexCandidate(
            vertex_indices=list(range(8)),
            faces=faces,
            edges=edges,
            edge_curves={},
            label="bigBlock",
        )

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
                fractions=(0.5, 0.5, 0.5),
                use_legacy_cell_count=True,
            )

        bmd = BlockMeshDict(case_dir / "system" / "blockMeshDict")
        block = bmd.blocks[0]
        assert block.cells == [2, 2, 2], f"Expected [2, 2, 2], got {block.cells}"

    def test_no_fractions_default_to_1(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
                fractions=None,
            )

        bmd = BlockMeshDict(case_dir / "system" / "blockMeshDict")
        block = bmd.blocks[0]
        assert block.cells == [1, 1, 1], f"Expected [1, 1, 1], got {block.cells}"


class TestGlobalFaceConsistencyCalled:
    def test_global_face_consistency_called(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "mm")), patch(
            "meshing_utils.operations.stp_pipeline.pipeline.check_global_face_consistency",
        ) as mock_check:
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
            assert mock_check.called, (
                "check_global_face_consistency must be called within run()"
            )


# ---------------------------------------------------------------------------
# Append mode tests (adapted from test_stpToBlockMeshDict_append.py)
# ---------------------------------------------------------------------------

class TestT1BasicAppend:
    def test_two_blocks_after_append(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(case_dir, [stub1], pool1)

        bmd_path = case_dir / "system" / "blockMeshDict"
        bmd_after_first = BlockMeshDict(bmd_path)
        assert len(bmd_after_first.blocks) == 1
        assert len(bmd_after_first.vertices) == 8

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=2.0, label="block1")
        _run_tool(case_dir, [stub2], pool2)

        bmd_after_second = BlockMeshDict(bmd_path)
        assert len(bmd_after_second.blocks) == 2
        assert len(bmd_after_second.vertices) == 16


class TestT2VertexDedup:
    def test_shared_vertex_not_duplicated(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(case_dir, [stub1], pool1)

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=1.0, label="block1")
        _run_tool(case_dir, [stub2], pool2)

        bmd_path = case_dir / "system" / "blockMeshDict"
        bmd = BlockMeshDict(bmd_path)
        assert len(bmd.vertices) == 12, (
            f"Expected 12 unique vertices (4 shared), got {len(bmd.vertices)}"
        )


class TestT3OverwriteRegression:
    def test_overwrite_creates_backup_and_discards_existing(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(case_dir, [stub1], pool1)

        bmd_path = case_dir / "system" / "blockMeshDict"
        content_first = bmd_path.read_text()

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=5.0, label="newBlock")
        _run_tool(case_dir, [stub2], pool2, overwrite=True)

        bak_path = bmd_path.with_suffix(".bak")
        assert bak_path.exists()
        assert bak_path.read_text() == content_first

        bmd = BlockMeshDict(bmd_path)
        assert len(bmd.blocks) == 1
        assert bmd.blocks[0].name == "newBlock"


class TestT5BlockNameCollision:
    def test_suffix_applied_on_name_collision(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(case_dir, [stub1], pool1)

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=5.0, label="block0")
        _run_tool(case_dir, [stub2], pool2, name_collision="suffix")

        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "block0_2" in content

    def test_error_policy_raises_on_collision(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(case_dir, [stub1], pool1)

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=5.0, label="block0")
        with pytest.raises((ValueError, SystemExit)):
            _run_tool(case_dir, [stub2], pool2, name_collision="error")


class TestT6EdgeConflict:
    def test_non_strict_duplicate_arc_edge_no_exception(self, tmp_path, caplog):
        bmd = BlockMeshDict()
        logger_obj = logging.getLogger("test_t6")
        edge_origin = {}
        arc1 = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.5, 0.5))
        stp_mod.add_edge_with_conflict_check(
            bmd, "v0", "v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=arc1, block_name="blockA",
            edge_origin=edge_origin, strict=False, logger=logger_obj,
        )

        arc2 = CurveInfo(kind="arc", support_points=[], arc_midpoint=(0.5, 0.6, 0.5))
        with caplog.at_level(logging.WARNING):
            stp_mod.add_edge_with_conflict_check(
                bmd, "v0", "v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=arc2, block_name="blockB",
                edge_origin=edge_origin, strict=False, logger=logger_obj,
            )

        assert len(list(bmd.edges)) == 1
        assert any("conflict" in r.message.lower() for r in caplog.records)


class TestT7DefaultPatchPreservation:
    def test_existing_default_patch_name_preserved_without_explicit_flag(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool1 = PointPool(tol=1e-6)
        stub1 = _make_unit_cube_candidate(pool1, offset=0.0, label="block0")
        _run_tool(
            case_dir, [stub1], pool1,
            default_patch_name="walls",
            default_patch_name_explicit=True,
        )

        bmd_path = case_dir / "system" / "blockMeshDict"
        assert "name walls;" in bmd_path.read_text()

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=2.0, label="block1")
        _run_tool(
            case_dir, [stub2], pool2,
            default_patch_name="defaultFaces",
            default_patch_name_explicit=False,
        )

        content = bmd_path.read_text()
        assert "name walls;" in content
        assert "name defaultFaces;" not in content


# ---------------------------------------------------------------------------
# Edge-conflict tests (adapted from test_stpToBlockMeshDict_edge_conflicts.py)
# ---------------------------------------------------------------------------

_PIPELINE_LOGGER = "meshing_utils.operations.stp_pipeline"


def _bmd_with_arc_edge(v_start: str, v_end: str) -> BlockMeshDict:
    """Return a BlockMeshDict that already contains one arc edge."""
    bmd = BlockMeshDict()
    bmd.vertices.add(Vertex(name_or_string=v_start, coords=[0.0, 0.0, 0.0]))
    bmd.vertices.add(Vertex(name_or_string=v_end,   coords=[1.0, 0.0, 0.0]))
    arc_edge = Edge(
        type_or_string="arc",
        v_start=v_start,
        v_end=v_end,
        coords=[0.5, 0.05, 0.0],
    )
    bmd.edges.add(arc_edge)
    return bmd


def _empty_bmd() -> BlockMeshDict:
    return BlockMeshDict()


def _line_curve_info() -> CurveInfo:
    return CurveInfo(kind="line", support_points=[])


def _arc_curve_info(midpoint=(0.5, 0.05, 0.0)) -> CurveInfo:
    return CurveInfo(kind="arc", support_points=[], arc_midpoint=midpoint)


class TestLineVsExistingArcNonStrict:
    def test_warns_with_both_block_names(self, caplog):
        bmd = _bmd_with_arc_edge("v0", "v1")
        edge_origin: dict[frozenset[str], str] = {
            frozenset({"v0", "v1"}): "blockA_fillet"
        }

        with caplog.at_level(logging.WARNING, logger=_PIPELINE_LOGGER):
            add_edge_with_conflict_check(
                bmd=bmd,
                name_a="v0",
                name_b="v1",
                coord_a=(0.0, 0.0, 0.0),
                coord_b=(1.0, 0.0, 0.0),
                curve_info=_line_curve_info(),
                block_name="blockB_main",
                edge_origin=edge_origin,
                strict=False,
                logger=logging.getLogger(_PIPELINE_LOGGER),
            )

        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "blockA_fillet" in msg
        assert "blockB_main" in msg
        assert "v0" in msg
        assert "v1" in msg
        assert "arc" in msg.lower()

    def test_arc_edge_unchanged_after_conflict(self):
        bmd = _bmd_with_arc_edge("v0", "v1")
        edge_origin: dict[frozenset[str], str] = {
            frozenset({"v0", "v1"}): "blockA_fillet"
        }

        add_edge_with_conflict_check(
            bmd=bmd,
            name_a="v0",
            name_b="v1",
            coord_a=(0.0, 0.0, 0.0),
            coord_b=(1.0, 0.0, 0.0),
            curve_info=_line_curve_info(),
            block_name="blockB_main",
            edge_origin=edge_origin,
            strict=False,
            logger=logging.getLogger(_PIPELINE_LOGGER),
        )

        assert len(list(bmd.edges)) == 1
        edge = next(iter(bmd.edges))
        assert edge.type == "arc"


class TestLineVsExistingArcStrict:
    def test_raises_value_error(self):
        bmd = _bmd_with_arc_edge("v0", "v1")
        edge_origin: dict[frozenset[str], str] = {
            frozenset({"v0", "v1"}): "blockA_fillet"
        }

        with pytest.raises(ValueError, match="blockA_fillet"):
            add_edge_with_conflict_check(
                bmd=bmd,
                name_a="v0",
                name_b="v1",
                coord_a=(0.0, 0.0, 0.0),
                coord_b=(1.0, 0.0, 0.0),
                curve_info=_line_curve_info(),
                block_name="blockB_main",
                edge_origin=edge_origin,
                strict=True,
                logger=logging.getLogger(_PIPELINE_LOGGER),
            )


class TestTwoSolidsShareLineNoWarning:
    def test_no_warning_and_no_edge_added(self, caplog):
        bmd = _empty_bmd()
        bmd.vertices.add(Vertex(name_or_string="v0", coords=[0.0, 0.0, 0.0]))
        bmd.vertices.add(Vertex(name_or_string="v1", coords=[1.0, 0.0, 0.0]))

        edge_origin: dict[frozenset[str], str] = {}
        log = logging.getLogger(_PIPELINE_LOGGER)

        with caplog.at_level(logging.WARNING, logger=_PIPELINE_LOGGER):
            add_edge_with_conflict_check(
                bmd=bmd, name_a="v0", name_b="v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=_line_curve_info(), block_name="blockA",
                edge_origin=edge_origin, strict=False, logger=log,
            )
            add_edge_with_conflict_check(
                bmd=bmd, name_a="v0", name_b="v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=_line_curve_info(), block_name="blockB",
                edge_origin=edge_origin, strict=False, logger=log,
            )

        assert len(caplog.records) == 0
        assert len(list(bmd.edges)) == 0


class TestArcVsExistingArcStillWarns:
    def test_curved_vs_curved_warns(self, caplog):
        bmd = _bmd_with_arc_edge("v0", "v1")
        edge_origin: dict[frozenset[str], str] = {
            frozenset({"v0", "v1"}): "blockA_fillet"
        }

        with caplog.at_level(logging.WARNING, logger=_PIPELINE_LOGGER):
            add_edge_with_conflict_check(
                bmd=bmd, name_a="v0", name_b="v1",
                coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
                curve_info=_arc_curve_info(midpoint=(0.5, 0.1, 0.0)),
                block_name="blockB_fillet",
                edge_origin=edge_origin, strict=False,
                logger=logging.getLogger(_PIPELINE_LOGGER),
            )

        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "v0" in msg or "v1" in msg


class TestArcAddedThenOriginTracked:
    def test_origin_tracked_after_arc(self):
        bmd = _empty_bmd()
        bmd.vertices.add(Vertex(name_or_string="v0", coords=[0.0, 0.0, 0.0]))
        bmd.vertices.add(Vertex(name_or_string="v1", coords=[1.0, 0.0, 0.0]))

        edge_origin: dict[frozenset[str], str] = {}

        add_edge_with_conflict_check(
            bmd=bmd, name_a="v0", name_b="v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=_arc_curve_info(), block_name="blockA",
            edge_origin=edge_origin, strict=False,
            logger=logging.getLogger(_PIPELINE_LOGGER),
        )

        key = frozenset({"v0", "v1"})
        assert key in edge_origin
        assert edge_origin[key] == "blockA"
        assert len(list(bmd.edges)) == 1
        assert next(iter(bmd.edges)).type == "arc"


class TestBSplineSampleAlignment:
    def test_inverted_samples_are_reversed_on_insert(self):
        bmd = _empty_bmd()
        bmd.vertices.add(Vertex(name_or_string="v0", coords=[0.0, 0.0, 0.0]))
        bmd.vertices.add(Vertex(name_or_string="v1", coords=[1.0, 0.0, 0.0]))

        inverted = [
            (0.95, 0.05, 0.0),
            (0.50, 0.10, 0.0),
            (0.05, 0.05, 0.0),
        ]
        ci = CurveInfo(kind="bspline", support_points=inverted)
        edge_origin: dict[frozenset[str], str] = {}

        add_edge_with_conflict_check(
            bmd=bmd, name_a="v0", name_b="v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=ci, block_name="blockA",
            edge_origin=edge_origin, strict=False,
            logger=logging.getLogger(_PIPELINE_LOGGER),
        )

        edges = list(bmd.edges)
        assert len(edges) == 1
        pts = edges[0].points
        assert pts[0] == [0.05, 0.05, 0.0]
        assert pts[-1] == [0.95, 0.05, 0.0]

    def test_aligned_samples_are_kept_as_is(self):
        bmd = _empty_bmd()
        bmd.vertices.add(Vertex(name_or_string="v0", coords=[0.0, 0.0, 0.0]))
        bmd.vertices.add(Vertex(name_or_string="v1", coords=[1.0, 0.0, 0.0]))

        aligned = [
            (0.05, 0.05, 0.0),
            (0.50, 0.10, 0.0),
            (0.95, 0.05, 0.0),
        ]
        ci = CurveInfo(kind="bspline", support_points=aligned)
        edge_origin: dict[frozenset[str], str] = {}

        add_edge_with_conflict_check(
            bmd=bmd, name_a="v0", name_b="v1",
            coord_a=(0.0, 0.0, 0.0), coord_b=(1.0, 0.0, 0.0),
            curve_info=ci, block_name="blockA",
            edge_origin=edge_origin, strict=False,
            logger=logging.getLogger(_PIPELINE_LOGGER),
        )

        edges = list(bmd.edges)
        assert len(edges) == 1
        pts = edges[0].points
        assert pts[0] == [0.05, 0.05, 0.0]
        assert pts[-1] == [0.95, 0.05, 0.0]


# ---------------------------------------------------------------------------
# load_solids_with_names integration: verify names reach HexCandidate.label
# ---------------------------------------------------------------------------

class TestLoadStepUsesLoadSolidsWithNames:
    """Verify that load_step delegates solid loading to load_solids_with_names
    and that the resolved names are forwarded to HexCandidate.label."""

    def test_solid_name_from_load_solids_with_names_propagates_to_candidate(
        self, tmp_path
    ):
        """When load_solids_with_names returns a NamedSolid with a known name,
        the resulting HexCandidate produced by load_step must carry that name
        in its .label attribute.

        The test avoids real OCC calls by patching load_solids_with_names and
        _is_hex_topology / _solid_to_hex_candidate so that one fake solid
        passes through the hex-filter and produces a candidate with the
        expected label.  The OCC guard-import at the top of load_step is
        bypassed by injecting a lightweight mock module into sys.modules.
        """
        from types import ModuleType
        from unittest.mock import patch

        from meshing_utils.cad.step_names import NamedSolid

        # Create a dummy STEP file path (content irrelevant — all OCC calls mocked)
        dummy_stp = tmp_path / "part.stp"
        dummy_stp.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        )

        sentinel_solid = object()
        named_solid = NamedSolid(
            solid=sentinel_solid,
            name="myBlockName",
            source="step_id",
        )

        expected_candidate = HexCandidate(
            vertex_indices=list(range(8)),
            faces=[],
            edges=[],
            edge_curves={},
            label="myBlockName",
        )

        # Build minimal OCP stub modules so the guard-import inside load_step passes
        ocp_pkg = ModuleType("OCP")
        step_mod = ModuleType("OCP.STEPControl")
        iface_mod = ModuleType("OCP.IFSelect")
        step_mod.STEPControl_Reader = MagicMock()
        iface_mod.IFSelect_RetDone = object()

        patched_modules = {
            "OCP": ocp_pkg,
            "OCP.STEPControl": step_mod,
            "OCP.IFSelect": iface_mod,
        }

        with (
            patch.dict(sys.modules, patched_modules),
            patch(
                "meshing_utils.operations.stp_pipeline.loading.load_solids_with_names",
                return_value=[named_solid],
            ),
            patch(
                "meshing_utils.operations.stp_pipeline.loading._is_hex_topology",
                return_value=True,
            ),
            patch(
                "meshing_utils.operations.stp_pipeline.loading._solid_to_hex_candidate",
                return_value=expected_candidate,
            ),
        ):
            candidates, _result_pool, _unit = stp_mod.load_step(dummy_stp)

        assert len(candidates) == 1, "Expected exactly one candidate"
        assert candidates[0].label == "myBlockName", (
            f"Expected label 'myBlockName', got {candidates[0].label!r}"
        )


# ---------------------------------------------------------------------------
# convertToMeters handling
# ---------------------------------------------------------------------------

class TestStepUnitToMeters:
    def test_mm(self):
        assert stp_mod.step_unit_to_meters("MM") == pytest.approx(0.001)

    def test_m_lowercase(self):
        assert stp_mod.step_unit_to_meters("m") == 1.0

    def test_inch(self):
        assert stp_mod.step_unit_to_meters("IN") == pytest.approx(0.0254)

    def test_unknown_returns_none(self):
        assert stp_mod.step_unit_to_meters("bogus") is None

    def test_none_returns_none(self):
        assert stp_mod.step_unit_to_meters(None) is None

    def test_empty_returns_none(self):
        assert stp_mod.step_unit_to_meters("") is None


class TestConvertToMeters:
    def _read_ctm(self, bmd_path: Path) -> float:
        return BlockMeshDict(bmd_path).convertToMeters

    def test_new_file_derives_from_step_unit_mm(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)
        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
        assert self._read_ctm(case_dir / "system" / "blockMeshDict") == pytest.approx(0.001)

    def test_new_file_unknown_unit_falls_back_to_one(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)
        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "unknown")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
        assert self._read_ctm(case_dir / "system" / "blockMeshDict") == 1.0

    def test_overwrite_preserves_existing_convert_to_meters(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        # First run creates the file with MM -> 0.001
        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)
        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
        bmd_path = case_dir / "system" / "blockMeshDict"
        assert self._read_ctm(bmd_path) == pytest.approx(0.001)

        # Manually bump the value to a non-default to prove preservation
        existing = BlockMeshDict(bmd_path)
        existing.convertToMeters = 0.0254
        existing.write(bmd_path)

        # Overwrite run with STEP unit MM must NOT clobber 0.0254
        pool2 = PointPool(tol=1e-6)
        stub2 = _make_hex_candidate_stub(pool2)
        with patch.object(stp_mod, "load_step", return_value=([stub2], pool2, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=True,
            )
        assert self._read_ctm(bmd_path) == pytest.approx(0.0254)

    def test_append_preserves_existing_convert_to_meters(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)

        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)
        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
        bmd_path = case_dir / "system" / "blockMeshDict"

        existing = BlockMeshDict(bmd_path)
        existing.convertToMeters = 0.5
        existing.write(bmd_path)

        pool2 = PointPool(tol=1e-6)
        stub2 = _make_unit_cube_candidate(pool2, offset=10.0, label="Other")
        with patch.object(stp_mod, "load_step", return_value=([stub2], pool2, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
            )
        assert self._read_ctm(bmd_path) == pytest.approx(0.5)

    def test_explicit_override_wins_over_step_unit(self, tmp_path):
        case_dir = _make_mock_case_dir(tmp_path)
        _place_stub_stp(case_dir)
        pool = PointPool(tol=1e-6)
        stub = _make_hex_candidate_stub(pool)

        with patch.object(stp_mod, "load_step", return_value=([stub], pool, "MM")):
            _run(
                case_dir=case_dir,
                origin=(0.0, 0.0, 0.0),
                tol=1e-6,
                n_samples=20,
                name_collision="suffix",
                strict=False,
                overwrite=False,
                convert_to_meters=0.01,
            )
        assert self._read_ctm(case_dir / "system" / "blockMeshDict") == pytest.approx(0.01)
