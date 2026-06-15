"""Unit tests for meshing_utils.cad.step_loader.

Migrated from tests/common/test_step_loader.py.
OCP-dependent tests are guarded by the ``requires_ocp`` skip marker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meshing_utils import (
    find_single_step_file,
    read_step_solid_names,
    read_step_unit,
)

# Skip marker for tests that build real OCP (cadquery-ocp) shapes. These are
# skipped when OCP is not importable (e.g. on a Python version without a
# matching cadquery-ocp wheel) rather than erroring.
requires_ocp = pytest.mark.skipif(
    importlib.util.find_spec("OCP") is None,
    reason="requires OCP (cadquery-ocp); not installed in this environment",
)


# ---------------------------------------------------------------------------
# find_single_step_file
# ---------------------------------------------------------------------------

class TestFindSingleStepFile:
    def test_happy_path_stp(self, tmp_path: Path):
        """Exactly one .stp file is found and returned."""
        f = tmp_path / "part.stp"
        f.write_text("dummy")
        result = find_single_step_file(tmp_path)
        assert result == f

    def test_happy_path_step(self, tmp_path: Path):
        """Exactly one .step file is found and returned."""
        f = tmp_path / "part.step"
        f.write_text("dummy")
        result = find_single_step_file(tmp_path)
        assert result == f

    def test_no_files_raises(self, tmp_path: Path):
        """FileNotFoundError when the directory contains no STEP files."""
        with pytest.raises(FileNotFoundError, match="No STEP file"):
            find_single_step_file(tmp_path)

    def test_two_files_raises(self, tmp_path: Path):
        """ValueError when the directory contains more than one STEP file."""
        (tmp_path / "a.stp").write_text("dummy")
        (tmp_path / "b.stp").write_text("dummy")
        with pytest.raises(ValueError, match="Multiple STEP files"):
            find_single_step_file(tmp_path)

    def test_mixed_extensions_two_files_raises(self, tmp_path: Path):
        """ValueError when mixing .stp and .step results in two files."""
        (tmp_path / "a.stp").write_text("dummy")
        (tmp_path / "b.step").write_text("dummy")
        with pytest.raises(ValueError, match="Multiple STEP files"):
            find_single_step_file(tmp_path)

    def test_returns_path_object(self, tmp_path: Path):
        """Returned value is a pathlib.Path instance."""
        (tmp_path / "part.stp").write_text("dummy")
        result = find_single_step_file(tmp_path)
        assert isinstance(result, Path)

    def test_accepts_string_dir(self, tmp_path: Path):
        """Accepts a string path as *geometry_dir*."""
        (tmp_path / "part.stp").write_text("dummy")
        result = find_single_step_file(str(tmp_path))
        assert result.name == "part.stp"


# ---------------------------------------------------------------------------
# read_step_solid_names
# ---------------------------------------------------------------------------

class TestReadStepSolidNames:
    def test_manifold_solid_brep(self, tmp_path: Path):
        content = (
            "ISO-10303-21;\n"
            "#1 = MANIFOLD_SOLID_BREP('MySolid', #2);\n"
        )
        f = tmp_path / "part.stp"
        f.write_text(content)
        names = read_step_solid_names(f)
        assert names == ["MySolid"]

    def test_multiple_solids(self, tmp_path: Path):
        content = (
            "#1 = MANIFOLD_SOLID_BREP('Solid1', #2);\n"
            "#3 = MANIFOLD_SOLID_BREP('Solid2', #4);\n"
        )
        f = tmp_path / "part.stp"
        f.write_text(content)
        names = read_step_solid_names(f)
        assert names == ["Solid1", "Solid2"]

    def test_no_matches(self, tmp_path: Path):
        f = tmp_path / "part.stp"
        f.write_text("no solids here\n")
        assert read_step_solid_names(f) == []

    def test_missing_file(self, tmp_path: Path):
        f = tmp_path / "nonexistent.stp"
        assert read_step_solid_names(f) == []

    def test_empty_name(self, tmp_path: Path):
        content = "#1 = MANIFOLD_SOLID_BREP('', #2);\n"
        f = tmp_path / "part.stp"
        f.write_text(content)
        names = read_step_solid_names(f)
        assert names == [""]


# ---------------------------------------------------------------------------
# read_step_unit
# ---------------------------------------------------------------------------

class TestReadStepUnit:
    def test_mm_unit(self, tmp_path: Path):
        content = "LENGTH_UNIT('MM', ());\n"
        f = tmp_path / "part.stp"
        f.write_text(content)
        assert read_step_unit(f) == "MM"

    def test_unknown_when_missing(self, tmp_path: Path):
        f = tmp_path / "part.stp"
        f.write_text("no unit here\n")
        assert read_step_unit(f) == "unknown"

    def test_missing_file(self, tmp_path: Path):
        f = tmp_path / "nonexistent.stp"
        assert read_step_unit(f) == "unknown"


# ---------------------------------------------------------------------------
# Refactoring guard: stpToBMD still works after refactor
# ---------------------------------------------------------------------------

class TestRefactoringGuard:
    def test_stp_to_bmd_cli_importable(self):
        """cli.stp_to_bmd must be importable after the refactor."""
        from meshing_utils.cli import stp_to_bmd  # noqa: F401

    def test_read_step_unit_still_importable_from_stp_module(self):
        """read_step_unit is re-exported from common and callable."""
        from meshing_utils import read_step_unit as ru
        assert callable(ru)

    def test_find_single_step_file_callable(self):
        from meshing_utils import find_single_step_file as f
        assert callable(f)


# ---------------------------------------------------------------------------
# OCC-dependent: load_step_solids
# ---------------------------------------------------------------------------

@requires_ocp
class TestLoadStepSolids:
    def test_load_box_solid(self, tmp_path: Path):
        """load_step_solids returns at least one solid from a box STEP file."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

        from meshing_utils import load_step_solids

        box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        stp_file = tmp_path / "box.stp"
        writer = STEPControl_Writer()
        writer.Transfer(box, STEPControl_AsIs)
        status = writer.Write(str(stp_file))
        assert status == IFSelect_RetDone

        pairs = load_step_solids(stp_file)
        assert len(pairs) >= 1
        solid, _label = pairs[0]
        assert solid is not None

    def test_load_solids_returns_list_of_tuples(self, tmp_path: Path):
        """Each element is a (solid, Optional[str]) tuple."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

        from meshing_utils import load_step_solids

        box = BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape()
        stp_file = tmp_path / "box2.stp"
        writer = STEPControl_Writer()
        writer.Transfer(box, STEPControl_AsIs)
        writer.Write(str(stp_file))

        pairs = load_step_solids(stp_file)
        for item in pairs:
            assert len(item) == 2
            _solid, label = item
            assert label is None or isinstance(label, str)

    def test_missing_file_raises(self, tmp_path: Path):
        """RuntimeError when the STEP file does not exist."""
        from meshing_utils import load_step_solids
        with pytest.raises(RuntimeError, match="Failed to read"):
            load_step_solids(tmp_path / "nonexistent.stp")


# ---------------------------------------------------------------------------
# load_solids_with_names  (mock-based, no real OCC geometry needed)
# ---------------------------------------------------------------------------

class _FakeSolid:
    """Sentinel for a TopoDS_Solid."""


def _mock_occ_imports(mock_reader_instance):
    """Return a context manager that patches the OCP imports inside load_solids_with_names.

    The function imports OCP lazily (``from OCP.STEPControl import ...``),
    so we must inject fake modules into ``sys.modules`` before calling it.
    """
    import sys
    import types

    # Fake IFSelect module
    fake_iface = types.ModuleType("OCP.IFSelect")
    fake_iface.IFSelect_RetDone = 1  # sentinel value

    # Fake STEPControl module
    fake_step = types.ModuleType("OCP.STEPControl")
    fake_step.STEPControl_Reader = MagicMock(return_value=mock_reader_instance)

    # Fake OCP parent package
    fake_ocp = sys.modules.get("OCP") or types.ModuleType("OCP")

    return patch.dict(
        sys.modules,
        {
            "OCP": fake_ocp,
            "OCP.STEPControl": fake_step,
            "OCP.IFSelect": fake_iface,
        },
    )


class TestLoadSolidsWithNames:
    """Mock-based tests for load_solids_with_names."""

    def test_returns_named_solids(self, tmp_path: Path):
        """load_solids_with_names returns a list of NamedSolid objects."""
        from meshing_utils.cad.step_names import NamedSolid

        stp = tmp_path / "test.stp"
        stp.write_text(
            "ISO-10303-21;\nDATA;\n"
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','fluid','',#3,#4,$);\n"
            "ENDSEC;\nEND-ISO-10303-21;\n",
            encoding="utf-8",
        )

        fake_solid = _FakeSolid()
        mock_reader = MagicMock()
        mock_reader.ReadFile.return_value = 1  # IFSelect_RetDone

        expected = [NamedSolid(solid=fake_solid, name="fluid", source="assembly")]

        with _mock_occ_imports(mock_reader), \
             patch("meshing_utils.cad.step_loader.explore_solids", return_value=[fake_solid]), \
             patch("meshing_utils.cad.step_loader.extract_solid_names", return_value=expected):
            from meshing_utils.cad.step_loader import load_solids_with_names
            result = load_solids_with_names(stp)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NamedSolid)
        assert result[0].name == "fluid"

    def test_missing_file_raises_runtime_error(self, tmp_path: Path):
        """RuntimeError when STEP file cannot be read (reader returns failure)."""
        stp = tmp_path / "bad.stp"
        stp.write_text("not a step file", encoding="utf-8")

        mock_reader = MagicMock()
        mock_reader.ReadFile.return_value = 99  # not IFSelect_RetDone (which is 1)

        with _mock_occ_imports(mock_reader):
            from meshing_utils.cad.step_loader import load_solids_with_names
            with pytest.raises(RuntimeError, match="Failed to read"):
                load_solids_with_names(stp)

    def test_no_solids_raises_runtime_error(self, tmp_path: Path):
        """RuntimeError when no solids are found in the STEP file."""
        stp = tmp_path / "empty.stp"
        stp.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")

        mock_reader = MagicMock()
        mock_reader.ReadFile.return_value = 1  # IFSelect_RetDone

        with _mock_occ_imports(mock_reader), \
             patch("meshing_utils.cad.step_loader.explore_solids", return_value=[]), \
             patch("meshing_utils.cad.step_loader.read_step_xcaf", return_value=[]):
            from meshing_utils.cad.step_loader import load_solids_with_names
            with pytest.raises(RuntimeError, match="No solids found"):
                load_solids_with_names(stp)

    def test_occ_not_installed_raises_import_error(self, tmp_path: Path):
        """ImportError when OCP is not available."""
        stp = tmp_path / "test.stp"
        stp.write_text("dummy", encoding="utf-8")

        import sys

        # Remove OCP from sys.modules to simulate it not being installed
        ocp_keys = [k for k in sys.modules if k.startswith("OCP")]
        # Patch builtins.__import__ to raise for OCP
        import builtins
        real_import = builtins.__import__

        def _no_ocp(name, *args, **kwargs):
            if name.startswith("OCP"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_ocp), \
             patch.dict(sys.modules, {k: None for k in ocp_keys}):  # type: ignore[misc]
            from meshing_utils.cad.step_loader import load_solids_with_names
            with pytest.raises(ImportError, match="cadquery-ocp"):
                load_solids_with_names(stp)
