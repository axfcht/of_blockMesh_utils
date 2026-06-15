"""Unit tests for meshing_utils.cli.extract_patches.

OCC-dependent integration tests are skipped when OCC is not installed.
Smoke tests (--help, import, missing files) run unconditionally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import meshing_utils.operations.extract_patches as op_mod
from meshing_utils.cad.step_names import NamedSolid
from meshing_utils.cli.extract_patches import main, run

# ---------------------------------------------------------------------------
# Minimal blockMeshDict fixture
# ---------------------------------------------------------------------------

_MINIMAL_BMD = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
\tformat      ascii;
\tclass       dictionary;
\tobject      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

geometry
{
}

vertices
(
\tname v0 (0.0 0.0 0.0)
\tname v1 (1.0 0.0 0.0)
\tname v2 (1.0 1.0 0.0)
\tname v3 (0.0 1.0 0.0)
\tname v4 (0.0 0.0 1.0)
\tname v5 (1.0 0.0 1.0)
\tname v6 (1.0 1.0 1.0)
\tname v7 (0.0 1.0 1.0)
);

edges
(
);

blocks
(
\tname testBlock hex (v0 v1 v2 v3 v4 v5 v6 v7) (1 1 1) simpleGrading (1 1 1)
);

defaultPatch
{
\tname defaultFaces;
\ttype empty;
}

boundary
(
);

// ************************************************************************* //
"""


def _write_bmd(case_dir: Path, content: str = _MINIMAL_BMD) -> Path:
    """Write *content* as ``system/blockMeshDict`` and return the file path."""
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    bmd_path = system_dir / "blockMeshDict"
    bmd_path.write_text(content, encoding="utf-8")
    return bmd_path


def _main_with_argv(*args: str) -> None:
    """Call main() with the given CLI args injected via sys.argv."""
    with patch.object(sys, "argv", ["extract-patches", *args]):
        main()


# ---------------------------------------------------------------------------
# Smoke test: --help exits cleanly (no OCC needed)
# ---------------------------------------------------------------------------

class TestHelpFlag:

    def test_help_exits_zero(self):
        """--help must exit with code 0."""
        with (
            patch.object(sys, "argv", ["extract-patches", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Error handling tests (no OCC needed)
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_blockMeshDict_exits_nonzero(self, tmp_path: Path):
        """When blockMeshDict is missing, the tool must exit with non-zero code."""
        # Create a dummy stp file so the tool doesn't fail before bmd check
        dummy_stp = tmp_path / "model.stp"
        dummy_stp.write_text("dummy", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--stpPath", str(dummy_stp),
            )
        assert exc_info.value.code != 0

    def test_missing_stp_file_exits_nonzero(self, tmp_path: Path):
        """When the given --stp-path does not exist, the tool must exit non-zero."""
        _write_bmd(tmp_path)
        nonexistent_stp = tmp_path / "nonexistent.stp"

        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--stpPath", str(nonexistent_stp),
            )
        assert exc_info.value.code != 0

    def test_no_stp_file_in_geometry_dir_exits_nonzero(self, tmp_path: Path):
        """When no STEP file is found in the default geometry dir, must exit non-zero."""
        _write_bmd(tmp_path)
        # Create geometry dir but leave it empty
        geometry_dir = tmp_path / "constant" / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
            )
        assert exc_info.value.code != 0

    def test_no_backup_suppresses_backup(self, tmp_path: Path):
        """--noBackup must not raise on import (OCC check is deferred to run())."""
        _write_bmd(tmp_path)
        # We expect a SystemExit (no STP file) but the .bak must not exist
        bmd_path = tmp_path / "system" / "blockMeshDict"
        geometry_dir = tmp_path / "constant" / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(SystemExit):
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--noBackup",
            )

        bak_path = bmd_path.with_suffix(".bak")
        assert not bak_path.exists(), ".bak backup should not exist with --noBackup"


# ---------------------------------------------------------------------------
# load_solids_with_names integration: verify names reach patch creation
# ---------------------------------------------------------------------------

class TestLoadSolidsWithNamesIntegration:
    """Verify that load_solids_with_names is called and names are forwarded
    to downstream patch creation (no OCC required â€” fully mocked)."""

    def test_named_solid_label_used_as_patch_name(self, tmp_path: Path):
        """When load_solids_with_names returns a NamedSolid with a specific
        name, that name must appear as the patch name in blockMeshDict.

        The test mocks both load_solids_with_names and the OCC face-matching
        helpers so that exactly one block face is matched to the solid and the
        patch is written with the expected name.
        """
        bmd_path = _write_bmd(tmp_path)

        # Create a dummy STP file so the path-existence check inside run() passes
        dummy_stp = tmp_path / "dummy.stp"
        dummy_stp.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
            encoding="utf-8",
        )

        # Build a minimal NamedSolid stub â€” solid object can be any sentinel
        sentinel_solid = object()
        named_solid = NamedSolid(
            solid=sentinel_solid,
            name="myFluidPatch",
            source="step_id",
        )

        # Coordinates matching the single block in _MINIMAL_BMD (unit cube)
        vertex_coords = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        mock_block_face = MagicMock()
        mock_block_face.block_name = "testBlock"
        mock_block_face.face_name = "bottom"
        mock_block_face.vertex_coords = vertex_coords
        mock_block_face.vertex_names = ["v0", "v1", "v2", "v3"]
        mock_block_face.support_points = []

        dummy_occ_face = MagicMock()

        import meshing_utils.cli.extract_patches as ep_mod

        with (
            patch.object(ep_mod, "load_solids_with_names", return_value=[named_solid]),
            patch.object(op_mod, "extract_block_faces", return_value=[mock_block_face]),
            patch.object(op_mod, "nearest_face_within_tol", return_value=dummy_occ_face),
            patch.object(op_mod, "find_dominant_face", return_value=(dummy_occ_face, [])),
            patch.object(op_mod, "surface_type_of", return_value="plane"),
            patch.object(op_mod, "effective_normal_tolerance", return_value=5.0),
            patch.object(op_mod, "compute_outward_normal", return_value=(0.0, 0.0, -1.0)),
            patch.object(op_mod, "normals_consistent", return_value=True),
            patch.object(op_mod, "local_surface_normal", return_value=(0.0, 0.0, -1.0)),
        ):
            run(
                bmd_path=bmd_path,
                stp_path=dummy_stp,
                tol=1e-4,
                normal_angle_tol=5.0,
                curved_normal_angle_tol=30.0,
                default_patch_type="patch",
                strict=False,
            )

        content = bmd_path.read_text(encoding="utf-8")
        assert "myFluidPatch" in content, (
            "Expected patch name 'myFluidPatch' to appear in written blockMeshDict"
        )
