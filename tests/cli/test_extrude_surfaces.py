"""Unit tests for meshing_utils.cli.extrude_surfaces."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from meshing_utils.cli.extrude_surfaces import main

# ---------------------------------------------------------------------------
# Minimal blockMeshDict fixture with marked vertices and block
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
\tname v0 (0 0 0)
\tname v1 (1 0 0)
\tname v2 (1 1 0)
\tname v3 (0 1 0)
\tname v4 (0 0 1) //* top
\tname v5 (1 0 1) //* top
\tname v6 (1 1 1) //* top
\tname v7 (0 1 1) //* top
);

edges
(
);

blocks
(
\tname b0 hex (v0 v1 v2 v3 v4 v5 v6 v7) (2 3 4) simpleGrading (1 1 1) //* extrude
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
    with patch.object(sys, "argv", ["extrude-surfaces", *args]):
        main()


# ---------------------------------------------------------------------------
# Smoke test: --help exits cleanly
# ---------------------------------------------------------------------------

class TestHelpFlag:

    def test_help_exits_zero(self):
        """--help must exit with code 0."""
        with (
            patch.object(sys, "argv", ["extrude-surfaces", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_single_layer_writes_default_output(self, tmp_path: Path):
        """Single layer extrusion must write extruded_blockMeshDict."""
        bmd_path = _write_bmd(tmp_path)
        expected_output = bmd_path.parent / "extruded_blockMeshDict"

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--layer", "block", "0", "0", "0.2",
            "--noBackup",
        )

        assert expected_output.exists(), "extruded_blockMeshDict was not created"
        content = expected_output.read_text(encoding="utf-8")
        assert "blockMeshDict" in content

    def test_two_layers_write_to_explicit_output(self, tmp_path: Path):
        """Two-layer extrusion with --output writes to the specified file."""
        _write_bmd(tmp_path)
        out_file = tmp_path / "my_output"

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--layer", "block", "0", "0", "0.2",
            "--layer", "block", "0", "0", "0.4",
            "--output", str(out_file),
            "--noBackup",
        )

        assert out_file.exists(), "Output file was not created"
        content = out_file.read_text(encoding="utf-8")
        assert "blockMeshDict" in content

    def test_no_backup_suppresses_backup(self, tmp_path: Path):
        """--noBackup must suppress the .bak file creation."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--layer", "block", "0", "0", "0.2",
            "--noBackup",
        )

        bak_path = bmd_path.with_suffix(".bak")
        assert not bak_path.exists(), ".bak backup should not exist with --noBackup"


# ---------------------------------------------------------------------------
# Edge-case / error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_blockMeshDict_exits_nonzero(self, tmp_path: Path):
        """When blockMeshDict is missing, the tool must exit with non-zero code."""
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--layer", "block", "0", "0", "0.2",
            )
        assert exc_info.value.code != 0

    def test_invalid_layer_kind_exits_nonzero(self, tmp_path: Path):
        """Unknown layer kind must exit with a non-zero code."""
        _write_bmd(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--layer", "bogus", "0", "0", "0.2",
            )
        assert exc_info.value.code != 0

    def test_missing_layer_arg_exits_nonzero(self):
        """When --layer is absent, argparse must exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv()
        assert exc_info.value.code == 2
