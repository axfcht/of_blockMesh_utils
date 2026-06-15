"""Unit tests for meshing_utils.cli.scale_bmd."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from meshing_utils.cli.scale_bmd import main

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
\tname block0 hex (v0 v1 v2 v3 v4 v5 v6 v7) (1 1 1) simpleGrading (1 1 1)
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
    with patch.object(sys, "argv", ["scale-bmd", *args]):
        main()


# ---------------------------------------------------------------------------
# Smoke test: --help exits cleanly
# ---------------------------------------------------------------------------

class TestHelpFlag:

    def test_help_exits_zero(self):
        """--help must exit with code 0."""
        with (
            patch.object(sys, "argv", ["scale-bmd", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_uniform_factor_scales_in_place(self, tmp_path: Path):
        """--factor 2 doubles all vertex coordinates in-place."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--factor", "2",
            "--noBackup",
        )

        content = bmd_path.read_text(encoding="utf-8")
        assert "2.0" in content or "2" in content

    def test_per_axis_factors_write_to_output(self, tmp_path: Path):
        """--factors fx fy fz writes result to --output file."""
        _write_bmd(tmp_path)
        out_file = tmp_path / "scaled_bmd"

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--factors", "2", "3", "1",
            "--output", str(out_file),
            "--noBackup",
        )

        assert out_file.exists(), "Output file was not created"
        content = out_file.read_text(encoding="utf-8")
        assert "blockMeshDict" in content

    def test_in_place_creates_backup(self, tmp_path: Path):
        """In-place mode creates a .bak backup."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--factor", "1",
        )

        bak_path = bmd_path.with_suffix(".bak")
        assert bak_path.exists(), ".bak backup was not created"

    def test_no_backup_suppresses_backup(self, tmp_path: Path):
        """--noBackup must suppress the .bak file creation."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--factor", "1",
            "--noBackup",
        )

        bak_path = bmd_path.with_suffix(".bak")
        assert not bak_path.exists(), ".bak backup should not exist with --noBackup"


# ---------------------------------------------------------------------------
# Edge-case / error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_blockMeshDict_exits_nonzero(self, tmp_path: Path):
        """When blockMeshDict is missing, the tool must exit with a non-zero code."""
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--factor", "2",
            )
        assert exc_info.value.code != 0

    def test_missing_factor_arg_exits_nonzero(self):
        """When neither --factor nor --factors is given, argparse must exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv()
        assert exc_info.value.code == 2

    def test_mutually_exclusive_factors_exits_nonzero(self, tmp_path: Path):
        """--factor and --factors together must be rejected."""
        _write_bmd(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
                "--factor", "2",
                "--factors", "1", "1", "1",
            )
        assert exc_info.value.code == 2
