"""Unit tests for meshing_utils.cli.clean_bmd."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from meshing_utils.cli.clean_bmd import main

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
    with patch.object(sys, "argv", ["clean-bmd", *args]):
        main()


# ---------------------------------------------------------------------------
# Smoke test: --help exits cleanly
# ---------------------------------------------------------------------------

class TestHelpFlag:

    def test_help_exits_zero(self):
        """--help must exit with code 0."""
        with (
            patch.object(sys, "argv", ["clean-bmd", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_default_output_creates_clean_blockMeshDict(self, tmp_path: Path):
        """Without --output, writes clean_blockMeshDict in the same dir."""
        bmd_path = _write_bmd(tmp_path)
        expected_output = bmd_path.parent / "clean_blockMeshDict"

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--noBackup",
        )

        assert expected_output.exists(), "clean_blockMeshDict was not created"
        content = expected_output.read_text(encoding="utf-8")
        assert "blockMeshDict" in content

    def test_explicit_output_writes_to_specified_path(self, tmp_path: Path):
        """--output FILE writes the result to the specified path."""
        _write_bmd(tmp_path)
        out_file = tmp_path / "normalised_bmd"

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--output", str(out_file),
            "--noBackup",
        )

        assert out_file.exists(), "Output file was not created"
        content = out_file.read_text(encoding="utf-8")
        assert "blockMeshDict" in content

    def test_in_place_output_creates_backup(self, tmp_path: Path):
        """Writing in-place (--output pointing to bmd) creates a .bak."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--output", str(bmd_path),
        )

        bak_path = bmd_path.with_suffix(".bak")
        assert bak_path.exists(), ".bak backup was not created"

    def test_no_backup_suppresses_backup(self, tmp_path: Path):
        """--noBackup must suppress the .bak file creation."""
        bmd_path = _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--noBackup",
        )

        bak_path = bmd_path.with_suffix(".bak")
        assert not bak_path.exists(), ".bak backup should not exist with --noBackup"

    def test_output_is_valid_blockMeshDict(self, tmp_path: Path):
        """The output file must be parseable as a BlockMeshDict."""
        from meshing_utils import BlockMeshDict

        _write_bmd(tmp_path)

        _main_with_argv(
            "--caseDir", str(tmp_path),
            "--noBackup",
        )

        output_path = tmp_path / "system" / "clean_blockMeshDict"
        # Should not raise
        bmd = BlockMeshDict(output_path)
        assert len(bmd.vertices) == 8
        assert len(bmd.blocks) == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_blockMeshDict_exits_nonzero(self, tmp_path: Path):
        """When blockMeshDict is missing, the tool must exit with non-zero code."""
        with pytest.raises(SystemExit) as exc_info:
            _main_with_argv(
                "--caseDir", str(tmp_path),
            )
        assert exc_info.value.code != 0
