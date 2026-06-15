"""CLI smoke tests for meshing_utils.cli.combine_bmd."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshing_utils import BlockMeshDict
from meshing_utils.cli.combine_bmd import main

# ---------------------------------------------------------------------------
# Minimal blockMeshDict fragment content
# ---------------------------------------------------------------------------

_BMD_FLUID = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\/     M anipulation  |
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
\tname vf0 (0.00000000 0.00000000 0.00000000)
\tname vf1 (1.00000000 0.00000000 0.00000000)
\tname vf2 (1.00000000 1.00000000 0.00000000)
\tname vf3 (0.00000000 1.00000000 0.00000000)
\tname vf4 (0.00000000 0.00000000 1.00000000)
\tname vf5 (1.00000000 0.00000000 1.00000000)
\tname vf6 (1.00000000 1.00000000 1.00000000)
\tname vf7 (0.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\thex (vf0 vf1 vf2 vf3 vf4 vf5 vf6 vf7) fluid (1 1 1) simpleGrading (1 1 1)
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

_BMD_SOLID = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\/     M anipulation  |
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
\tname vs0 (1.00000000 0.00000000 0.00000000)
\tname vs1 (2.00000000 0.00000000 0.00000000)
\tname vs2 (2.00000000 1.00000000 0.00000000)
\tname vs3 (1.00000000 1.00000000 0.00000000)
\tname vs4 (1.00000000 0.00000000 1.00000000)
\tname vs5 (2.00000000 0.00000000 1.00000000)
\tname vs6 (2.00000000 1.00000000 1.00000000)
\tname vs7 (1.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\thex (vs0 vs1 vs2 vs3 vs4 vs5 vs6 vs7) solid (1 1 1) simpleGrading (1 1 1)
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


# ---------------------------------------------------------------------------
# Fixture: case directory with two fragment files
# ---------------------------------------------------------------------------


@pytest.fixture()
def case_dir(tmp_path: Path) -> Path:
    """Write two minimal blockMeshDict fragment files into a tmp case structure."""
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    (system_dir / "blockMeshDict_fluid").write_text(_BMD_FLUID, encoding="utf-8")
    (system_dir / "blockMeshDict_solid").write_text(_BMD_SOLID, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Smoke test 1: combineBMD writes combined blockMeshDict
# ---------------------------------------------------------------------------


def test_cli_combines_fragments_into_output(case_dir: Path):
    """End-to-end: combineBMD writes a combined blockMeshDict with both blocks."""
    exit_code = main([
        "--caseDir", str(case_dir),
        "--noBackup",
    ])
    assert exit_code == 0

    output_path = case_dir / "system" / "blockMeshDict"
    assert output_path.is_file(), "Output blockMeshDict was not written."

    combined = BlockMeshDict(output_path)
    block_names = {b.name for b in combined.blocks}
    assert "b_fluid" not in block_names  # unnamed blocks have no name attribute set from zone
    # Verify we have 2 blocks total
    assert len(combined.blocks) == 2

    # Verify zones are preserved
    zones = {b.zone for b in combined.blocks}
    assert "fluid" in zones
    assert "solid" in zones

    # Verify vertex count (16 unique vertices: 8 per BMD, no shared)
    assert len(combined.vertices) == 16


# ---------------------------------------------------------------------------
# Smoke test 2: .bak backup is created when output already exists
# ---------------------------------------------------------------------------


def test_cli_backup_created_when_output_exists(case_dir: Path):
    """A .bak file is created when the output blockMeshDict already exists."""
    system_dir = case_dir / "system"
    output_path = system_dir / "blockMeshDict"
    # Pre-create output file
    output_path.write_text("existing content", encoding="utf-8")

    exit_code = main([
        "--caseDir", str(case_dir),
        # No --noBackup: backup should be created
    ])
    assert exit_code == 0

    bak_path = system_dir / "blockMeshDict.bak"
    assert bak_path.is_file(), ".bak file was not created."
    assert bak_path.read_text(encoding="utf-8") == "existing content"


# ---------------------------------------------------------------------------
# Smoke test 3: --noBackup suppresses backup
# ---------------------------------------------------------------------------


def test_cli_no_backup_suppresses_backup(case_dir: Path):
    """--noBackup prevents creation of a .bak file."""
    system_dir = case_dir / "system"
    output_path = system_dir / "blockMeshDict"
    # Pre-create output file
    output_path.write_text("existing content", encoding="utf-8")

    exit_code = main([
        "--caseDir", str(case_dir),
        "--noBackup",
    ])
    assert exit_code == 0

    bak_path = system_dir / "blockMeshDict.bak"
    assert not bak_path.exists(), ".bak file should NOT be created with --noBackup."
