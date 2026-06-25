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


_BMD_COLLISION_A = """\
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
\tname v0 (0.00000000 0.00000000 0.00000000)
\tname v1 (1.00000000 0.00000000 0.00000000)
\tname v2 (1.00000000 1.00000000 0.00000000)
\tname v3 (0.00000000 1.00000000 0.00000000)
\tname v4 (0.00000000 0.00000000 1.00000000)
\tname v5 (1.00000000 0.00000000 1.00000000)
\tname v6 (1.00000000 1.00000000 1.00000000)
\tname v7 (0.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\thex (v0 v1 v2 v3 v4 v5 v6 v7) zoneA (1 1 1) simpleGrading (1 1 1)
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

_BMD_COLLISION_B = """\
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
\tname v0 (10.00000000 0.00000000 0.00000000)
\tname vb1 (11.00000000 0.00000000 0.00000000)
\tname vb2 (11.00000000 1.00000000 0.00000000)
\tname vb3 (10.00000000 1.00000000 0.00000000)
\tname vb4 (10.00000000 0.00000000 1.00000000)
\tname vb5 (11.00000000 0.00000000 1.00000000)
\tname vb6 (11.00000000 1.00000000 1.00000000)
\tname vb7 (10.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\thex (v0 vb1 vb2 vb3 vb4 vb5 vb6 vb7) zoneB (1 1 1) simpleGrading (1 1 1)
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


@pytest.fixture()
def collision_case_dir(tmp_path: Path) -> Path:
    """Write two fragment files that share vertex name v0 with DIFFERENT coordinates."""
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    (system_dir / "blockMeshDict_collision_a").write_text(_BMD_COLLISION_A, encoding="utf-8")
    (system_dir / "blockMeshDict_collision_b").write_text(_BMD_COLLISION_B, encoding="utf-8")
    return tmp_path


def test_cli_vertex_name_collision_different_coords_renamed(collision_case_dir: Path):
    """CLI: same vertex name with different coords does not raise; v0_2 is present in output."""
    exit_code = main([
        "--caseDir", str(collision_case_dir),
        "--noBackup",
    ])
    assert exit_code == 0, "combineBMD should succeed even with vertex name collision"

    output_path = collision_case_dir / "system" / "blockMeshDict"
    assert output_path.is_file(), "Output blockMeshDict was not written."

    combined = BlockMeshDict(output_path)

    vertex_names = [v.name for v in combined.vertices]
    assert "v0" in vertex_names, "Original v0 must be present"
    assert "v0_2" in vertex_names, "Renamed v0_2 must be present in output"

    # Both blocks must be present (2 blocks total)
    assert len(combined.blocks) == 2

    # The block from collision_b must reference v0_2
    zones = {b.zone for b in combined.blocks}
    assert "zoneA" in zones
    assert "zoneB" in zones

    # Find the zoneB block and confirm it references v0_2
    zone_b_block = next(b for b in combined.blocks if b.zone == "zoneB")
    assert "v0_2" in zone_b_block.vertices, "zoneB block must reference renamed vertex v0_2"
    assert "v0" not in zone_b_block.vertices, "zoneB block must not reference original v0"


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
