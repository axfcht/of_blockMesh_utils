"""CLI smoke tests for meshing_utils.cli.split_bmd_by_zones."""

from __future__ import annotations

from pathlib import Path

import pytest

from meshing_utils import BlockMeshDict
from meshing_utils.cli.split_bmd_by_zones import main

# ---------------------------------------------------------------------------
# Minimal blockMeshDict content (two zones + one un-zoned block)
# ---------------------------------------------------------------------------

_BMD_FLUID_SOLID_NONE = """\
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
\tname vs0 (1.00000000 0.00000000 0.00000000)
\tname vs1 (2.00000000 0.00000000 0.00000000)
\tname vs2 (2.00000000 1.00000000 0.00000000)
\tname vs3 (1.00000000 1.00000000 0.00000000)
\tname vs4 (1.00000000 0.00000000 1.00000000)
\tname vs5 (2.00000000 0.00000000 1.00000000)
\tname vs6 (2.00000000 1.00000000 1.00000000)
\tname vs7 (1.00000000 1.00000000 1.00000000)
\tname vn0 (2.00000000 0.00000000 0.00000000)
\tname vn1 (3.00000000 0.00000000 0.00000000)
\tname vn2 (3.00000000 1.00000000 0.00000000)
\tname vn3 (2.00000000 1.00000000 0.00000000)
\tname vn4 (2.00000000 0.00000000 1.00000000)
\tname vn5 (3.00000000 0.00000000 1.00000000)
\tname vn6 (3.00000000 1.00000000 1.00000000)
\tname vn7 (2.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\thex (vf0 vf1 vf2 vf3 vf4 vf5 vf6 vf7) fluid (1 1 1) simpleGrading (1 1 1)
\thex (vs0 vs1 vs2 vs3 vs4 vs5 vs6 vs7) solid (1 1 1) simpleGrading (1 1 1)
\thex (vn0 vn1 vn2 vn3 vn4 vn5 vn6 vn7) (1 1 1) simpleGrading (1 1 1)
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
def case_dir(tmp_path: Path) -> Path:
    """Write a minimal blockMeshDict into a temporary case structure."""
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    bmd_path = system_dir / "blockMeshDict"
    bmd_path.write_text(_BMD_FLUID_SOLID_NONE, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Smoke test 1: --help exits with code 0
# ---------------------------------------------------------------------------


def test_cli_help_smoke():
    """--help exits cleanly with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Smoke test 2: Default mode round-trip
# ---------------------------------------------------------------------------


def test_cli_default_writes_per_zone_files(case_dir: Path):
    """End-to-end: default mode writes one file per zone."""
    exit_code = main([
        "--caseDir", str(case_dir),
        "--noBackup",
    ])
    assert exit_code == 0

    system_dir = case_dir / "system"
    written_names = {p.name for p in system_dir.iterdir()}

    assert "blockMeshDict_fluid" in written_names
    assert "blockMeshDict_solid" in written_names
    assert "blockMeshDict_no_zone" in written_names

    # Verify that each output contains exactly its own blocks
    fluid_bmd = BlockMeshDict(system_dir / "blockMeshDict_fluid")
    assert all(b.zone == "fluid" for b in fluid_bmd.blocks)

    solid_bmd = BlockMeshDict(system_dir / "blockMeshDict_solid")
    assert all(b.zone == "solid" for b in solid_bmd.blocks)

    no_zone_bmd = BlockMeshDict(system_dir / "blockMeshDict_no_zone")
    assert all((b.zone is None or b.zone == "") for b in no_zone_bmd.blocks)


# ---------------------------------------------------------------------------
# Smoke test 3: argparse mutually exclusive raises SystemExit(2)
# ---------------------------------------------------------------------------


def test_cli_include_and_exclude_conflict_exits_nonzero(case_dir: Path):
    """argparse raises SystemExit(2) when --include and --exclude are both given."""
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--caseDir", str(case_dir),
            "--include", "fluid",
            "--exclude", "solid",
            "--noBackup",
        ])
    assert exc_info.value.code != 0
