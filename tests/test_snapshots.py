"""Round-trip snapshot tests for BlockMeshDict.write().

Each test loads an input blockMeshDict, calls write() into a tmp_path, and
compares the result byte-by-byte against the pre-generated snapshot stored in
tests/fixtures/snapshots/.  A failing test means a regression in the write()
output format.

To regenerate snapshots after an intentional format change, run:
    python tests/generate_snapshots.py
"""

from pathlib import Path

import pytest

from meshing_utils import BlockMeshDict

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SNAPSHOTS_DIR = FIXTURES_DIR / "snapshots"

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
\tname v0 (1.0 0.0 0.0)
\tname v1 (2.0 0.0 0.0)
\tname v2 (2.0 1.0 0.0)
\tname v3 (1.0 1.0 0.0)
\tname v4 (1.0 0.0 1.0)
\tname v5 (2.0 0.0 1.0)
\tname v6 (2.0 1.0 1.0)
\tname v7 (1.0 1.0 1.0)
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
\touter
\t{
\t\ttype patch;
\t\tfaces
\t\t(
\t\t\t(v4 v5 v6 v7)
\t\t);
\t}
);

// ************************************************************************* //
"""

_EXTRUDE_BMD = """\
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
\ttopWall
\t{
\t\ttype wall;
\t\tfaces
\t\t(
\t\t\t(v4 v5 v6 v7)
\t\t);
\t}
);

// ************************************************************************* //
"""


def _load_from_string(content: str, tmp_path: Path) -> BlockMeshDict:
    src = tmp_path / "input_blockMeshDict"
    src.write_text(content, encoding="utf-8")
    return BlockMeshDict(src)


@pytest.mark.parametrize(
    "snapshot_name,input_source",
    [
        ("blockMeshDict_minimal.foam", "minimal"),
        ("blockMeshDict_extrude.foam", "extrude"),
        ("blockMeshDict_example.foam", "example"),
        ("blockMeshDict_komplett.foam", "komplett"),
    ],
)
def test_round_trip_unchanged(snapshot_name: str, input_source: str, tmp_path: Path) -> None:
    if input_source == "minimal":
        bmd = _load_from_string(_MINIMAL_BMD, tmp_path)
    elif input_source == "extrude":
        bmd = _load_from_string(_EXTRUDE_BMD, tmp_path)
    elif input_source == "example":
        bmd = BlockMeshDict(FIXTURES_DIR / "example_blockMeshDict")
    elif input_source == "komplett":
        bmd = BlockMeshDict(FIXTURES_DIR / "blockMeshDict_komplett")
    else:
        pytest.fail(f"Unknown input_source: {input_source}")

    out = tmp_path / "blockMeshDict"
    bmd.write(out)

    actual = out.read_text(encoding="utf-8")
    expected = (SNAPSHOTS_DIR / snapshot_name).read_text(encoding="utf-8")
    assert actual == expected, f"Round-trip output changed for {snapshot_name}"
