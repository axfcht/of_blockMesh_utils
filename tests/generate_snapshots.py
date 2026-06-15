"""One-time script to generate snapshot fixtures for round-trip tests.

Run once from the project root:
    python tests/generate_snapshots.py

The script loads each input file, calls BlockMeshDict.write(), and saves the
result as a .foam snapshot file in tests/fixtures/snapshots/.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshing_utils import BlockMeshDict

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SNAPSHOTS_DIR = FIXTURES_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

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


def _generate(snapshot_name: str, input_text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".foam", encoding="utf-8"
    ) as f:
        f.write(input_text)
        tmp_in = Path(f.name)
    try:
        bmd = BlockMeshDict(tmp_in)
        out_path = SNAPSHOTS_DIR / snapshot_name
        bmd.write(out_path)
        print(f"Generated: {out_path}")
    finally:
        tmp_in.unlink()


def _generate_from_file(snapshot_name: str, input_path: Path) -> None:
    bmd = BlockMeshDict(input_path)
    out_path = SNAPSHOTS_DIR / snapshot_name
    bmd.write(out_path)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    _generate("blockMeshDict_minimal.foam", _MINIMAL_BMD)
    _generate("blockMeshDict_extrude.foam", _EXTRUDE_BMD)
    _generate_from_file(
        "blockMeshDict_example.foam",
        FIXTURES_DIR / "example_blockMeshDict",
    )
    _generate_from_file(
        "blockMeshDict_komplett.foam",
        FIXTURES_DIR / "blockMeshDict_komplett",
    )
    print("All snapshots generated.")
