# of_blockMesh_utils

[![CI](https://github.com/axfcht/of_blockMesh_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/axfcht/of_blockMesh_utils/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Python command-line utilities for creating, transforming, and converting
OpenFOAM `blockMeshDict` files. With meshing-utils you can clean and normalise
existing dictionaries, scale and revolve meshes, extrude marked block faces,
and generate dictionaries directly from STEP CAD models.

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

The CAD tools (`stpToBMD`, `extractPatches`, `extractCZones`) rely on
[`cadquery-ocp`](https://pypi.org/project/cadquery-ocp/), which `pip` installs
automatically wherever a pre-built wheel is available for your interpreter and
platform (Python 3.13 is known to work). On other interpreters you may need to
install OCP separately. The pure-Python tools (`cleanBMD`, `scaleBMD`,
`revolveBMD`, `extrudeBMD`, `splitBMDZones`, `combineBMD`) work without OCP.

## Command-line tools

Every tool shares a common set of options:

- `--caseDir DIR` — OpenFOAM case root (default: current directory). The
  dictionary is read from `<caseDir>/system/blockMeshDict`.
- `--logLevel {DEBUG,INFO,WARNING,ERROR}` — verbosity (default: `INFO`).
- `--noBackup` — skip writing a `.bak` copy before overwriting.

| Command          | Purpose                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `cleanBMD`       | Parse and re-serialise a `blockMeshDict` to normalise formatting.        |
| `scaleBMD`       | Scale vertices by uniform or per-axis factors.                           |
| `revolveBMD`     | Revolve the mesh around an arbitrary axis to produce an annular sector.  |
| `extrudeBMD`     | Extrude marked block faces by one or more offset vectors.                |
| `stpToBMD`       | Generate a `blockMeshDict` from a STEP file containing hex solids.       |
| `extractPatches` | Match STEP surfaces to existing block faces and append boundary patches. |
| `extractCZones`  | Assign `cellZone` names to blocks via STEP solid containment.            |
| `splitBMDZones`  | Split a `blockMeshDict` into one file per `cellZone`.                    |
| `combineBMD`     | Combine multiple per-zone `blockMeshDict` fragments into one file.       |

Run any tool with `--help` to see its specific arguments.

## End-to-end workflow

A typical workflow starts with a STEP file describing the CAD geometry of the
flow domain. The default locations mirror the OpenFOAM case structure:

- STEP file: `<caseDir>/constant/geometry/<model>.stp`
- blockMeshDict: `<caseDir>/system/blockMeshDict`

```bash
# 1. Generate blockMeshDict from STEP hex solids (run in the case directory)
stpToBMD

# 2. (Optional) Match STEP surfaces to block faces and append boundary patches
extractPatches

# 3. (Optional) Assign cellZone names from STEP solid containment
extractCZones

# 4. (Optional) Split into per-zone files or combine multiple fragments
splitBMDZones
combineBMD

# 5. Run blockMesh inside WSL (OpenFOAM must be sourced)
wsl blockMesh -case /path/to/case
```

All tools accept `--caseDir <path>` to point at a different case root.

## Python API

The package also exposes a small, stable public API for use as a library:

```python
from meshing_utils import BlockMeshDict, scale

bmd = BlockMeshDict("system/blockMeshDict")
scaled = scale(bmd, 0.001, 0.001, 0.001)   # mm -> m
scaled.write("system/blockMeshDict")
```

Core data models (`BlockMeshDict`, `Vertex`, `Edge`, `Block`, `Face`, `Patch`,
`Boundary`) and high-level operations (`scale`, `revolve`, `extrude`,
`assign_cell_zones`, `combine_blockmeshdicts`, `split_blockmeshdict_by_zones`)
are importable directly from the top-level `meshing_utils` package.

## Environment

meshing-utils is developed on Windows, with OpenFOAM commands (`blockMesh`, ...)
run inside WSL 2 (Ubuntu). All file paths use `pathlib`, so the tools work from
either context.

## Testing

```bash
pip install -e .[dev]
pytest -m "not integration"
```

The suite includes byte-exact snapshot tests for `blockMeshDict` serialisation
as a regression safety net. STEP/CAD tests are skipped automatically when OCP
is not installed.

## License

Released under the MIT License.
