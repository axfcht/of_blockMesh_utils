# Contributing

Contributions are welcome. This document describes the local setup and the
checks that run in CI.

## Development setup

Requires Python >= 3.10.

```bash
pip install -e .[dev]
```

This installs the package in editable mode together with the development tools
(`ruff`, `mypy`, `pytest`, `pytest-timeout`).

## Checks

Before opening a pull request, please run the same checks as CI:

```bash
ruff check src tests          # lint
ruff format src tests         # format
mypy src/meshing_utils        # type check
pytest -m "not integration"   # tests
```

CAD/STEP tests that require [`cadquery-ocp`](https://pypi.org/project/cadquery-ocp/)
are skipped automatically when OCP is not installed, and the full suite is
selected with `-m integration`.

## Conventions

- **Code, comments, and commit messages:** English.
- **Style:** enforced by `ruff` (configuration in `pyproject.toml`).
- **Tests:** new features and bug fixes should come with `pytest` tests covering
  the happy path and the critical edge cases.
- **Architecture:** follow the layering `cli -> operations -> {foam, geometry,
  cad, io}`; subpackages must not import from `cli/`. Internal code imports from
  the defining submodule, not the top-level facade.
