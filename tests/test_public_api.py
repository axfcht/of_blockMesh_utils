"""Snapshot of the public API surface of ``meshing_utils``.

This test freezes the current ``__all__`` so that subsequent changes
cannot accidentally remove or add public symbols without an explicit,
traceable change to this file.
"""

from __future__ import annotations

import meshing_utils

EXPECTED_PUBLIC_API: frozenset[str] = frozenset(
    {
        # data models
        "BlockMeshDict",
        "Block",
        "Boundary",
        "Edge",
        "Face",
        "Patch",
        "Vertex",
        # high-level operations
        "assign_cell_zones",
        "combine_blockmeshdicts",
        "extrude",
        "extrude_with_steps",
        "revolve",
        "RevolveConfig",
        "scale",
        "split_blockmeshdict_by_zones",
        # public exception base classes
        "ExtrusionError",
        "HexValidationError",
        "MeshingUtilsError",
        "TopologyError",
    }
)


def test_public_api_matches_snapshot() -> None:
    """``meshing_utils.__all__`` must equal the frozen snapshot.

    Update both this snapshot and the docs in the same commit when
    intentionally changing the public API.
    """
    actual = frozenset(meshing_utils.__all__)
    missing = EXPECTED_PUBLIC_API - actual
    added = actual - EXPECTED_PUBLIC_API
    assert not missing, f"Public symbols removed without snapshot update: {sorted(missing)}"
    assert not added, f"Public symbols added without snapshot update: {sorted(added)}"


def test_public_api_symbols_are_importable() -> None:
    """Every name in ``__all__`` must be a real attribute on the package."""
    for name in meshing_utils.__all__:
        assert hasattr(meshing_utils, name), f"{name} is in __all__ but not importable"
