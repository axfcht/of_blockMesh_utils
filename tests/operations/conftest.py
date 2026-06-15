"""Shared fixtures for operations tests."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_mock_occ: do not auto-mock OCC helpers (for fail-fast tests).",
    )


@pytest.fixture(autouse=True)
def mock_occ_helpers(monkeypatch, request):
    """Mock OCC helpers in cell_zones so tests run without OCC installed.

    compute_solid_aabb returns a permissive AABB that overlaps any reasonable
    block AABB. make_solid_classifier returns None, which triggers the legacy
    per-point classify_point_in_solid path inside _classify_samples_against_solid.

    Tests can opt out via @pytest.mark.no_mock_occ.
    """
    if "no_mock_occ" in request.keywords:
        return

    def _fake_compute_solid_aabb(solid, tol):
        return (-1e10, -1e10, -1e10, 1e10, 1e10, 1e10)

    def _fake_make_solid_classifier(solid):
        return None

    # Phase 3.4: the helpers now live in the cell_zones.core submodule.
    monkeypatch.setattr(
        "meshing_utils.operations.cell_zones.core.compute_solid_aabb",
        _fake_compute_solid_aabb,
    )
    monkeypatch.setattr(
        "meshing_utils.operations.cell_zones.core.make_solid_classifier",
        _fake_make_solid_classifier,
    )
