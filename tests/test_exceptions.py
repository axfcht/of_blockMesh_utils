"""Regression tests for the meshing_utils exception hierarchy.

Verifies MRO guarantees introduced in the 16-item fix batch:
- MeshingUtilsError is the common base for all library-specific exceptions.
- TopologyError and ParseError keep ValueError in their MRO so that existing
  ``except ValueError`` catch-sites in cell_count / extrude_surfaces continue
  to work.
- HexValidationError, OrderingConsistencyError, ExtrusionError are
  MeshingUtilsError subclasses without being ValueError subclasses.

These tests are pure type-hierarchy checks; no instances are raised.
"""

from __future__ import annotations

import pytest

from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.geometry.hex_axes.detection import TopologyError
from meshing_utils.geometry.hex_topology.core import HexValidationError, OrderingConsistencyError
from meshing_utils.operations.extrusion import ExtrusionError
from meshing_utils.operations.extrusion.parsing import ParseError

# ---------------------------------------------------------------------------
# MeshingUtilsError is the root
# ---------------------------------------------------------------------------

class TestMeshingUtilsErrorRoot:

    def test_is_exception_subclass(self):
        """MeshingUtilsError must derive from Exception."""
        assert issubclass(MeshingUtilsError, Exception)

    def test_not_value_error_itself(self):
        """MeshingUtilsError must NOT be a ValueError (it is the library root)."""
        assert not issubclass(MeshingUtilsError, ValueError)


# ---------------------------------------------------------------------------
# TopologyError MRO
# ---------------------------------------------------------------------------

class TestTopologyErrorMRO:

    def test_is_meshing_utils_error(self):
        """TopologyError must inherit from MeshingUtilsError."""
        assert issubclass(TopologyError, MeshingUtilsError)

    def test_is_value_error(self):
        """TopologyError must inherit from ValueError so existing except-clauses catch it."""
        assert issubclass(TopologyError, ValueError)

    def test_is_exception(self):
        assert issubclass(TopologyError, Exception)

    def test_instance_caught_as_value_error(self):
        """An instance of TopologyError must be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise TopologyError("test")

    def test_instance_caught_as_meshing_utils_error(self):
        """An instance of TopologyError must be catchable as MeshingUtilsError."""
        with pytest.raises(MeshingUtilsError):
            raise TopologyError("test")


# ---------------------------------------------------------------------------
# ParseError MRO
# ---------------------------------------------------------------------------

class TestParseErrorMRO:

    def test_is_meshing_utils_error(self):
        """ParseError must inherit from MeshingUtilsError."""
        assert issubclass(ParseError, MeshingUtilsError)

    def test_is_value_error(self):
        """ParseError must inherit from ValueError so existing except-clauses catch it."""
        assert issubclass(ParseError, ValueError)

    def test_instance_caught_as_value_error(self):
        """An instance of ParseError must be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise ParseError("bad input")

    def test_instance_caught_as_meshing_utils_error(self):
        with pytest.raises(MeshingUtilsError):
            raise ParseError("bad input")


# ---------------------------------------------------------------------------
# HexValidationError MRO
# ---------------------------------------------------------------------------

class TestHexValidationErrorMRO:

    def test_is_meshing_utils_error(self):
        assert issubclass(HexValidationError, MeshingUtilsError)

    def test_is_not_value_error(self):
        """HexValidationError must NOT be a ValueError (it has its own semantics)."""
        assert not issubclass(HexValidationError, ValueError)

    def test_instance_caught_as_meshing_utils_error(self):
        with pytest.raises(MeshingUtilsError):
            raise HexValidationError("bad hex")


# ---------------------------------------------------------------------------
# OrderingConsistencyError MRO
# ---------------------------------------------------------------------------

class TestOrderingConsistencyErrorMRO:

    def test_is_meshing_utils_error(self):
        assert issubclass(OrderingConsistencyError, MeshingUtilsError)

    def test_is_not_value_error(self):
        assert not issubclass(OrderingConsistencyError, ValueError)

    def test_instance_caught_as_meshing_utils_error(self):
        with pytest.raises(MeshingUtilsError):
            raise OrderingConsistencyError("bad ordering")


# ---------------------------------------------------------------------------
# ExtrusionError MRO
# ---------------------------------------------------------------------------

class TestExtrusionErrorMRO:

    def test_is_meshing_utils_error(self):
        assert issubclass(ExtrusionError, MeshingUtilsError)

    def test_instance_caught_as_meshing_utils_error(self):
        with pytest.raises(MeshingUtilsError):
            raise ExtrusionError("extrusion failed")


# ---------------------------------------------------------------------------
# All public exception subclasses are distinct (not aliases of each other)
# ---------------------------------------------------------------------------

class TestExceptionDistinctness:

    def test_topology_error_is_not_parse_error(self):
        assert not issubclass(TopologyError, ParseError)
        assert not issubclass(ParseError, TopologyError)

    def test_hex_validation_error_is_not_topology_error(self):
        assert not issubclass(HexValidationError, TopologyError)
        assert not issubclass(TopologyError, HexValidationError)
