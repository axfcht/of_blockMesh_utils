"""Tests for meshing_utils.foam.elements.edge."""

import pytest

from meshing_utils.foam.elements.edge import Edge, Edges

EDGE_STR = "arc v0 v1 (-14.19974991 95.44955266 -34.75000000)"


# ---------------------------------------------------------------------------
# __init__ — string parsing
# ---------------------------------------------------------------------------

def test_edge_parse():
    e = Edge(EDGE_STR)
    assert e.type == "arc"
    assert e.v_start == "v0"
    assert e.v_end == "v1"
    assert e.coords == pytest.approx([-14.19974991, 95.44955266, -34.75])
    assert e.marker is None


def test_edge_parse_with_marker():
    e = Edge("arc v0 v1 (1.0 2.0 3.0) //* tag")
    assert e.type == "arc"
    assert e.coords == pytest.approx([1.0, 2.0, 3.0])
    assert e.marker == "tag"


def test_edge_parse_invalid_raises():
    with pytest.raises(ValueError):
        Edge("not a valid edge string")


def test_edge_parse_spline_type():
    e = Edge("spline v0 v1 (1.0 2.0 3.0)")
    assert e.type == "spline"


# ---------------------------------------------------------------------------
# __init__ — direct parameters
# ---------------------------------------------------------------------------

def test_edge_direct_init():
    e = Edge("arc", "v0", "v1", [1.0, 2.0, 3.0])
    assert e.type == "arc"
    assert e.v_start == "v0"
    assert e.v_end == "v1"
    assert e.coords == pytest.approx([1.0, 2.0, 3.0])
    assert e.marker is None


def test_edge_direct_init_with_marker():
    e = Edge("arc", "v0", "v1", [1.0, 2.0, 3.0], marker="tag")
    assert e.marker == "tag"


def test_edge_default_constructor():
    e = Edge()
    assert e.type == ""
    assert e.v_start == ""
    assert e.v_end == ""
    assert e.coords == pytest.approx([0.0, 0.0, 0.0])
    assert e.marker is None


# ---------------------------------------------------------------------------
# to_foam_string()
# ---------------------------------------------------------------------------

def test_edge_to_foam_string():
    e = Edge("arc", "v0", "v1", [1.0, 2.0, 3.0])
    assert e.to_foam_string() == "\tarc v0 v1 (1.00000000 2.00000000 3.00000000)"


def test_edge_to_foam_string_with_marker():
    e = Edge("arc", "v0", "v1", [1.0, 2.0, 3.0], marker="tag")
    assert e.to_foam_string() == "\tarc v0 v1 (1.00000000 2.00000000 3.00000000) //* tag"


def test_edge_roundtrip():
    e = Edge(EDGE_STR)
    assert e.to_foam_string() == f"\t{EDGE_STR}"


def test_edge_roundtrip_with_marker():
    s = f"{EDGE_STR} //* tag"
    e = Edge(s)
    assert e.to_foam_string() == f"\t{s}"


# ---------------------------------------------------------------------------
# Edges container
# ---------------------------------------------------------------------------

def test_edges_add_and_len():
    es = Edges()
    es.add(Edge("arc", "v0", "v1", [1.0, 2.0, 3.0]))
    es.add(Edge("arc", "v1", "v2", [4.0, 5.0, 6.0]))
    assert len(es) == 2


def test_edges_get():
    e = Edge("arc", "v0", "v1", [1.0, 2.0, 3.0])
    es = Edges([e])
    assert es.get("v0", "v1").type == "arc"


def test_edges_get_missing_raises():
    es = Edges()
    with pytest.raises(KeyError):
        es.get("v0", "v1")


def test_edges_remove():
    es = Edges([
        Edge("arc", "v0", "v1", [1.0, 2.0, 3.0]),
        Edge("arc", "v1", "v2", [4.0, 5.0, 6.0]),
    ])
    es.remove("v0", "v1")
    assert len(es) == 1


def test_edges_to_foam_string_empty():
    assert Edges().to_foam_string() == "edges\n(\n);"
