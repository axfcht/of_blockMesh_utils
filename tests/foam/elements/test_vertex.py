"""Tests for meshing_utils.foam.elements.vertex."""

import pytest

from meshing_utils.foam.elements.vertex import Vertex, Vertices

# ---------------------------------------------------------------------------
# __init__ — string parsing
# ---------------------------------------------------------------------------

def test_parse_string_with_name_and_trailing_space():
    v = Vertex("name v17 (0.00000000 96.50000000 -34.75000000 )")
    assert v.name == "v17"
    assert v.coords == pytest.approx([0.0, 96.5, -34.75])


def test_parse_string_with_name_no_trailing_space():
    v = Vertex("name v17 (0.00000000 96.50000000 -34.75000000)")
    assert v.name == "v17"
    assert v.coords == pytest.approx([0.0, 96.5, -34.75])


def test_parse_string_without_name():
    v = Vertex("(1.0 2.0 3.0)")
    assert v.name == ""
    assert v.coords == pytest.approx([1.0, 2.0, 3.0])


def test_parse_string_invalid_raises():
    with pytest.raises(ValueError):
        Vertex("this is not a valid vertex string")


# ---------------------------------------------------------------------------
# __init__ — direct coordinates
# ---------------------------------------------------------------------------

def test_init_with_name_and_coords():
    v = Vertex("v3", [1.0, 2.0, 3.0])
    assert v.name == "v3"
    assert v.coords == pytest.approx([1.0, 2.0, 3.0])


def test_init_with_empty_name_and_coords():
    v = Vertex("", [0.0, 0.0, 0.0])
    assert v.name == ""
    assert v.coords == pytest.approx([0.0, 0.0, 0.0])


def test_init_default_is_empty():
    v = Vertex()
    assert v.name == ""
    assert v.coords == pytest.approx([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

def test_add_list():
    v = Vertex("v1", [1.0, 2.0, 3.0])
    v.add([0.5, -1.0, 2.0])
    assert v.coords == pytest.approx([1.5, 1.0, 5.0])


def test_add_vertex():
    v1 = Vertex("v1", [1.0, 2.0, 3.0])
    v2 = Vertex("v2", [10.0, 20.0, 30.0])
    v1.add(v2)
    assert v1.coords == pytest.approx([11.0, 22.0, 33.0])


def test_add_zeros_is_identity():
    v = Vertex("v", [3.0, 4.0, 5.0])
    v.add([0.0, 0.0, 0.0])
    assert v.coords == pytest.approx([3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# to_foam_string()
# ---------------------------------------------------------------------------

def test_to_foam_string_with_name():
    v = Vertex("v17", [0.0, 96.5, -34.75])
    result = v.to_foam_string()
    assert result == "\tname v17 (0.00000000 96.50000000 -34.75000000)"


def test_to_foam_string_without_name():
    v = Vertex("", [1.0, 2.0, 3.0])
    result = v.to_foam_string()
    assert result == "\t(1.00000000 2.00000000 3.00000000)"


def test_roundtrip_named_vertex():
    original = "name v17 (0.00000000 96.50000000 -34.75000000)"
    v = Vertex(original)
    assert v.to_foam_string() == f"\t{original}"


# ---------------------------------------------------------------------------
# marker integration
# ---------------------------------------------------------------------------

def test_vertex_parse_with_marker():
    v = Vertex("name v0 (1.0 2.0 3.0) //* f1 (1 2 3)")
    assert v.name == "v0"
    assert v.coords == pytest.approx([1.0, 2.0, 3.0])
    assert v.marker == "f1 (1 2 3)"


def test_vertex_direct_init_with_marker():
    v = Vertex("v0", [1.0, 2.0, 3.0], marker="tag")
    assert v.marker == "tag"


def test_vertex_to_foam_string_with_marker():
    v = Vertex("v0", [1.0, 2.0, 3.0], marker="f1 (1 2 3)")
    assert v.to_foam_string() == "\tname v0 (1.00000000 2.00000000 3.00000000) //* f1 (1 2 3)"


# ---------------------------------------------------------------------------
# Vertices container
# ---------------------------------------------------------------------------

def test_vertices_add_and_len():
    vs = Vertices()
    vs.add(Vertex("v0", [0.0, 0.0, 0.0]))
    vs.add(Vertex("v1", [1.0, 0.0, 0.0]))
    assert len(vs) == 2


def test_vertices_get():
    vs = Vertices([Vertex("v0", [0.0, 0.0, 0.0])])
    assert vs.get("v0").name == "v0"


def test_vertices_get_missing_raises():
    vs = Vertices()
    with pytest.raises(KeyError):
        vs.get("missing")


def test_vertices_remove():
    vs = Vertices([Vertex("v0", [0.0, 0.0, 0.0]), Vertex("v1", [1.0, 0.0, 0.0])])
    vs.remove("v0")
    assert len(vs) == 1


def test_vertices_contains():
    vs = Vertices([Vertex("v0", [0.0, 0.0, 0.0])])
    assert "v0" in vs
    assert "v1" not in vs


def test_vertices_to_foam_string_empty():
    assert Vertices().to_foam_string() == "vertices\n(\n);"


def test_vertices_to_foam_string_populated():
    vs = Vertices([Vertex("v0", [1.0, 2.0, 3.0])])
    s = vs.to_foam_string()
    assert s.startswith("vertices\n(\n")
    assert s.endswith("\n);")
