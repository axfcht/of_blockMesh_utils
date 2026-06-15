"""Tests for meshing_utils.foam.elements.face."""

import pytest

from meshing_utils.foam.elements.face import Face


def test_face_parse():
    f = Face("(v0 v1 v5 v4)")
    assert f.vertices == ["v0", "v1", "v5", "v4"]
    assert f.marker is None


def test_face_parse_with_marker():
    f = Face("(v0 v1 v5 v4) //* tag")
    assert f.vertices == ["v0", "v1", "v5", "v4"]
    assert f.marker == "tag"


def test_face_parse_invalid_raises():
    with pytest.raises(ValueError):
        Face("not a face")


def test_face_direct_init():
    f = Face(["v0", "v1", "v5", "v4"])
    assert f.vertices == ["v0", "v1", "v5", "v4"]
    assert f.marker is None


def test_face_direct_init_with_marker():
    f = Face(["v0", "v1", "v5", "v4"], marker="tag")
    assert f.marker == "tag"


def test_face_default_constructor():
    f = Face()
    assert f.vertices == []
    assert f.marker is None


def test_face_to_foam_string():
    f = Face(["v0", "v1", "v5", "v4"])
    assert f.to_foam_string() == "\t(v0 v1 v5 v4)"


def test_face_to_foam_string_with_marker():
    f = Face(["v0", "v1", "v5", "v4"], marker="tag")
    assert f.to_foam_string() == "\t(v0 v1 v5 v4) //* tag"


def test_face_roundtrip():
    s = "(v0 v1 v5 v4)"
    assert Face(s).to_foam_string() == f"\t{s}"


def test_face_roundtrip_with_marker():
    s = "(v0 v1 v5 v4) //* tag"
    assert Face(s).to_foam_string() == f"\t{s}"
