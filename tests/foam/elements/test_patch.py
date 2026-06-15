"""Tests for meshing_utils.foam.elements.patch."""

import pytest

from meshing_utils.foam.elements.face import Face
from meshing_utils.foam.elements.patch import Boundary, DefaultPatch, Patch

# ===========================================================================
# Patch
# ===========================================================================

PATCH_STR = """innenring
{
    type patch;
    faces
    (
        (v0 v1 v5 v4)
        (v1 v17 v21 v5)
    );
}"""


def test_patch_parse_name_and_type():
    p = Patch(PATCH_STR)
    assert p.name == "innenring"
    assert p.type == "patch"
    assert len(p.faces) == 2
    assert p.faces[0].vertices == ["v0", "v1", "v5", "v4"]


def test_patch_parse_strips_normal_comments():
    s = """innenring
    {
        // this is ignored
        type patch;
        faces
        (
            // also ignored
            (v0 v1 v5 v4)
        );
    }"""
    p = Patch(s)
    assert len(p.faces) == 1
    assert p.faces[0].vertices == ["v0", "v1", "v5", "v4"]


def test_patch_parse_preserves_face_marker():
    s = """innenring
    {
        type patch;
        faces
        (
            (v0 v1 v5 v4) //* tag
        );
    }"""
    p = Patch(s)
    assert p.faces[0].marker == "tag"


def test_patch_parse_with_marker_on_name():
    s = """innenring //* topMark
    {
        type patch;
        faces
        (
            (v0 v1 v5 v4)
        );
    }"""
    p = Patch(s)
    assert p.marker == "topMark"
    assert p.name == "innenring"


def test_patch_direct_init():
    faces = [Face(["v0", "v1", "v5", "v4"])]
    p = Patch("innenring", "patch", faces)
    assert p.name == "innenring"
    assert p.type == "patch"
    assert len(p.faces) == 1


def test_patch_direct_init_default_type():
    p = Patch("innenring")
    assert p.type == "patch"
    assert p.faces == []


def test_patch_default_constructor():
    p = Patch()
    assert p.name == ""
    assert p.type == "patch"
    assert p.faces == []


def test_patch_to_foam_string_structure():
    faces = [Face(["v0", "v1", "v5", "v4"]), Face(["v1", "v17", "v21", "v5"])]
    p = Patch("innenring", "patch", faces)
    s = p.to_foam_string()
    assert s.startswith("innenring\n{\n")
    assert "\ttype patch;" in s
    assert "\tfaces" in s
    assert "\t(v0 v1 v5 v4)" in s
    assert s.endswith("}")


def test_patch_to_foam_string_with_marker():
    p = Patch("innenring", "patch", [Face(["v0", "v1", "v2", "v3"])], marker="tag")
    assert p.to_foam_string().startswith("innenring //* tag\n")


def test_patch_roundtrip():
    p = Patch(PATCH_STR)
    p2 = Patch(p.to_foam_string())
    assert p2.name == p.name
    assert p2.type == p.type
    assert [f.vertices for f in p2.faces] == [f.vertices for f in p.faces]


# ===========================================================================
# DefaultPatch
# ===========================================================================

DEFAULT_PATCH_STR = """defaultPatch
{
    name connectors;
    type empty;
}"""


def test_default_patch_parse():
    dp = DefaultPatch(DEFAULT_PATCH_STR)
    assert dp.name == "connectors"
    assert dp.type == "empty"


def test_default_patch_parse_invalid_raises():
    with pytest.raises(ValueError):
        DefaultPatch("nope\n{}")


def test_default_patch_direct_init():
    dp = DefaultPatch("connectors", "empty")
    assert dp.name == "connectors"
    assert dp.type == "empty"


def test_default_patch_default_constructor():
    dp = DefaultPatch()
    assert dp.name == ""
    assert dp.type == "empty"


def test_default_patch_to_foam_string():
    dp = DefaultPatch("connectors", "empty")
    expected = "defaultPatch\n{\n\tname connectors;\n\ttype empty;\n}"
    assert dp.to_foam_string() == expected


def test_default_patch_roundtrip():
    dp = DefaultPatch(DEFAULT_PATCH_STR)
    dp2 = DefaultPatch(dp.to_foam_string())
    assert dp2.name == dp.name
    assert dp2.type == dp.type


# ===========================================================================
# Boundary
# ===========================================================================

def _patch(name: str) -> Patch:
    return Patch(name, "patch", [Face(["v0", "v1", "v2", "v3"])])


def test_boundary_default_is_empty():
    b = Boundary()
    assert len(b) == 0


def test_boundary_add_and_iter():
    b = Boundary()
    b.add(_patch("p0"))
    b.add(_patch("p1"))
    assert len(b) == 2
    assert [p.name for p in b] == ["p0", "p1"]


def test_boundary_get_returns_patch():
    b = Boundary([_patch("p0"), _patch("p1")])
    assert b.get("p1").name == "p1"


def test_boundary_get_missing_raises():
    b = Boundary()
    with pytest.raises(KeyError):
        b.get("missing")


def test_boundary_remove_deletes():
    b = Boundary([_patch("p0"), _patch("p1"), _patch("p2")])
    b.remove("p1")
    assert [p.name for p in b] == ["p0", "p2"]


def test_boundary_contains():
    b = Boundary([_patch("p0")])
    assert "p0" in b
    assert "p1" not in b


def test_boundary_to_foam_string_empty():
    assert Boundary().to_foam_string() == "boundary\n(\n);"


def test_boundary_to_foam_string_populated():
    b = Boundary([_patch("p0")])
    s = b.to_foam_string()
    assert s.startswith("boundary\n(\n")
    assert s.endswith("\n);")
    assert "p0\n{" in s
