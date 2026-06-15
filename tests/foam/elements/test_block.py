"""Tests for meshing_utils.foam.elements.block."""

import pytest

from meshing_utils.foam.elements.block import Block, Blocks

VERTS = ["v108", "v63", "v79", "v111", "v369", "v370", "v371", "v372"]
BLOCK_STR = (
    "name block46 hex (v108 v63 v79 v111 v369 v370 v371 v372) "
    "(30 23 6) simpleGrading (1 1 1)"
)


# ---------------------------------------------------------------------------
# __init__ — string parsing
# ---------------------------------------------------------------------------

def test_block_parse_with_name():
    b = Block(BLOCK_STR)
    assert b.name == "block46"
    assert b.type == "hex"
    assert b.vertices == VERTS
    assert b.cells == [30, 23, 6]
    assert b.grading_type == "simpleGrading"
    assert b.grading_def == pytest.approx([1.0, 1.0, 1.0])


def test_block_parse_without_name():
    s = "hex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 1) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.name == ""
    assert b.type == "hex"
    assert b.vertices == ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    assert b.cells == [10, 10, 1]


def test_block_parse_invalid_raises():
    with pytest.raises(ValueError):
        Block("this is not a valid block string")


def test_block_parse_float_grading():
    s = "hex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 1) simpleGrading (1.5 2.0 1)"
    b = Block(s)
    assert b.grading_def == pytest.approx([1.5, 2.0, 1.0])


# ---------------------------------------------------------------------------
# __init__ — direct parameters
# ---------------------------------------------------------------------------

def test_block_direct_init_with_name():
    b = Block("myBlock", VERTS, [20, 20, 5])
    assert b.name == "myBlock"
    assert b.vertices == VERTS
    assert b.cells == [20, 20, 5]
    assert b.type == "hex"
    assert b.grading_type == "simpleGrading"
    assert b.grading_def == [1, 1, 1]


def test_block_default_constructor():
    b = Block()
    assert b.name == ""
    assert b.vertices == []
    assert b.cells == [1, 1, 1]
    assert b.grading_def == [1, 1, 1]


# ---------------------------------------------------------------------------
# set_cells() / set_grading()
# ---------------------------------------------------------------------------

def test_set_cells_updates_cells():
    b = Block(BLOCK_STR)
    b.set_cells([5, 10, 2])
    assert b.cells == [5, 10, 2]


def test_set_grading_updates_grading_def():
    b = Block(BLOCK_STR)
    b.set_grading([2.0, 1.0, 0.5])
    assert b.grading_def == pytest.approx([2.0, 1.0, 0.5])


# ---------------------------------------------------------------------------
# to_foam_string()
# ---------------------------------------------------------------------------

def test_to_foam_string_with_name():
    b = Block("block46", VERTS, [30, 23, 6])
    result = b.to_foam_string()
    assert result == f"\t{BLOCK_STR}"


def test_to_foam_string_without_name():
    b = Block("", ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"], [10, 10, 1])
    result = b.to_foam_string()
    assert result == "\thex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 1) simpleGrading (1 1 1)"


def test_roundtrip_named_block():
    b = Block(BLOCK_STR)
    assert b.to_foam_string() == f"\t{BLOCK_STR}"


def test_block_parse_with_marker():
    s = BLOCK_STR + " //* tagX"
    b = Block(s)
    assert b.name == "block46"
    assert b.marker == "tagX"


def test_block_to_foam_string_with_marker():
    b = Block("block46", VERTS, [30, 23, 6], marker="tagX")
    assert b.to_foam_string() == f"\t{BLOCK_STR} //* tagX"


# ---------------------------------------------------------------------------
# Blocks container
# ---------------------------------------------------------------------------

def test_blocks_add_and_len():
    bs = Blocks()
    bs.add(Block("b0", VERTS, [1, 1, 1]))
    assert len(bs) == 1


def test_blocks_get():
    bs = Blocks([Block("b0", VERTS, [1, 1, 1])])
    assert bs.get("b0").name == "b0"


def test_blocks_get_missing_raises():
    bs = Blocks()
    with pytest.raises(KeyError):
        bs.get("missing")


def test_blocks_remove():
    bs = Blocks([Block("b0", VERTS, [1, 1, 1]), Block("b1", VERTS, [2, 2, 2])])
    bs.remove("b0")
    assert len(bs) == 1


def test_blocks_contains():
    bs = Blocks([Block("b0", VERTS, [1, 1, 1])])
    assert "b0" in bs
    assert "b1" not in bs


def test_blocks_to_foam_string_empty():
    assert Blocks().to_foam_string() == "blocks\n(\n);"


# ---------------------------------------------------------------------------
# zone attribute — parsing
# ---------------------------------------------------------------------------

def test_parse_block_with_zone_and_name():
    s = "name block0 hex (v0 v1 v2 v3 v4 v5 v6 v7) fluid (10 10 10) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.name == "block0"
    assert b.zone == "fluid"
    assert b.type == "hex"
    assert b.cells == [10, 10, 10]
    assert b.grading_def == pytest.approx([1.0, 1.0, 1.0])


def test_parse_block_with_zone_without_name():
    s = "hex (0 1 2 3 4 5 6 7) solid (5 5 5) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.name == ""
    assert b.zone == "solid"
    assert b.cells == [5, 5, 5]


def test_parse_block_without_zone_backward_compat():
    s = "hex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 1) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.zone is None
    assert b.cells == [10, 10, 1]


def test_parse_block_with_name_no_zone():
    s = "name block0 hex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 10) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.name == "block0"
    assert b.zone is None
    assert b.cells == [10, 10, 10]


# ---------------------------------------------------------------------------
# zone attribute — serialization
# ---------------------------------------------------------------------------

def test_to_foam_string_with_zone():
    b = Block("myBlock", ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
              [10, 10, 10], zone="fluid")
    result = b.to_foam_string()
    assert " fluid " in result
    # Zone token must appear between vertex list ')' and cell list '('
    assert ") fluid (" in result


def test_to_foam_string_without_zone():
    b = Block("", ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"], [10, 10, 1])
    result = b.to_foam_string()
    assert result == "\thex (v0 v1 v2 v3 v4 v5 v6 v7) (10 10 1) simpleGrading (1 1 1)"


def test_roundtrip_parse_serialize_with_zone():
    s = "name block0 hex (v0 v1 v2 v3 v4 v5 v6 v7) fluid (10 10 10) simpleGrading (1 1 1)"
    b = Block(s)
    serialized = b.to_foam_string().strip()
    b2 = Block(serialized)
    assert b2.name == b.name
    assert b2.zone == b.zone
    assert b2.cells == b.cells
    assert b2.grading_def == pytest.approx(b.grading_def)
    assert b2.vertices == b.vertices


def test_named_vertices_with_zone():
    verts = [f"v{i}" for i in range(8)]
    s = f"hex ({' '.join(verts)}) waterRegion (20 20 5) simpleGrading (1 1 1)"
    b = Block(s)
    assert b.zone == "waterRegion"
    assert b.vertices == verts
    assert b.cells == [20, 20, 5]


def test_block_equality_no_eq_defined():
    """Block does not define __eq__, so two instances with identical fields are !=."""
    b1 = Block("myBlock", VERTS, [10, 10, 10], zone="fluid")
    b2 = Block("myBlock", VERTS, [10, 10, 10], zone="fluid")
    assert b1 != b2


def test_zone_none_by_default_in_direct_init():
    b = Block("myBlock", VERTS, [10, 10, 10])
    assert b.zone is None


def test_zone_passed_in_direct_init():
    b = Block("myBlock", VERTS, [10, 10, 10], zone="airRegion")
    assert b.zone == "airRegion"
