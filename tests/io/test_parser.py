"""Tests for meshing_utils.io.parser."""

import pytest

from meshing_utils.io.parser import (
    _clean_lines,
    _is_pure_comment,
    _join_balanced,
    _parse_sections,
    _split_patch_blocks,
    _starts_with_keyword,
    _strip_block_comments,
    _strip_inline_comment,
)

# ---------------------------------------------------------------------------
# _strip_block_comments
# ---------------------------------------------------------------------------

def test_strip_block_comments_basic():
    text = "before /* comment */ after"
    assert _strip_block_comments(text) == "before  after"


def test_strip_block_comments_multiline():
    text = "before /* line1\nline2 */ after"
    assert _strip_block_comments(text) == "before  after"


def test_strip_block_comments_no_comment():
    text = "no comments here"
    assert _strip_block_comments(text) == "no comments here"


def test_strip_block_comments_header_style():
    text = (
        "/*--------------------------------*- C++ "
        "-*----------------------------------*\\\n"
        "stuff\n"
        "\\*-----------*/\n"
        "code"
    )
    result = _strip_block_comments(text)
    assert "code" in result
    assert "stuff" not in result


# ---------------------------------------------------------------------------
# _is_pure_comment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("// this is a comment", True),
    ("//no space comment", True),
    ("//* marker", False),       # //* is NOT a pure comment
    ("not a comment", False),
    ("", False),
    ("code // trailing", False), # has content before //
])
def test_is_pure_comment(line, expected):
    assert _is_pure_comment(line) == expected


# ---------------------------------------------------------------------------
# _strip_inline_comment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("code // comment", "code"),
    ("code //* marker", "code //* marker"),  # //* is preserved
    ("no comment", "no comment"),
    ("code   // trailing   ", "code"),
    ("  code  ", "  code"),
])
def test_strip_inline_comment(line, expected):
    assert _strip_inline_comment(line) == expected


# ---------------------------------------------------------------------------
# _clean_lines (BMD version — strips block comments too)
# ---------------------------------------------------------------------------

def test_clean_lines_strips_block_comments():
    text = "/* block comment */\ncode"
    result = _clean_lines(text)
    assert result == ["code"]


def test_clean_lines_strips_pure_line_comments():
    text = "// line comment\ncode"
    result = _clean_lines(text)
    assert result == ["code"]


def test_clean_lines_preserves_marker():
    text = "code //* marker"
    result = _clean_lines(text)
    assert result == ["code //* marker"]


def test_clean_lines_strips_empty_lines():
    text = "\n\ncode\n\n"
    result = _clean_lines(text)
    assert result == ["code"]


def test_clean_lines_strips_inline_comments():
    text = "code // trailing comment"
    result = _clean_lines(text)
    assert result == ["code"]


# ---------------------------------------------------------------------------
# _starts_with_keyword
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, kw, expected", [
    ("vertices", "vertices", True),
    ("vertices (", "vertices", True),
    ("vertices\t(", "vertices", True),
    ("verticesExtra", "vertices", False),
    ("blocks", "vertices", False),
    ("", "vertices", False),
])
def test_starts_with_keyword(line, kw, expected):
    assert _starts_with_keyword(line, kw) == expected


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------

MINIMAL_BMD = """\
convertToMeters 0.001;
vertices
(
    name v0 (0 0 0)
)
edges
(
)
blocks
(
    hex (v0 v0 v0 v0 v0 v0 v0 v0) (1 1 1) simpleGrading (1 1 1)
)
defaultPatch
{
    name myDefault;
    type empty;
}
boundary
(
)
"""


def test_parse_sections_convert_to_meters():
    lines = _clean_lines(MINIMAL_BMD)
    sec = _parse_sections(lines)
    assert sec["convertToMeters"] == pytest.approx(0.001)


def test_parse_sections_vertices_body():
    lines = _clean_lines(MINIMAL_BMD)
    sec = _parse_sections(lines)
    assert any("v0" in ln for ln in sec["vertices_body"])


def test_parse_sections_blocks_body():
    lines = _clean_lines(MINIMAL_BMD)
    sec = _parse_sections(lines)
    assert any("hex" in ln for ln in sec["blocks_body"])


def test_parse_sections_default_patch_body():
    lines = _clean_lines(MINIMAL_BMD)
    sec = _parse_sections(lines)
    assert any("myDefault" in ln for ln in sec["defaultPatch_body"])


def test_parse_sections_missing_convert_to_meters():
    text = "vertices\n(\n)\n"
    lines = _clean_lines(text)
    sec = _parse_sections(lines)
    assert sec["convertToMeters"] is None


# ---------------------------------------------------------------------------
# _join_balanced
# ---------------------------------------------------------------------------

def test_join_balanced_single_balanced_line():
    result = _join_balanced(["arc v0 v1 (1.0 2.0 3.0)"])
    assert result == ["arc v0 v1 (1.0 2.0 3.0)"]


def test_join_balanced_multi_line_merge():
    lines = ["spline v0 v1 (", "(1 2 3)", "(4 5 6)", ")"]
    result = _join_balanced(lines)
    assert len(result) == 1
    assert "spline v0 v1" in result[0]
    assert "(1 2 3)" in result[0]


def test_join_balanced_multiple_balanced_entries():
    lines = [
        "arc v0 v1 (1 2 3)",
        "arc v1 v2 (4 5 6)",
    ]
    result = _join_balanced(lines)
    assert result == lines


def test_join_balanced_empty_input():
    assert _join_balanced([]) == []


# ---------------------------------------------------------------------------
# _split_patch_blocks
# ---------------------------------------------------------------------------

BOUNDARY_BODY = [
    "myPatch",
    "{",
    "type patch;",
    "faces",
    "(",
    "(v0 v1 v5 v4)",
    ");",
    "}",
]


def test_split_patch_blocks_single_patch():
    result = _split_patch_blocks(BOUNDARY_BODY)
    assert len(result) == 1
    assert "myPatch" in result[0]
    assert "type patch;" in result[0]


def test_split_patch_blocks_two_patches():
    body = [
        "patchA",
        "{",
        "type patch;",
        "faces",
        "(",
        "(v0 v1 v5 v4)",
        ");",
        "}",
        "patchB",
        "{",
        "type wall;",
        "faces",
        "(",
        "(v2 v3 v7 v6)",
        ");",
        "}",
    ]
    result = _split_patch_blocks(body)
    assert len(result) == 2
    assert "patchA" in result[0]
    assert "patchB" in result[1]


def test_split_patch_blocks_empty_body():
    result = _split_patch_blocks([])
    assert result == []


def test_split_patch_blocks_skips_parentheses_lines():
    body = ["(", ")", *BOUNDARY_BODY]
    result = _split_patch_blocks(body)
    assert len(result) == 1
