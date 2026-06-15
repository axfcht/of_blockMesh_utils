"""Tests for meshing_utils.foam.elements.markable."""


from meshing_utils.foam.elements.markable import Markable, _clean_lines

# ---------------------------------------------------------------------------
# Markable._split_marker
# ---------------------------------------------------------------------------

def test_split_marker_present():
    content, marker = Markable._split_marker("name v0 (1 2 3) //* f1 (1 2 3)")
    assert content == "name v0 (1 2 3)"
    assert marker == "f1 (1 2 3)"


def test_split_marker_absent():
    content, marker = Markable._split_marker("name v0 (1 2 3)")
    assert content == "name v0 (1 2 3)"
    assert marker is None


def test_split_marker_strips_surrounding_whitespace():
    content, marker = Markable._split_marker("name v0 (1 2 3)   //*    tag   ")
    assert content == "name v0 (1 2 3)"
    assert marker == "tag"


def test_split_marker_empty_label():
    content, marker = Markable._split_marker("(1 2 3) //*")
    assert content == "(1 2 3)"
    assert marker == ""


# ---------------------------------------------------------------------------
# Markable._marker_suffix
# ---------------------------------------------------------------------------

class ConcreteMarkable(Markable):
    """Minimal concrete subclass for testing Markable methods."""


def test_marker_suffix_none():
    m = ConcreteMarkable()
    m.marker = None
    assert m._marker_suffix() == ""


def test_marker_suffix_empty_string():
    m = ConcreteMarkable()
    m.marker = ""
    assert m._marker_suffix() == " //*"


def test_marker_suffix_with_label():
    m = ConcreteMarkable()
    m.marker = "boundary_top"
    assert m._marker_suffix() == " //* boundary_top"


# ---------------------------------------------------------------------------
# Markable.has_marker
# ---------------------------------------------------------------------------

def test_has_marker_returns_false_when_none():
    m = ConcreteMarkable()
    m.marker = None
    assert m.has_marker() is False


def test_has_marker_returns_true_when_empty_string():
    m = ConcreteMarkable()
    m.marker = ""
    assert m.has_marker() is True


def test_has_marker_returns_true_when_label():
    m = ConcreteMarkable()
    m.marker = "some_label"
    assert m.has_marker() is True


# ---------------------------------------------------------------------------
# _clean_lines
# ---------------------------------------------------------------------------

def test_clean_lines_strips_empty_lines():
    result = _clean_lines("\n\nfoo\n\nbar\n")
    assert result == ["foo", "bar"]


def test_clean_lines_strips_plain_comments():
    result = _clean_lines("// ignored\nfoo\n// also ignored")
    assert result == ["foo"]


def test_clean_lines_preserves_marker_lines():
    result = _clean_lines("//* marker\nfoo")
    assert result == ["//* marker", "foo"]


def test_clean_lines_strips_indentation():
    result = _clean_lines("   foo   \n   bar   ")
    assert result == ["foo", "bar"]


def test_clean_lines_empty_input():
    result = _clean_lines("")
    assert result == []


def test_clean_lines_only_comments():
    result = _clean_lines("// a\n// b\n// c")
    assert result == []
