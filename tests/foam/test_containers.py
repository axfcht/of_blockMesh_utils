import pytest

from meshing_utils import (
    Block,
    Blocks,
    Edge,
    Edges,
    Vertex,
    Vertices,
)

# ===========================================================================
# Vertices
# ===========================================================================

def _v(name: str, c=(0.0, 0.0, 0.0)) -> Vertex:
    return Vertex(name, list(c))


def test_vertices_default_is_empty():
    vs = Vertices()
    assert len(vs) == 0
    assert list(vs) == []


def test_vertices_init_with_items():
    items = [_v("v0"), _v("v1")]
    vs = Vertices(items)
    assert len(vs) == 2
    assert vs[0].name == "v0"


def test_vertices_init_stores_copy():
    items = [_v("v0")]
    vs = Vertices(items)
    items.append(_v("v1"))
    assert len(vs) == 1


def test_vertices_add_appends():
    vs = Vertices()
    vs.add(_v("v0"))
    vs.add(_v("v1"))
    assert len(vs) == 2
    assert [v.name for v in vs] == ["v0", "v1"]


def test_vertices_get_returns_vertex():
    vs = Vertices([_v("v0"), _v("v1", (1.0, 2.0, 3.0))])
    found = vs.get("v1")
    assert found.coords == pytest.approx([1.0, 2.0, 3.0])


def test_vertices_get_missing_raises():
    vs = Vertices([_v("v0")])
    with pytest.raises(KeyError):
        vs.get("missing")


def test_vertices_remove_deletes():
    vs = Vertices([_v("v0"), _v("v1"), _v("v2")])
    vs.remove("v1")
    assert [v.name for v in vs] == ["v0", "v2"]


def test_vertices_remove_missing_raises():
    vs = Vertices([_v("v0")])
    with pytest.raises(KeyError):
        vs.remove("missing")


def test_vertices_contains():
    vs = Vertices([_v("v0")])
    assert "v0" in vs
    assert "v1" not in vs


def test_vertices_iter_preserves_order():
    vs = Vertices([_v("a"), _v("b"), _v("c")])
    assert [v.name for v in vs] == ["a", "b", "c"]


def test_vertices_to_foam_string_empty():
    vs = Vertices()
    assert vs.to_foam_string() == "vertices\n(\n);"


def test_vertices_to_foam_string_populated():
    vs = Vertices([_v("v0", (1.0, 2.0, 3.0))])
    expected = "vertices\n(\n\tname v0 (1.00000000 2.00000000 3.00000000)\n);"
    assert vs.to_foam_string() == expected


# ===========================================================================
# Edges
# ===========================================================================

def _e(v_start: str, v_end: str, c=(0.0, 0.0, 0.0)) -> Edge:
    return Edge("arc", v_start, v_end, list(c))


def test_edges_default_is_empty():
    es = Edges()
    assert len(es) == 0


def test_edges_add_and_iter():
    es = Edges()
    es.add(_e("v0", "v1"))
    es.add(_e("v1", "v2"))
    assert len(es) == 2
    assert [(e.v_start, e.v_end) for e in es] == [("v0", "v1"), ("v1", "v2")]


def test_edges_get_returns_edge():
    es = Edges([_e("v0", "v1", (1.0, 2.0, 3.0))])
    e = es.get("v0", "v1")
    assert e.coords == pytest.approx([1.0, 2.0, 3.0])


def test_edges_get_missing_raises():
    es = Edges([_e("v0", "v1")])
    with pytest.raises(KeyError):
        es.get("v0", "v9")


def test_edges_remove_deletes():
    es = Edges([_e("v0", "v1"), _e("v1", "v2")])
    es.remove("v0", "v1")
    assert len(es) == 1
    assert es[0].v_start == "v1"


def test_edges_remove_missing_raises():
    es = Edges()
    with pytest.raises(KeyError):
        es.remove("v0", "v1")


def test_edges_contains():
    es = Edges([_e("v0", "v1")])
    assert ("v0", "v1") in es
    assert ("v0", "v9") not in es


def test_edges_init_stores_copy():
    items = [_e("v0", "v1")]
    es = Edges(items)
    items.append(_e("v1", "v2"))
    assert len(es) == 1


def test_edges_to_foam_string_empty():
    assert Edges().to_foam_string() == "edges\n(\n);"


def test_edges_to_foam_string_populated():
    es = Edges([_e("v0", "v1", (1.0, 2.0, 3.0))])
    expected = "edges\n(\n\tarc v0 v1 (1.00000000 2.00000000 3.00000000)\n);"
    assert es.to_foam_string() == expected


# ===========================================================================
# Blocks
# ===========================================================================

VERTS_A = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
VERTS_B = ["v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"]


def _b(name: str, verts=None) -> Block:
    return Block(name, verts if verts else VERTS_A, [1, 1, 1])


def test_blocks_default_is_empty():
    bs = Blocks()
    assert len(bs) == 0


def test_blocks_add_and_iter():
    bs = Blocks()
    bs.add(_b("block0"))
    bs.add(_b("block1", VERTS_B))
    assert len(bs) == 2
    assert [b.name for b in bs] == ["block0", "block1"]


def test_blocks_get_returns_block():
    bs = Blocks([_b("block0"), _b("block1", VERTS_B)])
    b = bs.get("block1")
    assert b.vertices == VERTS_B


def test_blocks_get_missing_raises():
    bs = Blocks([_b("block0")])
    with pytest.raises(KeyError):
        bs.get("missing")


def test_blocks_remove_deletes():
    bs = Blocks([_b("block0"), _b("block1"), _b("block2")])
    bs.remove("block1")
    assert [b.name for b in bs] == ["block0", "block2"]


def test_blocks_remove_missing_raises():
    bs = Blocks()
    with pytest.raises(KeyError):
        bs.remove("missing")


def test_blocks_contains():
    bs = Blocks([_b("block0")])
    assert "block0" in bs
    assert "block1" not in bs


def test_blocks_init_stores_copy():
    items = [_b("block0")]
    bs = Blocks(items)
    items.append(_b("block1"))
    assert len(bs) == 1


def test_blocks_to_foam_string_empty():
    assert Blocks().to_foam_string() == "blocks\n(\n);"


def test_blocks_to_foam_string_populated():
    bs = Blocks([_b("block0")])
    s = bs.to_foam_string()
    assert s.startswith("blocks\n(\n")
    assert s.endswith("\n);")
    assert "name block0 hex" in s
