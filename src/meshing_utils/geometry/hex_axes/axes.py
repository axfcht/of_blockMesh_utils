"""``BlockAxis`` dataclass, per-block axis construction, equivalence classes."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from meshing_utils.geometry.hex_axes.detection import (
    _dominant_axis,
    compute_axis_class_per_edge,
    detect_hex_faces,
    map_class_to_local_axis_index,
)

logger = logging.getLogger(__name__)


@dataclass
class BlockAxis:
    """One spatial axis of a single hex block.

    ``axis_index`` is 0/1/2 per Variant 2: the three V0-incident edges
    sorted by neighbour vertex index. ``dominant_global_axis`` is the
    world axis (0=X, 1=Y, 2=Z) whose component in ``direction`` has the
    largest absolute value.
    """

    block_id: int
    axis_index: int
    direction: tuple[float, float, float]
    edge_length: float
    dominant_global_axis: int
    block_vertex_names: tuple[str, ...] = field(default_factory=tuple)
    block_vertex_coords: tuple[tuple[float, float, float], ...] = field(
        default_factory=tuple
    )


def build_block_axes(
    block_id: int,
    vertex_coords: Sequence[tuple[float, float, float]],
    vertex_names: Sequence[str],
) -> list[BlockAxis]:
    """Build the three :class:`BlockAxis` objects for one hex block.

    Detects the topology geometrically (no assumed vertex ordering),
    derives the V0-incident axis order, and produces three axes with
    local indices 0/1/2 per Variant 2.
    """
    if len(vertex_coords) != 8 or len(vertex_names) != 8:
        raise ValueError(
            f"build_block_axes expects exactly 8 vertices, "
            f"got coords={len(vertex_coords)}, names={len(vertex_names)}"
        )

    coords_tuple = tuple(
        (float(c[0]), float(c[1]), float(c[2])) for c in vertex_coords
    )

    detected_faces = detect_hex_faces(coords_tuple)
    edge_to_class = compute_axis_class_per_edge(detected_faces)

    v0_edges: list[tuple[int, frozenset[int]]] = []
    for edge in edge_to_class:
        if 0 in edge:
            other = next(v for v in edge if v != 0)
            v0_edges.append((other, edge))
    v0_edges.sort(key=lambda t: t[0])

    axes: list[BlockAxis] = []
    for axis_idx, (other_vi, _edge) in enumerate(v0_edges):
        p0 = coords_tuple[0]
        p1 = coords_tuple[other_vi]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)

        if length > 0.0:
            nx, ny, nz = dx / length, dy / length, dz / length
        else:
            nx, ny, nz = 1.0, 0.0, 0.0

        dominant = _dominant_axis(nx, ny, nz)

        axes.append(
            BlockAxis(
                block_id=block_id,
                axis_index=axis_idx,
                direction=(nx, ny, nz),
                edge_length=length,
                dominant_global_axis=dominant,
                block_vertex_names=tuple(vertex_names),
                block_vertex_coords=coords_tuple,
            )
        )

    return axes


def compute_longest_edges_per_axis(
    vertex_coords: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the longest edge length per local axis (0, 1, 2) of a hex block.

    For each internal axis class a hex has four parallel edges; this
    helper returns the maximum length per class, keyed by the V0
    Variant-2 indexing convention. Raises :class:`ValueError` if
    *vertex_coords* does not contain exactly 8 points and propagates
    :class:`TopologyError` from :func:`detect_hex_faces`.
    """
    if len(vertex_coords) != 8:
        raise ValueError(
            f"compute_longest_edges_per_axis expects exactly 8 vertices, "
            f"got {len(vertex_coords)}"
        )

    coords = tuple(
        (float(c[0]), float(c[1]), float(c[2])) for c in vertex_coords
    )

    faces = detect_hex_faces(coords)
    edge_to_class = compute_axis_class_per_edge(faces)
    class_to_axis = map_class_to_local_axis_index(edge_to_class, faces)

    longest = [0.0, 0.0, 0.0]
    for edge_fs, class_root in edge_to_class.items():
        axis_idx = class_to_axis[class_root]
        vi_a, vi_b = tuple(edge_fs)
        p_a = coords[vi_a]
        p_b = coords[vi_b]
        dx = p_b[0] - p_a[0]
        dy = p_b[1] - p_a[1]
        dz = p_b[2] - p_a[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > longest[axis_idx]:
            longest[axis_idx] = length

    return (longest[0], longest[1], longest[2])


class _UnionFind:
    """Simple path-compressed weighted Union-Find."""

    def __init__(self, n: int) -> None:
        self._parent: list[int] = list(range(n))
        self._rank: list[int] = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


class AxisEquivalenceClasses:
    """Union-Find over all :class:`BlockAxis` objects across all blocks.

    Two axes are united when they are in-plane on a shared hex face
    (four shared vertex names) and their edge sets on that face match.
    Edge-only contact (fewer than four shared vertices) does not
    trigger a union.

    Usage: call :meth:`add_block_axes` for each block, then
    :meth:`build`, then query :meth:`class_of` or :meth:`all_classes`.
    """

    def __init__(self) -> None:
        self._all_axes: list[BlockAxis] = []
        self._built: bool = False

    def add_block_axes(self, axes: list[BlockAxis]) -> None:
        """Register all three axes for one block."""
        self._all_axes.extend(axes)
        self._built = False

    def build(self) -> None:
        """Perform the Union-Find merging step."""
        n = len(self._all_axes)
        uf = _UnionFind(n)

        self._union_via_shared_faces(uf)

        self._uf = uf
        self._built = True

        num_classes = len(set(uf.find(i) for i in range(n)))
        logger.info(
            "AxisEquivalenceClasses.build: %d axes, %d classes", n, num_classes
        )

    def _union_via_shared_faces(self, uf: _UnionFind) -> None:
        """Unite in-plane axes of blocks that share a complete hex face."""
        face_map: dict[
            frozenset[str],
            list[tuple[int, int, frozenset, frozenset]],
        ] = {}

        for global_idx, axis in enumerate(self._all_axes):
            if axis.axis_index != 0:
                continue
            coords = axis.block_vertex_coords
            names = axis.block_vertex_names
            if not names or not coords:
                continue

            block_global_axis_offset = global_idx

            detected_faces = detect_hex_faces(coords)
            edge_to_class = compute_axis_class_per_edge(detected_faces)
            class_to_axis_idx = map_class_to_local_axis_index(edge_to_class, detected_faces)

            for face in detected_faces:
                vi0, vi1, vi2, vi3 = face
                face_key: frozenset[str] = frozenset(
                    {names[vi0], names[vi1], names[vi2], names[vi3]}
                )

                pair_a_edge1 = frozenset({names[vi0], names[vi1]})
                pair_a_edge2 = frozenset({names[vi2], names[vi3]})
                pair_b_edge1 = frozenset({names[vi1], names[vi2]})
                pair_b_edge2 = frozenset({names[vi3], names[vi0]})

                class_a = edge_to_class[frozenset({vi0, vi1})]
                class_b = edge_to_class[frozenset({vi1, vi2})]

                axis_idx_a = class_to_axis_idx[class_a]
                axis_idx_b = class_to_axis_idx[class_b]

                global_axis_a = block_global_axis_offset + axis_idx_a
                global_axis_b = block_global_axis_offset + axis_idx_b

                face_map.setdefault(face_key, []).append((
                    global_axis_a,
                    global_axis_b,
                    frozenset({pair_a_edge1, pair_a_edge2}),
                    frozenset({pair_b_edge1, pair_b_edge2}),
                ))

        n_unions = 0
        n_shared_faces = 0

        for face_key, entries in face_map.items():
            if len(entries) < 2:
                continue
            n_shared_faces += 1

            for i_entry in range(len(entries)):
                ga_a, gb_a, pairs_a_set, _pairs_b_set = entries[i_entry]
                for j_entry in range(i_entry + 1, len(entries)):
                    ga_b, gb_b, pairs_a_b, _pairs_b_b = entries[j_entry]

                    if pairs_a_set & pairs_a_b:
                        match_a_for_a = ga_b
                        match_b_for_b = gb_b
                    else:
                        match_a_for_a = gb_b
                        match_b_for_b = ga_b

                    if uf.find(ga_a) != uf.find(match_a_for_a):
                        logger.debug(
                            "union via face %s: axis %d <-> axis %d",
                            sorted(face_key),
                            ga_a,
                            match_a_for_a,
                        )
                        n_unions += 1
                    uf.union(ga_a, match_a_for_a)

                    if uf.find(gb_a) != uf.find(match_b_for_b):
                        logger.debug(
                            "union via face %s: axis %d <-> axis %d",
                            sorted(face_key),
                            gb_a,
                            match_b_for_b,
                        )
                        n_unions += 1
                    uf.union(gb_a, match_b_for_b)

        logger.debug(
            "face-based union pass: %d unions across %d shared faces",
            n_unions,
            n_shared_faces,
        )

    def class_of(self, axis_global_index: int) -> int:
        """Return the representative (root) index for the given axis."""
        if not self._built:
            raise RuntimeError("Call build() before querying class_of().")
        return self._uf.find(axis_global_index)

    def all_classes(self) -> dict[int, list[int]]:
        """Return a mapping from representative index to list of member indices."""
        if not self._built:
            raise RuntimeError("Call build() before querying all_classes().")
        classes: dict[int, list[int]] = {}
        for idx in range(len(self._all_axes)):
            root = self._uf.find(idx)
            classes.setdefault(root, []).append(idx)
        return classes

    def get_axis(self, global_index: int) -> BlockAxis:
        """Return the :class:`BlockAxis` at the given global index."""
        return self._all_axes[global_index]

    def num_axes(self) -> int:
        """Return the total number of axes registered."""
        return len(self._all_axes)
