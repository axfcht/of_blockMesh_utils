# Edge and Edges classes for FOAM blockMeshDict elements.

import re

from meshing_utils.foam.elements.markable import Markable


class Edge(Markable):

    def __init__(self, type_or_string: str = "", v_start: str = "", v_end: str = "",
                 coords: list[float] | None = None, marker: str | None = None,
                 points: list[list[float]] | None = None):
        if coords is None and points is None:
            if type_or_string.strip():
                self._parse(type_or_string)
            else:
                self.type = ""
                self.v_start = ""
                self.v_end = ""
                self.points = [[0.0, 0.0, 0.0]]
                self.marker = None
        else:
            self.type = type_or_string
            self.v_start = v_start
            self.v_end = v_end
            self.marker = marker
            if points is not None:
                self.points = [list(p) for p in points]
            else:
                self.points = [list(coords)]

    @property
    def coords(self) -> list[float]:
        return self.points[0] if self.points else [0.0, 0.0, 0.0]

    @coords.setter
    def coords(self, value: list[float]) -> None:
        if self.points:
            self.points[0] = list(value)
        else:
            self.points = [list(value)]

    def _parse(self, s: str):
        content, marker = self._split_marker(s.strip())
        match = re.match(
            r'^(\w+)\s+(\S+)\s+(\S+)\s+\((.*)\)\s*$',
            content,
            re.DOTALL,
        )
        if not match:
            raise ValueError(f"Cannot parse Edge from string: {s!r}")
        self.type = match.group(1)
        self.v_start = match.group(2)
        self.v_end = match.group(3)
        inner = match.group(4)
        self.marker = marker

        nested = re.findall(r'\(\s*([^)]+?)\s*\)', inner)
        if nested:
            self.points = [[float(x) for x in pt.split()] for pt in nested]
        else:
            tokens = inner.split()
            if len(tokens) != 3:
                raise ValueError(f"Cannot parse Edge coords from string: {s!r}")
            self.points = [[float(tokens[0]), float(tokens[1]), float(tokens[2])]]

    def to_foam_string(self) -> str:
        if len(self.points) == 1:
            x, y, z = self.points[0]
            coords = f"({x:.8f} {y:.8f} {z:.8f})"
            return f"\t{self.type} {self.v_start} {self.v_end} {coords}{self._marker_suffix()}"
        head = f"\t{self.type} {self.v_start} {self.v_end} ("
        body = "\n".join(
            f"\t\t({x:.8f} {y:.8f} {z:.8f})" for x, y, z in self.points
        )
        tail = f"\t){self._marker_suffix()}"
        return f"{head}\n{body}\n{tail}"


class Edges:

    def __init__(self, items: list[Edge] | None = None):
        self._items: list[Edge] = list(items) if items else []

    def add(self, edge: Edge) -> None:
        self._items.append(edge)

    def remove(self, v_start: str, v_end: str) -> None:
        for i, e in enumerate(self._items):
            if e.v_start == v_start and e.v_end == v_end:
                del self._items[i]
                return
        raise KeyError(f"Edge ({v_start}, {v_end}) not found")

    def get(self, v_start: str, v_end: str) -> Edge:
        for e in self._items:
            if e.v_start == v_start and e.v_end == v_end:
                return e
        raise KeyError(f"Edge ({v_start}, {v_end}) not found")

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Edge:
        return self._items[index]

    def __contains__(self, key: tuple[str, str]) -> bool:
        v_start, v_end = key
        return any(e.v_start == v_start and e.v_end == v_end for e in self._items)

    def to_foam_string(self) -> str:
        if not self._items:
            return "edges\n(\n);"
        body = "\n".join(e.to_foam_string() for e in self._items)
        return f"edges\n(\n{body}\n);"
