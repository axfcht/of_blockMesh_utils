# Vertex and Vertices classes for FOAM blockMeshDict elements.

import re

from meshing_utils.foam.elements.markable import Markable


class Vertex(Markable):

    def __init__(
        self,
        name_or_string: str = "",
        coords: list[float] | None = None,
        marker: str | None = None,
    ):
        if coords is None:
            if name_or_string.strip():
                self._parse(name_or_string)
            else:
                self.name = ""
                self.coords = [0.0, 0.0, 0.0]
                self.marker = None
        else:
            self.name = name_or_string
            self.coords = list(coords)
            self.marker = marker

    def _parse(self, s: str):
        content, marker = self._split_marker(s.strip())
        match = re.match(
            r'^(?:name\s+(\S+)\s+)?\(\s*([0-9eE.+\-]+)\s+([0-9eE.+\-]+)\s+([0-9eE.+\-]+)\s*\)',
            content
        )
        if not match:
            raise ValueError(f"Cannot parse Vertex from string: {s!r}")
        self.name = match.group(1) or ""
        self.coords = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
        self.marker = marker

    def add(self, other: "list[float] | Vertex") -> None:
        other_coords = other.coords if isinstance(other, Vertex) else list(other)
        self.coords = [a + b for a, b in zip(self.coords, other_coords, strict=False)]

    def to_foam_string(self) -> str:
        coords = f"({self.coords[0]:.8f} {self.coords[1]:.8f} {self.coords[2]:.8f})"
        if self.name:
            return f"\tname {self.name} {coords}{self._marker_suffix()}"
        return f"\t{coords}{self._marker_suffix()}"


class Vertices:

    def __init__(self, items: list[Vertex] | None = None):
        self._items: list[Vertex] = list(items) if items else []

    def add(self, vertex: Vertex) -> None:
        self._items.append(vertex)

    def remove(self, name: str) -> None:
        for i, v in enumerate(self._items):
            if v.name == name:
                del self._items[i]
                return
        raise KeyError(f"Vertex {name!r} not found")

    def get(self, name: str) -> Vertex:
        for v in self._items:
            if v.name == name:
                return v
        raise KeyError(f"Vertex {name!r} not found")

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Vertex:
        return self._items[index]

    def __contains__(self, name: str) -> bool:
        return any(v.name == name for v in self._items)

    def to_foam_string(self) -> str:
        if not self._items:
            return "vertices\n(\n);"
        body = "\n".join(v.to_foam_string() for v in self._items)
        return f"vertices\n(\n{body}\n);"
