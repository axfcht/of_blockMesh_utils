# Block and Blocks classes for FOAM blockMeshDict elements.

import re

from meshing_utils.foam.elements.markable import Markable


class Block(Markable):

    def __init__(
        self,
        name_or_string: str = "",
        vertices: list[str] | None = None,
        cells: list[int] | None = None,
        type: str = "hex",
        grading_type: str = "simpleGrading",
        grading_def: list | None = None,
        marker: str | None = None,
        zone: str | None = None,
    ):
        if vertices is None:
            if name_or_string.strip():
                self._parse(name_or_string)
            else:
                self.name = ""
                self.type = type
                self.vertices = []
                self.cells = list(cells) if cells else [1, 1, 1]
                self.grading_type = grading_type
                self.grading_def = list(grading_def) if grading_def else [1, 1, 1]
                self.marker = None
                self.zone = zone
        else:
            self.name = name_or_string
            self.type = type
            self.vertices = list(vertices)
            self.cells = list(cells) if cells else [1, 1, 1]
            self.grading_type = grading_type
            self.grading_def = list(grading_def) if grading_def else [1, 1, 1]
            self.marker = marker
            self.zone = zone

    def _parse(self, s: str):
        content, marker = self._split_marker(s.strip())
        # Tokenize-based parser: handles optional zone token between vertex
        # list and cell list.
        #   Syntax without zone: hex (v0..v7) (nx ny nz) grading (...)
        #   Syntax with zone:    hex (v0..v7) zoneName (nx ny nz) grading (...)
        #   Optional name prefix: name <blockName> hex (...)
        match = re.match(
            r'^(?:name\s+(\S+)\s+)?(\w+)\s+\(([^)]+)\)\s*(.*)',
            content
        )
        if not match:
            raise ValueError(f"Cannot parse Block from string: {s!r}")
        self.name = match.group(1) or ""
        self.type = match.group(2)
        self.vertices = match.group(3).split()
        remainder = match.group(4).strip()

        # Determine whether next token is a zone name (bare word, not starting
        # with '(' and not a known grading keyword).
        zone_match = re.match(r'^([A-Za-z_]\S*)\s+(.*)', remainder)
        if zone_match and not zone_match.group(1).startswith("("):
            # Candidate token — accept as zone only when it is not followed
            # immediately by grading_type content without a leading '('
            candidate = zone_match.group(1)
            rest_after_candidate = zone_match.group(2).strip()
            if rest_after_candidate.startswith("("):
                # Next is the cell-count tuple → candidate is a zone name
                self.zone = candidate
                remainder = rest_after_candidate
            else:
                self.zone = None
        else:
            self.zone = None

        cells_grading_match = re.match(
            r'^\(([^)]+)\)\s+(\S+)\s+\(([^)]+)\)',
            remainder
        )
        if not cells_grading_match:
            raise ValueError(f"Cannot parse Block cells/grading from string: {s!r}")
        self.cells = [int(x) for x in cells_grading_match.group(1).split()]
        self.grading_type = cells_grading_match.group(2)
        self.grading_def = [float(x) for x in cells_grading_match.group(3).split()]
        self.marker = marker

    def get_vertex_names(self) -> list[str]:
        return list(self.vertices)

    def get_vertices(self) -> list[str]:
        return list(self.vertices)

    def set_cells(self, cells: list[int]) -> None:
        self.cells = list(cells)

    def set_grading(self, grading_def: list) -> None:
        self.grading_def = list(grading_def)

    def to_foam_string(self) -> str:
        verts = " ".join(self.vertices)
        cells = " ".join(str(c) for c in self.cells)
        grading = " ".join(f"{v:g}" for v in self.grading_def)
        zone_token = f" {self.zone}" if self.zone else ""
        body = f"{self.type} ({verts}){zone_token} ({cells}) {self.grading_type} ({grading})"
        if self.name:
            return f"\tname {self.name} {body}{self._marker_suffix()}"
        return f"\t{body}{self._marker_suffix()}"


class Blocks:

    def __init__(self, items: list[Block] | None = None):
        self._items: list[Block] = list(items) if items else []

    def add(self, block: Block) -> None:
        self._items.append(block)

    def remove(self, name: str) -> None:
        for i, b in enumerate(self._items):
            if b.name == name:
                del self._items[i]
                return
        raise KeyError(f"Block {name!r} not found")

    def get(self, name: str) -> Block:
        for b in self._items:
            if b.name == name:
                return b
        raise KeyError(f"Block {name!r} not found")

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Block:
        return self._items[index]

    def __contains__(self, name: str) -> bool:
        return any(b.name == name for b in self._items)

    def to_foam_string(self) -> str:
        if not self._items:
            return "blocks\n(\n);"
        body = "\n".join(b.to_foam_string() for b in self._items)
        return f"blocks\n(\n{body}\n);"
