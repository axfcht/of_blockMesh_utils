# Face class for FOAM blockMeshDict elements.

import re

from meshing_utils.foam.elements.markable import Markable


class Face(Markable):

    def __init__(self, vertices_or_string="", marker: str | None = None):
        if isinstance(vertices_or_string, str):
            if vertices_or_string.strip():
                self._parse(vertices_or_string)
            else:
                self.vertices: list[str] = []
                self.marker = None
        else:
            self.vertices = list(vertices_or_string)
            self.marker = marker

    def _parse(self, s: str):
        content, marker = self._split_marker(s.strip())
        match = re.match(r'^\(\s*([^)]+?)\s*\)\s*$', content)
        if not match:
            raise ValueError(f"Cannot parse Face from string: {s!r}")
        self.vertices = match.group(1).split()
        self.marker = marker

    def to_foam_string(self) -> str:
        verts = " ".join(self.vertices)
        return f"\t({verts}){self._marker_suffix()}"
