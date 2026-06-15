# Patch, DefaultPatch, and Boundary classes for FOAM blockMeshDict elements.

import re

from meshing_utils.foam.elements.face import Face
from meshing_utils.foam.elements.markable import Markable, _clean_lines


class Patch(Markable):

    def __init__(self, name_or_string: str = "", type: str = "patch",
                 faces: list[Face] | None = None, marker: str | None = None):
        if faces is None and ("\n" in name_or_string or "{" in name_or_string):
            self._parse(name_or_string)
        else:
            self.name = name_or_string
            self.type = type
            self.faces: list[Face] = list(faces) if faces else []
            self.marker = marker

    def _parse(self, s: str):
        lines = _clean_lines(s)
        if not lines:
            raise ValueError(f"Cannot parse Patch from string: {s!r}")

        name_content, marker = self._split_marker(lines[0])
        self.name = name_content.strip()
        self.marker = marker
        self.type = "patch"
        self.faces = []

        i = 1
        n = len(lines)
        while i < n:
            ln = lines[i]
            if ln in ("{", "}"):
                i += 1
                continue
            type_match = re.match(r'^type\s+(\S+?);?\s*$', ln)
            if type_match:
                self.type = type_match.group(1)
                i += 1
                continue
            if ln.startswith("faces"):
                i += 1
                while i < n and not lines[i].startswith("("):
                    i += 1
                if i < n and lines[i] == "(":
                    i += 1
                while i < n and not lines[i].startswith(")"):
                    self.faces.append(Face(lines[i]))
                    i += 1
                i += 1
                continue
            i += 1

    def to_foam_string(self) -> str:
        if self.faces:
            faces_body = "\n".join(f.to_foam_string() for f in self.faces)
            faces_block = f"\t(\n{faces_body}\n\t);"
        else:
            faces_block = "\t(\n\t);"
        return (
            f"{self.name}{self._marker_suffix()}\n"
            f"{{\n"
            f"\ttype {self.type};\n"
            f"\tfaces\n"
            f"{faces_block}\n"
            f"}}"
        )


class DefaultPatch(Markable):

    def __init__(self, name_or_string: str = "", type: str = "empty", marker: str | None = None):
        if "\n" in name_or_string or "{" in name_or_string:
            self._parse(name_or_string)
        else:
            self.name = name_or_string
            self.type = type
            self.marker = marker

    def _parse(self, s: str):
        lines = _clean_lines(s)
        if not lines or "defaultPatch" not in lines[0]:
            raise ValueError(f"Cannot parse DefaultPatch from string: {s!r}")
        _, marker = self._split_marker(lines[0])
        self.marker = marker
        self.name = "default_patch"
        self.type = "empty"
        for ln in lines[1:]:
            if ln in ("{", "}"):
                continue
            name_match = re.match(r'^name\s+(\S+?);?\s*$', ln)
            if name_match:
                self.name = name_match.group(1)
                continue
            type_match = re.match(r'^type\s+(\S+?);?\s*$', ln)
            if type_match:
                self.type = type_match.group(1)

    def to_foam_string(self) -> str:
        return (
            f"defaultPatch{self._marker_suffix()}\n"
            f"{{\n"
            f"\tname {self.name};\n"
            f"\ttype {self.type};\n"
            f"}}"
        )


class Boundary:

    def __init__(self, items: list[Patch] | None = None):
        self._items: list[Patch] = list(items) if items else []

    def add(self, patch: Patch) -> None:
        self._items.append(patch)

    def remove(self, name: str) -> None:
        for i, p in enumerate(self._items):
            if p.name == name:
                del self._items[i]
                return
        raise KeyError(f"Patch {name!r} not found")

    def get(self, name: str) -> Patch:
        for p in self._items:
            if p.name == name:
                return p
        raise KeyError(f"Patch {name!r} not found")

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Patch:
        return self._items[index]

    def __contains__(self, name: str) -> bool:
        return any(p.name == name for p in self._items)

    def to_foam_string(self) -> str:
        if not self._items:
            return "boundary\n(\n);"
        body = "\n".join(p.to_foam_string() for p in self._items)
        return f"boundary\n(\n{body}\n);"
