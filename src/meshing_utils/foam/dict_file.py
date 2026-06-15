"""BlockMeshDict orchestrator: reads, mutates, and writes OpenFOAM
``blockMeshDict`` files using the model classes from :mod:`models`."""

import re
from collections.abc import Iterable
from pathlib import Path

from meshing_utils.foam.elements import (
    Block,
    Blocks,
    Boundary,
    DefaultPatch,
    Edge,
    Edges,
    Face,
    Patch,
    Vertex,
    Vertices,
)
from meshing_utils.foam.templates import (
    BLOCKMESHDICT_HEADER as _HEADER,
)
from meshing_utils.foam.templates import (
    FOOTER as _FOOTER,
)
from meshing_utils.io.parser import _clean_lines

# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

_SECTIONS = {
    "FoamFile": "{",
    "geometry": "{",
    "vertices": "(",
    "edges": "(",
    "blocks": "(",
    "defaultPatch": "{",
    "boundary": "(",
}


def _starts_with_keyword(line: str, kw: str) -> bool:
    if line == kw:
        return True
    if line.startswith(kw):
        rest = line[len(kw):]
        return rest.startswith((" ", "\t"))
    return False


def _parse_sections(lines: list[str]) -> dict:
    out: dict = {
        "convertToMeters": None,
        "geometry_body": [],
        "vertices_body": [],
        "edges_body": [],
        "blocks_body": [],
        "defaultPatch_body": [],
        "boundary_body": [],
    }
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]

        cm = re.match(r"convertToMeters\s+([\d.eE+\-]+)", line)
        if cm:
            out["convertToMeters"] = float(cm.group(1))
            i += 1
            continue

        kw = next((k for k in _SECTIONS if _starts_with_keyword(line, k)), None)
        if kw is None:
            i += 1
            continue

        opener = _SECTIONS[kw]
        closer = ")" if opener == "(" else "}"
        i += 1
        while i < n and not lines[i].startswith(opener):
            i += 1
        if i >= n:
            break
        i += 1  # skip opener line

        depth = 1
        body: list[str] = []
        while i < n:
            ln = lines[i]
            opens = ln.count(opener)
            closes = ln.count(closer)
            new_depth = depth + opens - closes
            if new_depth <= 0:
                i += 1
                break
            body.append(ln)
            depth = new_depth
            i += 1

        out[kw + "_body"] = body
    return out


def _join_balanced(lines: list[str]) -> list[str]:
    """Merge a list of lines so each entry has balanced ``()`` parentheses.

    Lines that already balance on their own pass through unchanged; otherwise,
    consecutive lines are joined until their combined paren count balances.
    """
    out: list[str] = []
    buffer: list[str] = []
    depth = 0
    for ln in lines:
        opens = ln.count("(")
        closes = ln.count(")")
        if not buffer and opens == closes:
            out.append(ln)
            continue
        buffer.append(ln)
        depth += opens - closes
        if depth <= 0:
            out.append(" ".join(buffer))
            buffer = []
            depth = 0
    if buffer:
        out.append(" ".join(buffer))
    return out


def _split_patch_blocks(body_lines: list[str]) -> list[str]:
    """Split a ``boundary``-section body into reconstructed patch strings."""
    patches: list[str] = []
    i = 0
    n = len(body_lines)
    while i < n:
        line = body_lines[i]
        if line in ("(", ")", "(;", ");"):
            i += 1
            continue
        name_line = line
        i += 1
        while i < n and body_lines[i] != "{":
            i += 1
        if i >= n:
            break
        i += 1  # skip '{'
        depth = 1
        patch_body: list[str] = []
        while i < n:
            ln = body_lines[i]
            opens = ln.count("{")
            closes = ln.count("}")
            new_depth = depth + opens - closes
            if new_depth <= 0:
                i += 1
                break
            patch_body.append(ln)
            depth = new_depth
            i += 1
        patches.append(name_line + "\n{\n" + "\n".join(patch_body) + "\n}")
    return patches


# ---------------------------------------------------------------------------
# BlockMeshDict
# ---------------------------------------------------------------------------

class BlockMeshDict:
    """Object-oriented container for an OpenFOAM ``blockMeshDict`` file."""

    def __init__(self, filepath: str | Path | None = None):
        self.filepath: Path | None = Path(filepath) if filepath is not None else None
        self.convertToMeters: float = 1.0
        self.geometry_body: str = ""
        self.vertices = Vertices()
        self.edges = Edges()
        self.blocks = Blocks()
        self.default_patch = DefaultPatch()
        if not self.default_patch.name:
            self.default_patch.name = "defaultFaces"
        self.boundary = Boundary()
        # Lazy snap-to-grid lookup: maps grid-key -> vertex name
        self._vertex_grid: dict[tuple[int, int, int], str] = {}
        # Tolerance used to build _vertex_grid (None = not yet initialised)
        self._vertex_grid_tol: float | None = None

        if self.filepath is not None and self.filepath.exists():
            self._load()

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        text = self.filepath.read_text(encoding="utf-8")
        lines = _clean_lines(text)
        sec = _parse_sections(lines)

        if sec["convertToMeters"] is not None:
            self.convertToMeters = sec["convertToMeters"]

        self.geometry_body = "\n".join(sec["geometry_body"])

        self.vertices = Vertices([Vertex(ln) for ln in sec["vertices_body"]])
        self.edges = Edges([Edge(ln) for ln in _join_balanced(sec["edges_body"])])
        self.blocks = Blocks([Block(ln) for ln in sec["blocks_body"]])

        if sec["defaultPatch_body"]:
            dp_str = "defaultPatch\n{\n" + "\n".join(sec["defaultPatch_body"]) + "\n}"
            self.default_patch = DefaultPatch(dp_str)

        if sec["boundary_body"]:
            patches = [Patch(ps) for ps in _split_patch_blocks(sec["boundary_body"])]
            self.boundary = Boundary(patches)

    # -- marker access ----------------------------------------------------

    def get_marked(self, type_filter: type | None = None) -> list:
        results: list = []
        for v in self.vertices:
            if v.marker:
                results.append(v)
        for e in self.edges:
            if e.marker:
                results.append(e)
        for b in self.blocks:
            if b.marker:
                results.append(b)
        if self.default_patch.marker:
            results.append(self.default_patch)
        for p in self.boundary:
            if p.marker:
                results.append(p)
            for f in p.faces:
                if f.marker:
                    results.append(f)
        if type_filter is not None:
            results = [r for r in results if isinstance(r, type_filter)]
        return results

    # -- layout helpers ---------------------------------------------------

    def _block_order(self) -> list[str]:
        return [b.name for b in self.blocks if b.name]

    def _vertex_owner_map(self) -> dict[str, str]:
        owner: dict[str, str] = {}
        for b in self.blocks:
            if not b.name:
                continue
            for vname in b.vertices:
                owner.setdefault(vname, b.name)
        return owner

    def _edge_owner_map(self) -> dict[tuple[str, str], str]:
        owner: dict[tuple[str, str], str] = {}
        block_sets: list[tuple[str, set[str]]] = [
            (b.name, set(b.vertices)) for b in self.blocks if b.name
        ]
        for e in self.edges:
            for name, vset in block_sets:
                if e.v_start in vset and e.v_end in vset:
                    owner[(e.v_start, e.v_end)] = name
                    break
        return owner

    def _face_block_ref(self, face: Face) -> str:
        face_set = set(face.vertices)
        for b in self.blocks:
            if not b.name:
                continue
            if face_set.issubset(set(b.vertices)):
                return b.name
        return ""

    def _grouped(self, items: Iterable, owner: dict, key_fn) -> list[tuple[str, list]]:
        order = self._block_order()
        groups: dict[str, list] = {bn: [] for bn in order}
        unreferenced: list = []
        for it in items:
            k = key_fn(it)
            if k in owner:
                groups[owner[k]].append(it)
            else:
                unreferenced.append(it)
        out: list[tuple[str, list]] = []
        for bn in order:
            if groups[bn]:
                out.append((bn, groups[bn]))
        if unreferenced:
            out.append(("", unreferenced))
        return out

    # -- new API: vertex naming -------------------------------------------

    def _next_vertex_name(self) -> str:
        """Return the next available vertex name of the form ``v<n>``.

        The index is ``max(existing v<int> indices) + 1``, or ``0`` when no
        such vertices exist yet.  Vertices whose names do not match the
        ``v<int>`` pattern are ignored.
        """
        max_idx = -1
        pattern = re.compile(r"^v(\d+)$")
        for v in self.vertices:
            m = pattern.match(v.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return f"v{max_idx + 1}"

    def _grid_key(self, coord: tuple[float, float, float], tol: float) -> tuple[int, int, int]:
        """Return the integer snap-to-grid key for *coord* at *tol* resolution."""
        return (
            round(coord[0] / tol),
            round(coord[1] / tol),
            round(coord[2] / tol),
        )

    def find_or_add_vertex(
        self,
        coord: tuple[float, float, float],
        tol: float,
    ) -> str:
        """Return the name of the vertex at *coord*, creating one if absent.

        Uses a snap-to-grid approach: coordinates that map to the same integer
        grid cell (rounded at resolution *tol*) are considered identical.  When
        the tolerance changes between calls the internal grid is rebuilt.

        Parameters
        ----------
        coord:
            ``(x, y, z)`` tuple of the vertex position.
        tol:
            Grid cell size for the snap-to-grid lookup.

        Returns
        -------
        str
            The name of the existing or newly created vertex.
        """
        if tol != self._vertex_grid_tol:
            # Rebuild grid with the new tolerance
            self._vertex_grid = {}
            self._vertex_grid_tol = tol
            for v in self.vertices:
                key = self._grid_key((v.coords[0], v.coords[1], v.coords[2]), tol)
                if key not in self._vertex_grid:
                    self._vertex_grid[key] = v.name

        key = self._grid_key(coord, tol)
        if key in self._vertex_grid:
            return self._vertex_grid[key]

        name = self._next_vertex_name()
        self.vertices.add(Vertex(name, list(coord)))
        self._vertex_grid[key] = name
        return name

    # -- new API: block queries -------------------------------------------

    def has_block_with_vertex_set(self, names: list[str]) -> bool:
        """Return ``True`` when a block whose vertex set equals *names* exists.

        The comparison is order-independent (set equality).
        """
        target = set(names)
        return any(set(b.vertices) == target for b in self.blocks)

    def has_block_named(self, name: str) -> bool:
        """Return ``True`` when a block with the given *name* exists."""
        return name in self.blocks

    def next_block_index(self) -> int:
        """Return ``max(existing block<int> indices) + 1``, or ``0`` if none.

        Blocks whose names do not match the ``block<int>`` pattern are ignored.
        """
        max_idx = -1
        pattern = re.compile(r"^block(\d+)$")
        for b in self.blocks:
            m = pattern.match(b.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1

    # -- new API: edge queries --------------------------------------------

    def find_edge(self, name1: str, name2: str) -> Edge | None:
        """Return the :class:`~meshing_utils.models.Edge` connecting the
        two named vertices, or ``None`` if no such edge exists.

        The lookup is symmetric: ``find_edge(a, b)`` and ``find_edge(b, a)``
        return the same edge regardless of the stored direction.
        """
        for e in self.edges:
            if (e.v_start == name1 and e.v_end == name2) or \
               (e.v_start == name2 and e.v_end == name1):
                return e
        return None

    # -- writing ----------------------------------------------------------

    def write(self, filepath: str | Path) -> None:
        parts: list[str] = []
        parts.append(_HEADER)
        parts.append("")
        parts.append(f"convertToMeters {self.convertToMeters:g};")
        parts.append("")

        parts.append("geometry")
        parts.append("{")
        if self.geometry_body.strip():
            parts.append(self.geometry_body)
        parts.append("}")
        parts.append("")

        # Vertices
        parts.append("vertices")
        parts.append("(")
        v_owner = self._vertex_owner_map()
        v_groups = self._grouped(self.vertices, v_owner, lambda v: v.name)
        for idx, (label, items) in enumerate(v_groups):
            if idx > 0:
                parts.append("")
            comment = f"\t// Vertices {label}" if label else "\t// Unreferenced vertices"
            parts.append(comment)
            for v in items:
                parts.append(v.to_foam_string())
        parts.append(");")
        parts.append("")

        # Edges
        parts.append("edges")
        parts.append("(")
        e_owner = self._edge_owner_map()
        e_groups = self._grouped(self.edges, e_owner, lambda e: (e.v_start, e.v_end))
        for idx, (label, items) in enumerate(e_groups):
            if idx > 0:
                parts.append("")
            comment = f"\t// Edges for {label}" if label else "\t// Unreferenced edges"
            parts.append(comment)
            for e in items:
                parts.append(e.to_foam_string())
        parts.append(");")
        parts.append("")

        # Blocks
        parts.append("blocks")
        parts.append("(")
        for b in self.blocks:
            parts.append(b.to_foam_string())
        parts.append(");")
        parts.append("")

        # DefaultPatch
        parts.append(self.default_patch.to_foam_string())
        parts.append("")

        # Boundary (rendered with proper nesting + face block-ref annotations)
        parts.append("boundary")
        parts.append("(")
        for idx, p in enumerate(self.boundary):
            if idx > 0:
                parts.append("")
            parts.append(f"\t{p.name}{p._marker_suffix()}")
            parts.append("\t{")
            parts.append(f"\t\ttype {p.type};")
            parts.append("\t\tfaces")
            parts.append("\t\t(")
            for f in p.faces:
                verts = " ".join(f.vertices)
                if f.marker:
                    parts.append(f"\t\t\t({verts}) //* {f.marker}")
                else:
                    ref = self._face_block_ref(f)
                    if ref:
                        parts.append(f"\t\t\t({verts})\t// {ref}")
                    else:
                        parts.append(f"\t\t\t({verts})")
            parts.append("\t\t);")
            parts.append("\t}")
        parts.append(");")
        parts.append("")
        parts.append(_FOOTER)
        parts.append("")

        Path(filepath).write_text("\n".join(parts), encoding="utf-8")
