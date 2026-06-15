"""Parser functions for OpenFOAM blockMeshDict files.

Extracted from meshing_utils.block_mesh_dict — no logic changes.
"""

import re

# ---------------------------------------------------------------------------
# Line preprocessing helpers
# ---------------------------------------------------------------------------

def _strip_block_comments(text: str) -> str:
    """Remove all ``/* ... */`` block comments (DOTALL)."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _is_pure_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") and not s.startswith("//*")


def _strip_inline_comment(line: str) -> str:
    """Remove a trailing ``// ...`` comment but preserve ``//*`` markers."""
    i = 0
    n = len(line)
    while i < n - 1:
        if line[i] == "/" and line[i + 1] == "/":
            if i + 2 < n and line[i + 2] == "*":
                return line.rstrip()
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _clean_lines(text: str) -> list[str]:
    """Return non-empty, comment-free lines (preserving ``//*`` markers)."""
    text = _strip_block_comments(text)
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or _is_pure_comment(s):
            continue
        s = _strip_inline_comment(s)
        if s:
            out.append(s)
    return out


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
