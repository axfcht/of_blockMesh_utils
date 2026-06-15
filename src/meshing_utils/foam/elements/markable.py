# Markable base class and _clean_lines helper for FOAM elements.


class Markable:
    """Mixin providing inline ``//*``-marker support for FOAM elements."""

    marker: str | None = None

    @staticmethod
    def _split_marker(s: str) -> tuple[str, str | None]:
        if "//*" in s:
            content, _, raw = s.partition("//*")
            return content.rstrip(), raw.strip()
        return s, None

    def _marker_suffix(self) -> str:
        if self.marker is None:
            return ""
        if self.marker == "":
            return " //*"
        return f" //* {self.marker}"

    def has_marker(self) -> bool:
        """Return True if this element carries a //* marker (including label-less)."""
        return self.marker is not None


def _clean_lines(s: str) -> list[str]:
    """Split a multi-line string and strip empty lines and plain ``//`` comments,
    while preserving ``//*`` markers."""
    out: list[str] = []
    for ln in s.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("//") and not ln.startswith("//*"):
            continue
        out.append(ln)
    return out
