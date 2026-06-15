"""Token models and offset-string parser for the extrusion CLI.

Encodes the skip-aware syntax: ``(x y z)`` is a NORMAL block-producing
vector, ``[x y z]`` is a SKIP gap that contributes to the next block's
``skip_offset``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.foam.dict_file import BlockMeshDict


@dataclass
class VectorToken:
    """A single parsed vector token from an offset string.

    ``kind`` is ``"NORMAL"`` for ``(x y z)`` block tokens or ``"SKIP"``
    for ``[x y z]`` gap markers. ``vector`` is the (x, y, z) displacement.
    """
    kind: Literal["NORMAL", "SKIP"]
    vector: tuple[float, float, float]


@dataclass
class LayerStep:
    """Describes one extruded block: an optional skip gap followed by the block.

    ``skip_offset`` is the cumulative displacement of all leading SKIP tokens
    before the block (``(0, 0, 0)`` when there is no skip).
    ``block_delta`` is the incremental displacement vector of the block itself.
    """
    skip_offset: tuple[float, float, float]
    block_delta: tuple[float, float, float]


@dataclass
class ExtrusionRequest:
    """Encapsulates a single extrusion operation."""
    source_bmd: BlockMeshDict
    cumulative_offsets: list[tuple[float, float, float]]


class ParseError(MeshingUtilsError, ValueError):
    """Raised when an offset string cannot be parsed into valid LayerSteps."""


_FLOAT_PAT = r"[0-9eE.+\-]+"
_NORMAL_RE = re.compile(
    r"\(\s*(" + _FLOAT_PAT + r")\s+(" + _FLOAT_PAT + r")\s+(" + _FLOAT_PAT + r")\s*\)"
)
_SKIP_RE = re.compile(
    r"\[\s*(" + _FLOAT_PAT + r")\s+(" + _FLOAT_PAT + r")\s+(" + _FLOAT_PAT + r")\s*\]"
)


def _tokenize(inner: str) -> list[VectorToken]:
    """Extract all NORMAL ``(...)`` and SKIP ``[...]`` tokens from *inner*."""
    tokens: list[VectorToken] = []
    events: list[tuple[int, str, re.Match]] = []
    for m in _NORMAL_RE.finditer(inner):
        events.append((m.start(), "NORMAL", m))
    for m in _SKIP_RE.finditer(inner):
        events.append((m.start(), "SKIP", m))
    events.sort(key=lambda t: t[0])
    for _, kind, m in events:
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        tokens.append(VectorToken(kind=kind, vector=(x, y, z)))
    return tokens


def _vec_add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the element-wise sum of two 3-tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def build_layer_steps(tokens: list[VectorToken]) -> list[LayerStep]:
    """Fold a sequence of NORMAL/SKIP tokens into :class:`LayerStep` objects.

    Raises :class:`ParseError` when the token sequence is empty, contains no
    NORMAL token, or ends on a SKIP token.
    """
    if not tokens:
        raise ParseError("No vectors provided.")

    if not any(t.kind == "NORMAL" for t in tokens):
        raise ParseError(
            "Token sequence contains only SKIP markers — at least one "
            "NORMAL vector is required."
        )

    if tokens[-1].kind == "SKIP":
        raise ParseError("Token sequence must not end with a SKIP marker.")

    steps: list[LayerStep] = []
    accumulated_skip: tuple[float, float, float] = (0.0, 0.0, 0.0)
    for token in tokens:
        if token.kind == "SKIP":
            accumulated_skip = _vec_add(accumulated_skip, token.vector)
        else:
            steps.append(LayerStep(
                skip_offset=accumulated_skip,
                block_delta=token.vector,
            ))
            accumulated_skip = (0.0, 0.0, 0.0)
    return steps


def parse_layer_steps(arg: str) -> list[LayerStep]:
    """Parse an offset string into a list of :class:`LayerStep` objects.

    Supported syntax (outer parentheses required):

    * ``(x y z)``  — NORMAL token: produces one extrusion block.
    * ``[x y z]``  — SKIP token: skips a gap before the next NORMAL block.
    """
    stripped = arg.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        raise ParseError(
            f"Offset string must be wrapped in parentheses, got: {arg!r}"
        )

    inner = stripped[1:-1]
    tokens = _tokenize(inner)

    if not tokens:
        raise ParseError(f"No vectors found in offset string: {arg!r}")

    try:
        return build_layer_steps(tokens)
    except ParseError as exc:
        raise ParseError(f"{exc} (offset string: {arg!r})") from None


def parse_offsets(arg: str) -> list[tuple[float, float, float]]:
    """Parse an offset string into a list of incremental (x, y, z) tuples.

    Raises :class:`ParseError` on invalid input or ``ValueError`` if any
    *block* vector has zero length.
    """
    steps = parse_layer_steps(arg)
    result: list[tuple[float, float, float]] = []
    for step in steps:
        x, y, z = step.block_delta
        length = math.sqrt(x * x + y * y + z * z)
        if length == 0.0:
            raise ValueError("Zero-length offset vector is not allowed.")
        result.append((x, y, z))
    return result


def cumulative_offsets(
    offsets: list[tuple[float, float, float]]
) -> list[tuple[float, float, float]]:
    """Convert incremental offset vectors to cumulative (absolute) offsets."""
    result: list[tuple[float, float, float]] = []
    acc: tuple[float, float, float] = (0.0, 0.0, 0.0)
    for dx, dy, dz in offsets:
        acc = (acc[0] + dx, acc[1] + dy, acc[2] + dz)
        result.append(acc)
    return result
