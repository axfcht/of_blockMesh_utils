"""CLI entry point: extrude marked face(s) in a blockMeshDict.

Layer syntax
------------
The extrusion sequence is built from one or more ``--layer`` flags.  Each
flag takes four arguments: a kind selector followed by an ``(x, y, z)``
displacement vector.

``--layer block dx dy dz``
    NORMAL layer — produces one extruded block whose incremental
    displacement vector is ``(dx, dy, dz)``.

``--layer skip dx dy dz``
    SKIP layer — advances the starting position by ``(dx, dy, dz)``
    without creating a block.  Consecutive SKIP layers are summed
    element-wise and applied to the next NORMAL block.

The order of ``--layer`` flags is preserved.  At least one ``block`` layer
is required and the sequence must not end with a ``skip`` layer.

Examples
--------
::

    extrudeBMD --layer block 0 0 0.2 --layer block 0 0 0.6
    extrudeBMD --layer skip 0 0 0.5 --layer block 0 0 1.0

Usage
-----
::

    extrudeBMD --layer KIND dx dy dz [--layer KIND dx dy dz ...]
                         [--output PATH]
                         [--caseDir DIR] [--logLevel LEVEL] [--noBackup]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    log_bmd_summary,
    resolve_bmd_path,
    run_cli,
)
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.extrusion import (
    ExtrusionError,
    LayerStep,
    ParseError,
    VectorToken,
    build_layer_steps,
    extrude_with_steps,
)

logger = logging.getLogger(__name__)


_LAYER_KINDS = {"block": "NORMAL", "skip": "SKIP"}


def _layers_to_steps(layers: list[list[str]]) -> list[LayerStep]:
    """Convert raw ``--layer`` argument groups into :class:`LayerStep` objects.

    Each entry in *layers* is a 4-element list ``[kind, dx, dy, dz]`` as
    captured by argparse with ``nargs=4``.
    """
    tokens: list[VectorToken] = []
    for raw in layers:
        kind_str, dx_s, dy_s, dz_s = raw
        kind_lower = kind_str.lower()
        if kind_lower not in _LAYER_KINDS:
            raise ParseError(
                f"Invalid layer kind {kind_str!r}; expected 'block' or 'skip'."
            )
        try:
            vec = (float(dx_s), float(dy_s), float(dz_s))
        except ValueError as exc:
            raise ParseError(
                f"Invalid numeric component in --layer {kind_str} "
                f"{dx_s} {dy_s} {dz_s}: {exc}"
            ) from None
        tokens.append(VectorToken(kind=_LAYER_KINDS[kind_lower], vector=vec))
    return build_layer_steps(tokens)


# ---------------------------------------------------------------------------
# run — main pipeline (separated for testability)
# ---------------------------------------------------------------------------

def run(
    bmd_path: Path,
    steps: list[LayerStep],
    output: Path | None,
) -> None:
    """Execute the extrusion pipeline."""
    out_path: Path = (
        output if output is not None
        else bmd_path.parent / "extruded_blockMeshDict"
    )

    bmd = BlockMeshDict(bmd_path)
    log_bmd_summary(bmd, include_boundary=False, logger=logger)

    num_blocks = len(steps)
    logger.info("Extruding with %d block layer(s) ...", num_blocks)

    try:
        result = extrude_with_steps(bmd, steps)
    except ExtrusionError as exc:
        logger.error("Extrusion error: %s", exc)
        sys.exit(1)

    result.write(out_path)
    logger.info(
        "Written extruded blockMeshDict to %s (%d vertices, %d blocks)",
        out_path,
        len(result.vertices),
        len(result.blocks),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``extrude-surfaces`` CLI tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Extrude marked face(s) in a blockMeshDict along given layer vectors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Layer examples:\n"
            "  Two normal layers:\n"
            "    --layer block 0 0 0.2 --layer block 0 0 0.6\n"
            "  One skip then one block:\n"
            "    --layer skip 0 0 0.5 --layer block 0 0 1.0\n"
        ),
    )

    parser.add_argument(
        "--layer",
        action="append",
        nargs=4,
        metavar=("KIND", "DX", "DY", "DZ"),
        required=True,
        help=(
            "Add one layer to the extrusion sequence.  KIND is 'block' for a "
            "NORMAL block layer or 'skip' for a SKIP gap.  May be passed "
            "multiple times; the order is preserved."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write path for the result.  When omitted, the result is written "
            "to ``<case-dir>/system/extruded_blockMeshDict``."
        ),
    )

    # --- Common args: --caseDir, --logLevel, --noBackup ---
    add_common_args(parser)

    run_cli(parser, _execute)


def _execute(args: argparse.Namespace) -> None:
    """Translate parsed CLI args into a single ``run()`` invocation."""
    _, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)
    try:
        steps = _layers_to_steps(args.layer)
    except ParseError as exc:
        logger.error("Error parsing --layer arguments: %s", exc)
        sys.exit(1)
    run(bmd_path=bmd_path, steps=steps, output=args.output)


if __name__ == "__main__":
    main()
