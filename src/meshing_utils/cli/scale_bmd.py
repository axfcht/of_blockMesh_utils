"""CLI entry point: scale a blockMeshDict by per-axis (or uniform) factors.

Usage
-----
    scaleBMD --factors fx fy fz
                       [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

    scaleBMD --factor s
                       [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

The tool reads ``<case-dir>/system/blockMeshDict``, scales the vertex and
edge coordinates by the given factors, and writes the result.  By default
the output overwrites the source file in-place after creating a ``.bak``
backup.

Exactly one of ``--factors`` (three separate values fx fy fz) or
``--factor`` (a single uniform value applied to all axes) must be supplied.
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
from meshing_utils.operations.scale import scale, validate_factors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run — main pipeline (separated for testability)
# ---------------------------------------------------------------------------

def run(
    bmd_path: Path,
    fx: float,
    fy: float,
    fz: float,
    output: Path | None,
) -> None:
    """Execute the scale pipeline.

    Parameters
    ----------
    bmd_path:
        Resolved path to the blockMeshDict file.
    fx, fy, fz:
        Scale factors for the x, y, and z axes.  Must be strictly positive.
    output:
        Write path for the result.  ``None`` means in-place (overwrite the
        source file).
    """
    out_path: Path = output if output is not None else bmd_path

    source = BlockMeshDict(bmd_path)
    log_bmd_summary(source, logger=logger)

    try:
        validate_factors([fx, fy, fz])
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        result = scale(source, fx, fy, fz)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    result.write(out_path)
    logger.info(
        "Written scaled blockMeshDict to %s (%d vertices, %d blocks)",
        out_path,
        len(result.vertices),
        len(result.blocks),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _execute(args: argparse.Namespace) -> None:
    """Translate parsed CLI args into a single ``run()`` invocation."""
    _, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)
    if args.factors is not None:
        fx, fy, fz = args.factors
    else:
        fx = fy = fz = args.factor
    run(bmd_path=bmd_path, fx=fx, fy=fy, fz=fz, output=args.output)


def main() -> None:
    """Entry point for the ``scaleBMD`` CLI tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Scale a blockMeshDict by per-axis or uniform factors.  "
            "Vertex and edge coordinates are multiplied by the given "
            "factor(s); all other sections are left unchanged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Non-uniform scaling (x*2, y*3, z*1):\n"
            "  scaleBMD --factors 2 3 1\n\n"
            "  # Uniform scaling (all axes * 0.001, e.g. mm to m):\n"
            "  scaleBMD --factor 0.001\n"
        ),
    )

    factor_group = parser.add_mutually_exclusive_group(required=True)
    factor_group.add_argument(
        "--factors",
        type=float,
        nargs=3,
        metavar=("FX", "FY", "FZ"),
        help=(
            "Per-axis scale factors (three floats: fx fy fz).  "
            "Each factor must be strictly positive."
        ),
    )
    factor_group.add_argument(
        "--factor",
        type=float,
        metavar="S",
        help=(
            "Uniform scale factor applied to all three axes.  "
            "Equivalent to --factors S S S.  Must be strictly positive."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write path for the result.  When omitted, the source "
            "blockMeshDict is overwritten in-place (after backup)."
        ),
    )
    add_common_args(parser)

    run_cli(parser, _execute)


if __name__ == "__main__":
    main()
