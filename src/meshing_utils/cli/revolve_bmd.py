"""CLI entry point: revolve a blockMeshDict around an arbitrary axis.

Usage
-----
    revolveBMD --axisPoint px py pz --axisDir vx vy vz
                         --count N [--angle ALPHA]
                         [--output PATH] [--tol TOL]
                         [--uniquePatches [PATCH ...]]
                         [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

The tool reads ``<case-dir>/system/blockMeshDict``, revolves the mesh
geometry around the specified axis, and writes the result.  By default the
output overwrites the source file in-place after creating a ``.bak`` backup.
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
from meshing_utils.operations.revolve import RevolveConfig, revolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run — main pipeline (separated for testability)
# ---------------------------------------------------------------------------

def run(
    bmd_path: Path,
    axis_point: tuple[float, float, float],
    axis_dir: tuple[float, float, float],
    count: int,
    angle: float,
    tol: float | None,
    output: Path | None,
    unique_patches: list[str] | None = None,
) -> None:
    """Execute the revolve pipeline.

    Parameters
    ----------
    bmd_path:
        Resolved path to the blockMeshDict file.
    axis_point:
        A point on the rotation axis (x, y, z).
    axis_dir:
        Direction vector of the rotation axis.  Normalised internally.
    count:
        Total number of instances including the original (>= 2).
    angle:
        Total rotation angle in degrees.  Must satisfy ``0 < |angle| <= 360``.
    tol:
        Snap-to-grid tolerance for vertex deduplication.  ``None`` activates
        the automatic tolerance derived from the mesh bounding box.
    output:
        Write path for the result.  ``None`` means in-place (overwrite the
        source file).
    unique_patches:
        ``None`` disables the feature; ``[]`` uniquifies all boundary patches;
        a non-empty list uniquifies only the named patches.
    """
    out_path: Path = output if output is not None else bmd_path

    source = BlockMeshDict(bmd_path)
    log_bmd_summary(source, logger=logger)

    cfg = RevolveConfig(
        axis_point=axis_point,
        axis_dir=axis_dir,
        count=count,
        angle=angle,
        tol=tol,
        unique_patches=unique_patches,
    )

    try:
        result = revolve(source, cfg)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    result.write(out_path)
    logger.info(
        "Written revolved blockMeshDict to %s (%d blocks, %d vertices)",
        out_path,
        len(result.blocks),
        len(result.vertices),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``revolve-bmd`` CLI tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Revolve a blockMeshDict around an arbitrary axis, creating "
            "rotational copies of all mesh elements."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  revolveBMD --axisPoint 0 0 0 --axisDir 0 0 1 "
            "--count 4 --angle 360\n"
        ),
    )

    # --- Rotation axis ---
    parser.add_argument(
        "--axisPoint",
        dest="axis_point",
        type=float,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        required=True,
        help="A point on the rotation axis (three floats: px py pz).",
    )
    parser.add_argument(
        "--axisDir",
        dest="axis_dir",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        required=True,
        help=(
            "Direction vector of the rotation axis (three floats: vx vy vz). "
            "Will be normalised automatically.  Must not be the zero vector."
        ),
    )

    # --- Revolution parameters ---
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help=(
            "Total number of instances including the original (>= 2).  "
            "E.g. --count 4 produces 3 additional rotated copies."
        ),
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=360.0,
        help=(
            "Total rotation angle in degrees.  Must satisfy 0 < |angle| <= 360.  "
            "Default: 360.  Negative values reverse the rotation direction."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write path for the result.  When omitted, the source "
            "blockMeshDict is overwritten in-place (after backup)."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help=(
            "Snap-to-grid tolerance for vertex deduplication.  "
            "Default: max(1e-9, 1e-7 * bounding-box diagonal)."
        ),
    )

    # --- Unique patches ---
    parser.add_argument(
        "--uniquePatches",
        dest="unique_patches",
        nargs="*",
        default=None,
        metavar="PATCH",
        help=(
            "Activate unique-patch mode. Without arguments: all boundary patches "
            "are uniquified. With arguments: only the listed patches are uniquified."
        ),
    )

    # --- Common args: --caseDir, --logLevel, --noBackup ---
    add_common_args(parser)

    run_cli(parser, _execute)


def _execute(args: argparse.Namespace) -> None:
    """Translate parsed CLI args into a single ``run()`` invocation."""
    _, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)
    run(
        bmd_path=bmd_path,
        axis_point=tuple(args.axis_point),
        axis_dir=tuple(args.axis_dir),
        count=args.count,
        angle=args.angle,
        tol=args.tol,
        output=args.output,
        unique_patches=args.unique_patches,
    )


if __name__ == "__main__":
    main()
