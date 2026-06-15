"""CLI entry point: combine multiple blockMeshDict fragments into one file.

Usage
-----
    combineBMD [--caseDir DIR]
               [--logLevel LEVEL]
               [--noBackup]
               [--exclude FILE]...
               [--combineCellZones ZONE_NAME]
               [--vertexTol FLOAT]
               [--strict]

The tool discovers all files in ``<case-dir>/system/`` whose names contain
``"blockMeshDict"`` (excluding the main ``"blockMeshDict"`` file itself),
combines them into a single BlockMeshDict, and writes the result to
``<case-dir>/system/blockMeshDict``.

Source files are processed in alphabetical order.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    run_cli,
)
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.combine import (
    combine_blockmeshdicts,
    discover_source_files,
)

logger = logging.getLogger(__name__)


def _execute(args: argparse.Namespace) -> None:
    """Run the per-fragment combine pipeline."""
    case_dir: Path = args.case_dir if args.case_dir is not None else Path.cwd()
    system_dir: Path = case_dir / "system"
    output_path: Path = system_dir / "blockMeshDict"

    if not system_dir.is_dir():
        logger.error("system/ directory not found: %s", system_dir)
        sys.exit(1)

    try:
        source_paths = discover_source_files(system_dir, excludes=args.exclude)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    for p in source_paths:
        logger.info("Found source: %s", p.name)

    sources: list[BlockMeshDict] = []
    for p in source_paths:
        logger.info("Reading: %s", p)
        sources.append(BlockMeshDict(p))

    try:
        combined = combine_blockmeshdicts(
            sources,
            source_labels=[p.name for p in source_paths],
            combine_cell_zones=args.combine_cell_zones,
            vertex_tol=args.vertex_tol,
            strict=args.strict,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    backup_if_needed(output_path, args.no_backup)
    combined.write(output_path)
    logger.info("Written combined blockMeshDict to %s", output_path)
    logger.info(
        "Done. Combined %d source file(s) into %s.",
        len(source_paths),
        output_path,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``combineBMD`` CLI tool."""
    parser = argparse.ArgumentParser(
        prog="combineBMD",
        description=(
            "Combine multiple blockMeshDict fragment files into a single "
            "blockMeshDict.  Counterpart to splitBMDZones."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="FILE",
        default=None,
        dest="exclude",
        help=(
            "File name (not path) to exclude from discovery. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--combineCellZones",
        dest="combine_cell_zones",
        metavar="ZONE_NAME",
        default=None,
        help=(
            "Override the zone of ALL blocks in the combined output to this "
            "value.  If omitted, each block retains its original zone."
        ),
    )
    parser.add_argument(
        "--vertexTol",
        dest="vertex_tol",
        type=float,
        default=1e-9,
        metavar="FLOAT",
        help=(
            "Tolerance for vertex coordinate conflict detection (Euclidean "
            "distance).  Default: 1e-9."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat edge, patch, face, and scalar conflicts as errors instead "
            "of warnings."
        ),
    )
    add_common_args(parser)

    run_cli(parser, _execute, argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
