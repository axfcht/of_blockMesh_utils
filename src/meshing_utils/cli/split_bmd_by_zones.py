"""CLI entry point: split a blockMeshDict into per-zone files.

Usage
-----
    splitBMDZones [--caseDir DIR]
                    [--include ZONE [ZONE ...] | --exclude ZONE [ZONE ...]]
                    [--outputDir DIR]
                    [--reindexVertices]
                    [--keepEmptyPatches]
                    [--logLevel LEVEL]
                    [--noBackup]

The tool reads ``<case-dir>/system/blockMeshDict``, groups its blocks by their
``zone`` attribute, and writes one ``blockMeshDict_<zone>`` file per group into
``--outputDir`` (default: ``<case-dir>/system/``).

Un-zoned blocks are collected in ``blockMeshDict_no_zone``.

With ``--include``: only the listed zones get individual files; all remaining
blocks (including un-zoned ones) are written to ``blockMeshDict_rest``.

With ``--exclude``: the listed zones are collected into ``blockMeshDict_rest``;
every other zone bucket gets its own file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    resolve_bmd_path,
    run_cli,
)
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.split_by_zones import split_blockmeshdict_by_zones

logger = logging.getLogger(__name__)


def _execute(args: argparse.Namespace) -> None:
    """Run the per-zone splitting pipeline."""
    case_dir, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)

    output_dir: Path = (
        args.output_dir if args.output_dir is not None else case_dir / "system"
    )

    logger.info("Reading blockMeshDict: %s", bmd_path)
    bmd = BlockMeshDict(bmd_path)
    logger.info("Loaded %d block(s)", len(bmd.blocks))

    try:
        written = split_blockmeshdict_by_zones(
            bmd,
            output_dir=output_dir,
            include=args.include,
            exclude=args.exclude,
            reindex_vertices=args.reindex_vertices,
            keep_empty_patches=args.keep_empty_patches,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    for path in written:
        logger.info("Written: %s", path)
    logger.info("Done. %d file(s) written.", len(written))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``splitBMDZones`` CLI tool."""
    parser = argparse.ArgumentParser(
        prog="splitBMDZones",
        description="Split a blockMeshDict into multiple per-zone files.",
    )

    zone_group = parser.add_mutually_exclusive_group()
    zone_group.add_argument(
        "--include",
        nargs="+",
        metavar="ZONE",
        default=None,
        help=(
            "Zone names that get their own output file. All remaining blocks "
            "(including un-zoned ones) are collected into blockMeshDict_rest."
        ),
    )
    zone_group.add_argument(
        "--exclude",
        nargs="+",
        metavar="ZONE",
        default=None,
        help=(
            "Zone names that are collected into blockMeshDict_rest. All other "
            "zone buckets get their own output file."
        ),
    )
    parser.add_argument(
        "--outputDir",
        dest="output_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory where the split files are written. "
            "Defaults to <case-dir>/system/."
        ),
    )
    parser.add_argument(
        "--reindexVertices",
        dest="reindex_vertices",
        action="store_true",
        help=(
            "Rename vertices in each output file to a compact v0…vN-1 sequence "
            "and update all references consistently."
        ),
    )
    parser.add_argument(
        "--keepEmptyPatches",
        dest="keep_empty_patches",
        action="store_true",
        help=(
            "Keep patches that end up with zero faces after zone filtering "
            "(by default such patches are omitted)."
        ),
    )
    add_common_args(parser)

    run_cli(parser, _execute, argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
