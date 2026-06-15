"""CLI entry point: parse and re-serialise a blockMeshDict (normalise formatting).

Usage
-----
    cleanBMD [--output PATH]
                       [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

The tool reads ``<case-dir>/system/blockMeshDict``, parses it into the
internal representation, and writes a normalised version.  By default the
output is written to ``<case-dir>/system/clean_blockMeshDict`` to avoid
overwriting the original.  Use ``--output`` to control the destination.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    log_bmd_summary,
    resolve_bmd_path,
    run_cli,
)
from meshing_utils.foam.dict_file import BlockMeshDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run — main pipeline (separated for testability)
# ---------------------------------------------------------------------------

def run(
    bmd_path: Path,
    output: Path | None,
) -> None:
    """Execute the clean pipeline.

    Parameters
    ----------
    bmd_path:
        Resolved path to the blockMeshDict file.
    output:
        Write path for the result.  ``None`` means the default output path
        ``<same-dir>/clean_blockMeshDict``.
    """
    out_path: Path = (
        output if output is not None
        else bmd_path.parent / "clean_blockMeshDict"
    )

    bmd = BlockMeshDict(bmd_path)
    log_bmd_summary(bmd, logger=logger)

    bmd.write(out_path)
    logger.info("Written clean blockMeshDict to %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _execute(args: argparse.Namespace) -> None:
    """Translate parsed CLI args into a single ``run()`` invocation."""
    _, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)
    run(bmd_path=bmd_path, output=args.output)


def main() -> None:
    """Entry point for the ``cleanBMD`` CLI tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse a blockMeshDict and re-serialise it with normalised "
            "formatting.  Useful for stripping comments and re-indenting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cleanBMD --caseDir /path/to/case --output system/blockMeshDict\n"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write path for the result.  When omitted, the result is written "
            "to ``<case-dir>/system/clean_blockMeshDict``."
        ),
    )
    add_common_args(parser)

    run_cli(parser, _execute)


if __name__ == "__main__":
    main()
