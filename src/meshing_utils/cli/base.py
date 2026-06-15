"""Shared CLI helpers — used by all meshing_utils CLI entry points."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meshing_utils.foam.dict_file import BlockMeshDict


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --caseDir, --logLevel, and --noBackup to a parser."""
    parser.add_argument(
        "--caseDir",
        dest="case_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root of the OpenFOAM case. Defaults to the current directory.",
    )
    parser.add_argument(
        "--logLevel",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--noBackup",
        dest="no_backup",
        action="store_true",
        help="Skip creating a .bak backup of the blockMeshDict before writing.",
    )


def configure_logging(args: argparse.Namespace) -> None:
    """Configure root logger from args.log_level."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )


def resolve_bmd_path(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (case_dir, bmd_path) resolved from args.

    Exits with code 1 if blockMeshDict does not exist.
    """
    case_dir: Path = args.case_dir if args.case_dir is not None else Path.cwd()
    bmd_path: Path = case_dir / "system" / "blockMeshDict"
    if not bmd_path.exists():
        logging.getLogger(__name__).error("blockMeshDict not found: %s", bmd_path)
        sys.exit(1)
    return case_dir, bmd_path


def backup_if_needed(bmd_path: Path, no_backup: bool) -> None:
    """Copy bmd_path to <bmd_path>.bak unless no_backup is True."""
    if no_backup or not bmd_path.exists():
        return
    bak = bmd_path.with_suffix(".bak")
    if bak.is_symlink():
        logging.getLogger(__name__).warning(
            "Backup target %s is a symlink; skipping backup to avoid following it.", bak)
        return
    shutil.copy2(bmd_path, bak)
    logging.getLogger(__name__).info("Backed up to %s", bak)


def log_bmd_summary(
    bmd: BlockMeshDict,
    label: str = "Loaded",
    *,
    include_boundary: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Log a ``"<label> blockMeshDict: N vertices, M blocks[, K boundary patches]"`` line.

    Centralises the summary line that previously appeared at the top of
    every CLI ``run()`` function. ``include_boundary=False`` matches the
    shorter form used by ``extrude_surfaces`` and ``extract_patches``.
    """
    log = logger or logging.getLogger(__name__)
    if include_boundary:
        log.info(
            "%s blockMeshDict: %d vertices, %d blocks, %d boundary patches",
            label,
            len(bmd.vertices),
            len(bmd.blocks),
            len(bmd.boundary),
        )
    else:
        log.info(
            "%s blockMeshDict: %d vertices, %d blocks",
            label,
            len(bmd.vertices),
            len(bmd.blocks),
        )


def run_cli(
    parser: argparse.ArgumentParser,
    pipeline: Callable[[argparse.Namespace], None],
    argv: list[str] | None = None,
) -> None:
    """Parse args, configure logging, run ``pipeline(args)``, handle errors.

    The pipeline is wrapped in the standard try/except block that every
    CLI duplicated previously:

    * ``SystemExit`` is re-raised (so explicit ``sys.exit(N)`` from inside
      the pipeline propagates unchanged).
    * Any other ``Exception`` is logged and the process exits with code 1.

    On success this returns normally so tests can call ``main()`` without
    catching ``SystemExit``; the calling script's ``if __name__ ==
    "__main__"`` block handles process exit naturally. ``argv`` is forwarded
    to :meth:`argparse.ArgumentParser.parse_args` for CLIs whose ``main``
    accepts an injectable argument list.
    """
    args = parser.parse_args(argv)
    configure_logging(args)
    logger = logging.getLogger(pipeline.__module__)
    try:
        pipeline(args)
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)
