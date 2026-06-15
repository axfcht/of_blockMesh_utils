"""CLI entry point: extract boundary patches from a STEP file and append
them to a blockMeshDict.

Usage
-----
    extractPatches [--stpPath PATH]
                          [--tolerance DIST] [--normalAngleTol DEG]
                          [--curvedNormalAngleTol DEG]
                          [--defaultPatchType TYPE] [--strict]
                          [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

The tool reads all solids from a STEP file, matches each hex block face in
``system/blockMeshDict`` against the STP solid surfaces, and appends the
matched faces as named patches to the boundary section.

OCP (PythonOCC) is imported lazily so that the module is always importable
(e.g. for unit-test collection) even when OCP is not installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meshing_utils.cad.step_loader import find_single_step_file, load_solids_with_names
from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    log_bmd_summary,
    resolve_bmd_path,
    run_cli,
)
from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.extract_patches import extract_patches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run — main pipeline (separated for testability)
# ---------------------------------------------------------------------------

def run(
    bmd_path: Path,
    stp_path: Path | None,
    tol: float,
    normal_angle_tol: float,
    curved_normal_angle_tol: float,
    default_patch_type: str,
    strict: bool,
) -> None:
    """Execute the full patch-extraction pipeline.

    Parameters
    ----------
    bmd_path:
        Resolved path to the blockMeshDict file.
    stp_path:
        Path to the STEP file.  When ``None``, the tool looks for a single
        ``*.stp`` / ``*.step`` file in ``<case_dir>/constant/geometry/``.
    tol:
        Distance tolerance in model units.
    normal_angle_tol:
        Angular tolerance in degrees for the normal consistency check on planar
        surfaces.  Cylinder and cone surfaces use ``2 * normal_angle_tol``.
    curved_normal_angle_tol:
        Angular tolerance in degrees for normal consistency on strongly curved
        surfaces.
    default_patch_type:
        OpenFOAM patch type string assigned to newly created patches.
    strict:
        When ``True``, multiple solid matches for a single block face cause
        a ``SystemExit`` instead of only logging a warning.
    """
    case_dir = bmd_path.parent.parent

    # --- Resolve STEP path ---
    if stp_path is not None:
        stp_path = Path(stp_path)
        if not stp_path.exists():
            logger.error("STEP file not found: %s", stp_path)
            sys.exit(1)
    else:
        geometry_dir = case_dir / "constant" / "geometry"
        try:
            stp_path = find_single_step_file(geometry_dir)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    logger.info("Using STEP file: %s", stp_path)

    # --- Load blockMeshDict ---
    bmd = BlockMeshDict(bmd_path)
    log_bmd_summary(bmd, include_boundary=False, logger=logger)

    # --- Load STEP solids ---
    try:
        named_solids = load_solids_with_names(stp_path)
    except (ImportError, RuntimeError) as exc:
        logger.error("Failed to load STEP file: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d solid(s) from %s", len(named_solids), stp_path.name)
    logger.debug(
        "Name sources: %s",
        ", ".join(ns.source for ns in named_solids),
    )

    try:
        patches_added = extract_patches(
            bmd,
            named_solids,
            tol=tol,
            normal_angle_tol=normal_angle_tol,
            curved_normal_angle_tol=curved_normal_angle_tol,
            default_patch_type=default_patch_type,
            strict=strict,
        )
    except MeshingUtilsError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    bmd.write(bmd_path)
    logger.info(
        "Written blockMeshDict to %s (%d new patches added)",
        bmd_path,
        patches_added,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``extract-patches`` CLI tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract boundary patches from a STEP file and append them to a "
            "blockMeshDict."
        )
    )

    parser.add_argument(
        "--stpPath",
        dest="stp_path",
        type=Path,
        default=None,
        help=(
            "Path to the STEP file.  When omitted, the tool looks for a single "
            "*.stp / *.step file in <case-dir>/constant/geometry/."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Distance tolerance in model units for face matching. Default: 1e-4.",
    )
    parser.add_argument(
        "--normalAngleTol",
        dest="normal_angle_tol",
        type=float,
        default=5.0,
        help=(
            "Angular tolerance in degrees for normal consistency check on planar "
            "surfaces.  Cylinder/cone surfaces use 2x this value. Default: 5.0."
        ),
    )
    parser.add_argument(
        "--curvedNormalAngleTol",
        dest="curved_normal_angle_tol",
        type=float,
        default=30.0,
        help=(
            "Angular tolerance in degrees for normal consistency check on strongly "
            "curved surfaces. Default: 30.0."
        ),
    )
    parser.add_argument(
        "--defaultPatchType",
        dest="default_patch_type",
        default="wall",
        help="OpenFOAM patch type for new patches. Default: wall.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat multiple solid matches for a single block face as a fatal error."
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
        stp_path=args.stp_path,
        tol=args.tolerance,
        normal_angle_tol=args.normal_angle_tol,
        curved_normal_angle_tol=args.curved_normal_angle_tol,
        default_patch_type=args.default_patch_type,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
