"""CLI entry point: assign cellZone names to hex blocks based on STEP solid containment.

Usage
-----
    extractCZones [--stpPath PATH]
                         [--tolerance TOL] [--strict]
                         [--samplingStrategy {centroid,inset}]
                         [--insetFactor F]
                         [--votePolicy {majority}]
                         [--naming {auto,generic}]
                         [--caseDir DIR] [--logLevel LEVEL] [--noBackup]

The tool reads all solids from a STEP file, tests each hex block's sample
points for point-in-solid containment, and writes the resulting zone names
into the ``blocks`` section of ``system/blockMeshDict``.

``--naming auto`` (default) uses the three-path name extraction strategy from
:mod:`meshing_utils.cad.step_names` to obtain reliable solid names.
``--naming generic`` reproduces the legacy behaviour (``zone{i}`` names).

OCP (PythonOCC) is imported lazily so that the module is always importable
(e.g. for unit-test collection) even when OCP is not installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meshing_utils.cad.step_loader import (
    find_single_step_file,
    load_solids_with_names,
    load_step_solids,
)
from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    resolve_bmd_path,
    run_cli,
)
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.cell_zones import assign_cell_zones

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``extractCZones`` CLI tool."""
    parser = argparse.ArgumentParser(
        prog="extractCZones",
        description=(
            "Assign cellZone names to hex blocks based on containment in "
            "STEP solids."
        ),
    )

    parser.add_argument(
        "--stpPath",
        dest="stp_path",
        type=Path,
        default=None,
        help=(
            "Path to STEP file. Defaults to single .stp/.step file in "
            "<case-dir>/constant/geometry/."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="OCC classifier tolerance for point-in-solid test. Default: 1e-7.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat ambiguous block-to-solid assignments as fatal "
            "(>=2 IN-hits, or unresolved >=2 ON-hits after perturbation)."
        ),
    )
    parser.add_argument(
        "--samplingStrategy",
        dest="sampling_strategy",
        choices=["centroid", "inset"],
        default="inset",
        help=(
            "Sample-point strategy. 'inset' (default): centroid + 8 inset samples + 8 raw "
            "vertices (two-stage decision — robust for thin/curved boundary blocks). "
            "'centroid': centroid only (legacy single-sample mode)."
        ),
    )
    parser.add_argument(
        "--insetFactor",
        dest="inset_factor",
        type=float,
        default=0.5,
        help=(
            "Inset factor f in (0, 1) for the 'inset' strategy. "
            "Sample = centroid + f * (vertex - centroid). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--votePolicy",
        dest="vote_policy",
        choices=["majority"],
        default="majority",
        help="Aggregation policy for multi-sample results. Currently only 'majority'.",
    )
    parser.add_argument(
        "--noAabbFilter",
        dest="no_aabb_filter",
        action="store_true",
        default=False,
        help=(
            "Disable the axis-aligned bounding box pre-filter "
            "(diagnostic; slower but identical result)."
        ),
    )
    parser.add_argument(
        "--naming",
        choices=["auto", "generic"],
        default="auto",
        help=(
            "Solid-name extraction strategy. "
            "'auto' (default): use three-path STEP name extraction for reliable names. "
            "'generic': use legacy zone{i} names (reproduces old behaviour)."
        ),
    )

    # Common args: --caseDir, --logLevel, --noBackup
    add_common_args(parser)

    run_cli(parser, _execute, argv)
    return 0


def _execute(args: argparse.Namespace) -> None:
    """Run the cell-zone assignment pipeline."""
    case_dir, bmd_path = resolve_bmd_path(args)
    backup_if_needed(bmd_path, args.no_backup)

    # --- STEP file resolution ---
    if args.stp_path is None:
        geometry_dir = case_dir / "constant" / "geometry"
        try:
            stp_path = find_single_step_file(geometry_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        stp_path = args.stp_path
        if not stp_path.exists():
            logger.error("STEP file not found: %s", stp_path)
            sys.exit(1)

    logger.info("Using STEP file: %s", stp_path)

    # --- Load blockMeshDict ---
    bmd = BlockMeshDict(bmd_path)
    logger.info("Loaded blockMeshDict: %d blocks", len(bmd.blocks))

    # --- Load STEP solids ---
    if args.naming == "auto":
        # Use three-path name extraction for reliable solid names.
        try:
            named_solids = load_solids_with_names(stp_path)
        except (ImportError, RuntimeError) as exc:
            logger.error("Failed to load STEP file: %s", exc)
            sys.exit(1)
        solid_label_pairs = [(ns.solid, ns.name) for ns in named_solids]
        logger.info("Loaded %d solid(s) [naming=auto]", len(named_solids))
        logger.debug(
            "Name sources: %s",
            ", ".join(ns.source for ns in named_solids),
        )
    else:
        # Legacy generic naming: zone0, zone1, ...
        try:
            raw_pairs = load_step_solids(stp_path)
        except (ImportError, RuntimeError) as exc:
            logger.error("Failed to load STEP file: %s", exc)
            sys.exit(1)
        solid_label_pairs = [
            (solid, f"zone{i}") for i, (solid, _) in enumerate(raw_pairs)
        ]
        logger.info(
            "Loaded %d solid(s) [naming=generic]",
            len(solid_label_pairs),
        )

    if args.sampling_strategy == "inset":
        logger.info(
            "Sampling: inset (factor=%g, raw vertices=8), vote: %s",
            args.inset_factor,
            args.vote_policy,
        )
    else:
        logger.info(
            "Sampling: %s (factor=N/A), vote: %s",
            args.sampling_strategy,
            args.vote_policy,
        )

    if args.no_aabb_filter:
        logger.info("AABB pre-filter disabled (diagnostic mode).")

    # --- Assign cell zones ---
    try:
        mapping = assign_cell_zones(
            bmd,
            solid_label_pairs,
            tol=args.tolerance,
            strict=args.strict,
            sampling_strategy=args.sampling_strategy,
            inset_factor=args.inset_factor,
            vote_policy=args.vote_policy,
            use_aabb_filter=not args.no_aabb_filter,
        )
    except ValueError as exc:
        logger.error("Invalid argument: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Strict mode aborted: %s", exc)
        sys.exit(2)

    # --- Write result ---
    bmd.write(bmd_path)
    logger.info(
        "Written %s (%d/%d blocks zoned)",
        bmd_path,
        len(mapping),
        len(bmd.blocks),
    )


if __name__ == "__main__":
    sys.exit(main())
