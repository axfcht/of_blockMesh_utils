"""CLI entry point for the STEP -> blockMeshDict converter.

Usage
-----
    stpToBMD [--density AX AY AZ] [--minCellCount N]
                       [--blockCount NAME N0 N1 N2 ...]
                       [--useLegacyCellCount [--legacyDensity FX FY FZ]]
                       [--origin x y z] [--tolerance TOL]
                       [--bsplineSamples N]
                       [--nameCollision {suffix,error,rename}]
                       [--cellConflict {error,warn-max,warn-first}]
                       [--strict] [--overwrite]
                       [--defaultPatchName NAME]
                       [--caseDir DIR] [--logLevel LEVEL] [--noBackup]
"""

import argparse
import logging
from pathlib import Path

from meshing_utils.cli.base import add_common_args, run_cli
from meshing_utils.operations.stp_pipeline import StpPipelineConfig, run

logger = logging.getLogger(__name__)


def _validate_density_new(values: list[float]) -> tuple[float, float, float]:
    """Validate density values for the new (default) cell-count strategy.

    Components must be non-negative; zero is allowed and effectively
    disables the contribution of that global axis to the L2 projection.
    """
    ax, ay, az = values
    for name, val in (("ax", ax), ("ay", ay), ("az", az)):
        if val < 0.0:
            raise ValueError(
                f"--density {name}={val} must be non-negative (>= 0)."
            )
    return (ax, ay, az)


def _validate_density_legacy(values: list[float]) -> tuple[float, float, float]:
    """Validate density values for the legacy fraction-of-bbox strategy.

    Components must be strictly positive. A warning is emitted for values
    greater than 1, since the legacy semantics interpret them as fractions
    of the global bounding box.
    """
    fx, fy, fz = values
    for name, val in (("fx", fx), ("fy", fy), ("fz", fz)):
        if val <= 0.0:
            raise ValueError(
                f"--density {name}={val} must be strictly positive (> 0) "
                f"in legacy mode."
            )
    for name, val in (("fx", fx), ("fy", fy), ("fz", fz)):
        if val > 1.0:
            logger.warning(
                "Legacy mode: --density %s=%s > 1 will be interpreted as a "
                "fraction of the global bounding box.",
                name,
                val,
            )
    return (fx, fy, fz)


def _parse_block_overrides(
    raw_entries: list[list[str]] | None,
) -> dict[str, tuple[int, int, int]]:
    """Parse a list of ``--blockCount NAME N0 N1 N2`` entries.

    Returns a dict preserving CLI insertion order. Raises ``ValueError`` on
    malformed counts, non-positive counts, or duplicate names.
    """
    overrides: dict[str, tuple[int, int, int]] = {}
    if not raw_entries:
        return overrides
    for entry in raw_entries:
        if len(entry) != 4:
            raise ValueError(
                f"--blockCount expects exactly 4 arguments "
                f"(NAME N0 N1 N2), got {entry!r}"
            )
        name, s0, s1, s2 = entry
        try:
            n0, n1, n2 = int(s0), int(s1), int(s2)
        except ValueError as exc:
            raise ValueError(
                f"--blockCount {name}: cell counts must be integers, "
                f"got ({s0!r}, {s1!r}, {s2!r})"
            ) from exc
        if min(n0, n1, n2) < 1:
            raise ValueError(
                f"--blockCount {name}: cell counts must be >= 1, "
                f"got ({n0}, {n1}, {n2})"
            )
        if name in overrides:
            raise ValueError(
                f"--blockCount: duplicate entry for block '{name}'."
            )
        overrides[name] = (n0, n1, n2)
    return overrides


def main() -> None:
    """Entry point for the ``stpToBMD`` CLI tool."""
    parser = argparse.ArgumentParser(
        description="Convert a STEP file to an OpenFOAM blockMeshDict."
    )
    parser.add_argument(
        "--density",
        nargs=3,
        type=float,
        metavar=("AX", "AY", "AZ"),
        default=None,
        help=(
            "DEFAULT STRATEGY ONLY. Cell density per global axis (X, Y, Z): "
            "cells per length unit; the longest of the four parallel edges per "
            "block axis is projected (L2 norm) onto this density to obtain the "
            "raw cell count. Non-negative; 0 disables that axis. "
            "Default: 1.0 1.0 1.0. Mutually exclusive with --useLegacyCellCount; "
            "use --legacyDensity for legacy fraction-of-bbox semantics."
        ),
    )
    parser.add_argument(
        "--legacyDensity",
        dest="legacy_density",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        default=None,
        help=(
            "LEGACY STRATEGY ONLY. Fraction of the global bounding-box extent "
            "to use as the target cell size per global axis. Must be strictly "
            "positive. Requires --useLegacyCellCount."
        ),
    )
    parser.add_argument(
        "--minCellCount",
        dest="min_cell_count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Global minimum cell count per axis equivalence class "
            "(default strategy only; ignored in legacy mode). "
            "Must be >= 1."
        ),
    )
    parser.add_argument(
        "--blockCount",
        dest="block_count",
        nargs=4,
        action="append",
        metavar=("NAME", "N0", "N1", "N2"),
        default=None,
        help=(
            "Per-block cell-count override (default strategy only; "
            "ignored in legacy mode). Repeatable. NAME matches the STEP "
            "solid label (XCAF label); unlabelled solids fall back to "
            "'block0', 'block1', ... in solid order. Counts apply to the "
            "block's three LOCAL axes 0, 1, 2 (not world axes). "
            "Overrides are NOT clamped to --minCellCount. "
            "Tip: for names starting with '-' use '--blockCount=NAME ...' "
            "so argparse does not treat NAME as a flag. "
            "On duplicate NAMEs within the same invocation, an error is "
            "raised. When a block name appears multiple times in the "
            "STEP file, the first matching solid is overridden."
        ),
    )
    parser.add_argument(
        "--useLegacyCellCount",
        dest="use_legacy_cell_count",
        action="store_true",
        default=False,
        help=(
            "Use the legacy PropagatedCellCountStrategy (fraction-of-bbox "
            "heuristic, supports append mode). Default: the L2 projected "
            "Euclidean strategy."
        ),
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="Origin offset (x y z) subtracted from all coordinates. Default: 0 0 0.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Snap-to-grid tolerance for vertex de-duplication. Default: 1e-6.",
    )
    parser.add_argument(
        "--bsplineSamples",
        dest="bspline_samples",
        type=int,
        default=20,
        help="Number of inner sample points for bspline edges. Default: 20.",
    )
    parser.add_argument(
        "--nameCollision",
        dest="name_collision",
        choices=["suffix", "error", "rename"],
        default="suffix",
        help="Policy for duplicate block names. Default: suffix.",
    )
    parser.add_argument(
        "--cellConflict",
        dest="cell_conflict",
        choices=["error", "warn-max", "warn-first"],
        default="warn-max",
        help=(
            "Policy for handling cell-count conflicts between pre-existing blocks "
            "in append mode (legacy strategy only). Default: warn-max."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat edge conflicts as fatal errors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing blockMeshDict (creates a .bak backup).",
    )
    parser.add_argument(
        "--convertToMeters",
        dest="convert_to_meters",
        type=float,
        default=None,
        metavar="FACTOR",
        help=(
            "Explicit convertToMeters scale factor for the blockMeshDict. "
            "If omitted: preserved from the existing blockMeshDict (overwrite "
            "or append mode) or derived from the STEP LENGTH_UNIT header for "
            "newly created files (MM -> 0.001, M -> 1.0, IN -> 0.0254, ...). "
            "Unknown units fall back to 1.0."
        ),
    )
    parser.add_argument(
        "--defaultPatchName",
        dest="default_patch_name",
        default=None,
        help=(
            "Name for the defaultPatch entry in the blockMeshDict. "
            "Default: 'defaultFaces' (OpenFOAM standard). "
            "In append mode, the existing value is preserved unless this flag is set."
        ),
    )

    # --- Common args: --caseDir, --logLevel, --noBackup ---
    add_common_args(parser)

    def _execute(args: argparse.Namespace) -> None:
        _run_pipeline(parser, args)

    run_cli(parser, _execute)


def _run_pipeline(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Translate parsed CLI args into a :class:`StpPipelineConfig` and dispatch.

    Field-level invariants live in ``StpPipelineConfig.__post_init__``;
    the CLI still owns the **UX-level** concerns: mapping the overloaded
    ``--density`` semantics to either ``fractions`` (legacy) or
    ``density`` (default), warning when legacy-incompatible flags are
    used, and re-raising ``ValueError`` from config construction as
    ``parser.error`` so the user sees an argparse-style message.
    """
    case_dir: Path = args.case_dir if args.case_dir is not None else Path.cwd()

    # --- Strict, mode-aware flag combinations ---
    if args.use_legacy_cell_count:
        if args.density is not None:
            parser.error(
                "--density is only valid in the default strategy. "
                "Use --legacyDensity with --useLegacyCellCount."
            )
        if args.min_cell_count is not None:
            parser.error(
                "--minCellCount is not supported in legacy mode "
                "(--useLegacyCellCount)."
            )
        if args.block_count:
            parser.error(
                "--blockCount is not supported in legacy mode "
                "(--useLegacyCellCount)."
            )
    elif args.legacy_density is not None:
        parser.error(
            "--legacyDensity requires --useLegacyCellCount. "
            "Use --density for the default strategy."
        )

    fractions: tuple[float, float, float] | None = None
    density: tuple[float, float, float] = (1.0, 1.0, 1.0)
    if args.use_legacy_cell_count and args.legacy_density is not None:
        try:
            fractions = _validate_density_legacy(args.legacy_density)
        except ValueError as exc:
            parser.error(str(exc))
    elif not args.use_legacy_cell_count and args.density is not None:
        try:
            density = _validate_density_new(args.density)
        except ValueError as exc:
            parser.error(str(exc))

    block_overrides: dict[str, tuple[int, int, int]] = {}
    if not args.use_legacy_cell_count:
        try:
            block_overrides = _parse_block_overrides(args.block_count)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        config = StpPipelineConfig(
            origin=tuple(args.origin),
            tol=args.tolerance,
            n_samples=args.bspline_samples,
            name_collision=args.name_collision,
            strict=args.strict,
            overwrite=args.overwrite,
            default_patch_name=args.default_patch_name or "defaultFaces",
            default_patch_name_explicit=args.default_patch_name is not None,
            fractions=fractions,
            cell_conflict=args.cell_conflict,
            use_legacy_cell_count=args.use_legacy_cell_count,
            density=density,
            min_cell_count=args.min_cell_count,
            block_overrides=block_overrides,
            convert_to_meters=args.convert_to_meters,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run(config, case_dir=case_dir)


if __name__ == "__main__":
    main()
