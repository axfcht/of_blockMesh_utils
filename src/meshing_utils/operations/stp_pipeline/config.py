"""Frozen configuration dataclass for the STEP -> blockMeshDict pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StpPipelineConfig:
    """Frozen configuration for :func:`meshing_utils.operations.stp_pipeline.run`.

    Centralises every pipeline knob previously passed as a keyword argument
    to ``run`` and enforces invariants up front in ``__post_init__`` instead
    of relying on the CLI layer to validate each field. ``case_dir`` is
    intentionally **not** part of the config — it is execution context, not
    a tunable pipeline parameter.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tol: float = 1e-6
    n_samples: int = 20
    name_collision: str = "suffix"
    strict: bool = False
    overwrite: bool = False
    default_patch_name: str = "defaultFaces"
    default_patch_name_explicit: bool = False
    fractions: tuple[float, float, float] | None = None
    cell_conflict: str = "warn-max"
    use_legacy_cell_count: bool = False
    density: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_cell_count: int | None = None
    block_overrides: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    convert_to_meters: float | None = None

    def __post_init__(self) -> None:
        if self.name_collision not in ("suffix", "error", "rename"):
            raise ValueError(
                f"name_collision must be one of suffix/error/rename, "
                f"got {self.name_collision!r}."
            )
        if self.cell_conflict not in ("error", "warn-max", "warn-first"):
            raise ValueError(
                f"cell_conflict must be one of error/warn-max/warn-first, "
                f"got {self.cell_conflict!r}."
            )
        if self.use_legacy_cell_count:
            if self.fractions is not None:
                for name, val in zip(("fx", "fy", "fz"), self.fractions, strict=False):
                    if val <= 0.0:
                        raise ValueError(
                            f"--density {name}={val} must be strictly positive (> 0) "
                            f"in legacy mode."
                        )
        else:
            for name, val in zip(("ax", "ay", "az"), self.density, strict=False):
                if val < 0.0:
                    raise ValueError(
                        f"--density {name}={val} must be non-negative (>= 0)."
                    )
            if self.min_cell_count is not None and self.min_cell_count < 1:
                raise ValueError(
                    f"--minCellCount must be >= 1, got {self.min_cell_count}."
                )
        for name, counts in self.block_overrides.items():
            if len(counts) != 3:
                raise ValueError(
                    f"block_overrides[{name!r}] must have exactly 3 cell counts, "
                    f"got {len(counts)}."
                )
            for axis, n in enumerate(counts):
                if n < 1:
                    raise ValueError(
                        f"--blockCount {name}: cell counts must be >= 1, "
                        f"got axis {axis} = {n}."
                    )
        if self.convert_to_meters is not None and self.convert_to_meters <= 0.0:
            raise ValueError(
                f"--convertToMeters must be strictly positive (> 0), "
                f"got {self.convert_to_meters}."
            )
