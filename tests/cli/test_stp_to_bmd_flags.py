"""CLI flag and validator tests for meshing_utils.cli.stp_to_bmd.

Focus: argparse validation, density semantics by mode, block-count parsing.
The pipeline ``run`` is patched out so STEP files / OCP are not required.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest

from meshing_utils.cli.stp_to_bmd import (
    _parse_block_overrides,
    _validate_density_legacy,
    _validate_density_new,
    main,
)

# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def test_validate_density_new_accepts_zero() -> None:
    assert _validate_density_new([0.0, 1.0, 2.5]) == (0.0, 1.0, 2.5)


def test_validate_density_new_rejects_negative() -> None:
    with pytest.raises(ValueError):
        _validate_density_new([-0.1, 1.0, 1.0])


def test_validate_density_legacy_rejects_zero() -> None:
    with pytest.raises(ValueError):
        _validate_density_legacy([0.0, 0.5, 0.5])


def test_validate_density_legacy_warns_above_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="meshing_utils.cli.stp_to_bmd")
    _validate_density_legacy([1.5, 0.5, 0.5])
    assert any("> 1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Block-overrides parser
# ---------------------------------------------------------------------------


def test_parse_block_overrides_none() -> None:
    assert _parse_block_overrides(None) == {}


def test_parse_block_overrides_happy() -> None:
    result = _parse_block_overrides([["a", "1", "2", "3"], ["b", "4", "5", "6"]])
    assert result == {"a": (1, 2, 3), "b": (4, 5, 6)}


def test_parse_block_overrides_preserves_order() -> None:
    result = _parse_block_overrides([["z", "1", "1", "1"], ["a", "2", "2", "2"]])
    assert list(result.keys()) == ["z", "a"]


def test_parse_block_overrides_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _parse_block_overrides([["a", "1", "1", "1"], ["a", "2", "2", "2"]])


def test_parse_block_overrides_non_integer_raises() -> None:
    with pytest.raises(ValueError, match="must be integers"):
        _parse_block_overrides([["a", "1.5", "2", "3"]])


def test_parse_block_overrides_below_one_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _parse_block_overrides([["a", "0", "1", "1"]])


# ---------------------------------------------------------------------------
# End-to-end CLI argument dispatching
# ---------------------------------------------------------------------------


def _invoke(argv: list[str]) -> dict:
    """Invoke main() with patched run() and return captured config fields.

    The flattened mapping mirrors the legacy kwargs-bag so existing
    test assertions like ``kwargs["density"]`` keep working after the
    Phase 2.3 ``StpPipelineConfig`` migration.
    """
    import dataclasses

    captured: dict = {}

    def fake_run(config, *, case_dir):
        captured["case_dir"] = case_dir
        for f in dataclasses.fields(config):
            captured[f.name] = getattr(config, f.name)

    with (
        patch("meshing_utils.cli.stp_to_bmd.run", side_effect=fake_run),
        patch.object(sys, "argv", ["stpToBMD", *argv]),
    ):
        main()
    return captured


def test_default_strategy_density_default(tmp_path) -> None:
    kwargs = _invoke(["--caseDir", str(tmp_path)])
    assert kwargs["use_legacy_cell_count"] is False
    assert kwargs["density"] == (1.0, 1.0, 1.0)
    assert kwargs["min_cell_count"] is None
    assert kwargs["block_overrides"] == {}
    assert kwargs["fractions"] is None


def test_default_strategy_with_density(tmp_path) -> None:
    kwargs = _invoke(
        ["--caseDir", str(tmp_path), "--density", "2", "3", "4"]
    )
    assert kwargs["density"] == (2.0, 3.0, 4.0)


def test_legacy_flag_routes_legacy_density_to_fractions(tmp_path) -> None:
    kwargs = _invoke(
        [
            "--caseDir",
            str(tmp_path),
            "--useLegacyCellCount",
            "--legacyDensity",
            "0.1",
            "0.2",
            "0.3",
        ]
    )
    assert kwargs["use_legacy_cell_count"] is True
    assert kwargs["fractions"] == (0.1, 0.2, 0.3)


def test_density_in_legacy_mode_errors(tmp_path) -> None:
    """--density + --useLegacyCellCount is now a hard parser error."""
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--useLegacyCellCount",
                "--density",
                "0.1",
                "0.2",
                "0.3",
            ]
        )


def test_legacy_density_without_legacy_flag_errors(tmp_path) -> None:
    """--legacyDensity requires --useLegacyCellCount."""
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--legacyDensity",
                "0.1",
                "0.2",
                "0.3",
            ]
        )


def test_legacy_rejects_min_cell_count(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--useLegacyCellCount",
                "--minCellCount",
                "5",
            ]
        )


def test_legacy_rejects_block_count(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--useLegacyCellCount",
                "--blockCount",
                "a",
                "1",
                "1",
                "1",
            ]
        )


def test_negative_density_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(["--caseDir", str(tmp_path), "--density", "-1", "1", "1"])


def test_legacy_zero_density_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--useLegacyCellCount",
                "--legacyDensity",
                "0",
                "0.5",
                "0.5",
            ]
        )


def test_block_count_collected(tmp_path) -> None:
    kwargs = _invoke(
        [
            "--caseDir",
            str(tmp_path),
            "--blockCount",
            "first",
            "2",
            "3",
            "4",
            "--blockCount",
            "second",
            "5",
            "6",
            "7",
        ]
    )
    assert kwargs["block_overrides"] == {
        "first": (2, 3, 4),
        "second": (5, 6, 7),
    }


def test_duplicate_block_count_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--blockCount",
                "a",
                "1",
                "1",
                "1",
                "--blockCount",
                "a",
                "2",
                "2",
                "2",
            ]
        )


def test_block_count_zero_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(
            [
                "--caseDir",
                str(tmp_path),
                "--blockCount",
                "a",
                "0",
                "1",
                "1",
            ]
        )


def test_min_cell_count_below_one_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(["--caseDir", str(tmp_path), "--minCellCount", "0"])


def test_min_cell_count_propagates(tmp_path) -> None:
    kwargs = _invoke(["--caseDir", str(tmp_path), "--minCellCount", "7"])
    assert kwargs["min_cell_count"] == 7
