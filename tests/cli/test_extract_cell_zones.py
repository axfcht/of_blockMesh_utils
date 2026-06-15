"""Unit tests for meshing_utils.cli.extract_cell_zones.

OCC-dependent integration tests are skipped when OCC is not installed.
All tests that exercise actual zone assignment mock out load_solids_with_names
(or load_step_solids for --naming generic) and classify_point_in_solid, so no
OCC dependency is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from meshing_utils.cad.step_names import NamedSolid
from meshing_utils.cli.extract_cell_zones import main
from meshing_utils.geometry.containment import STATE_IN, STATE_OUT

# ---------------------------------------------------------------------------
# Minimal blockMeshDict fixture
# ---------------------------------------------------------------------------

_MINIMAL_BMD = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
\tformat      ascii;
\tclass       dictionary;
\tobject      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

geometry
{
}

vertices
(
\tname v0 (0.00000000 0.00000000 0.00000000)
\tname v1 (1.00000000 0.00000000 0.00000000)
\tname v2 (1.00000000 1.00000000 0.00000000)
\tname v3 (0.00000000 1.00000000 0.00000000)
\tname v4 (0.00000000 0.00000000 1.00000000)
\tname v5 (1.00000000 0.00000000 1.00000000)
\tname v6 (1.00000000 1.00000000 1.00000000)
\tname v7 (0.00000000 1.00000000 1.00000000)
);

edges
(
);

blocks
(
\tname b0 hex (v0 v1 v2 v3 v4 v5 v6 v7) (1 1 1) simpleGrading (1 1 1)
);

defaultPatch
{
\tname defaultFaces;
\ttype empty;
}

boundary
(
);

// ************************************************************************* //
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSolid:
    """Sentinel solid object â€” no OCC needed."""
    def __init__(self, name: str = ""):
        self._name = name


def _write_bmd(case_dir: Path, content: str = _MINIMAL_BMD) -> Path:
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    bmd_path = system_dir / "blockMeshDict"
    bmd_path.write_text(content, encoding="utf-8")
    return bmd_path


def _write_stp(directory: Path, filename: str = "model.stp") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stp = directory / filename
    stp.write_text("dummy STEP content", encoding="utf-8")
    return stp


def _make_fake_solids(names: list[str]) -> list[tuple]:
    """Legacy helper: returns (solid, name) tuple list for --naming generic tests."""
    return [(_FakeSolid(n), n) for n in names]


def _make_named_solids(names: list[str]) -> list[NamedSolid]:
    """Return NamedSolid list for --naming auto tests (default path)."""
    return [NamedSolid(solid=_FakeSolid(n), name=n, source="assembly") for n in names]


# ---------------------------------------------------------------------------
# Happy-path test
# ---------------------------------------------------------------------------

class TestCLIHappyPath:

    def test_cli_happy_path(self, tmp_path: Path):
        """Happy path: single STP in geometry/, block gets zoned, exit 0, .bak created."""
        _write_bmd(tmp_path)
        geometry_dir = tmp_path / "constant" / "geometry"
        _write_stp(geometry_dir, "foo.stp")

        solid = _FakeSolid("fluid")
        fake_named = _make_named_solids(["fluid"])
        fake_named[0] = NamedSolid(solid=solid, name="fluid", source="assembly")

        # The centroid of the unit cube is (0.5, 0.5, 0.5)
        def _fake_classify(s, point, tol):
            return STATE_IN if s is solid else STATE_OUT

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
        ]):
            result = main()

        assert result == 0

        # .bak must exist
        bak = tmp_path / "system" / "blockMeshDict.bak"
        assert bak.exists()

        # Written blockMeshDict must contain the zone token
        bmd_text = (tmp_path / "system" / "blockMeshDict").read_text(encoding="utf-8")
        assert "fluid" in bmd_text

    def test_cli_explicit_stp_path(self, tmp_path: Path):
        """--stp-path pointing outside geometry/ must be used directly."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "my_model.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        solid = _FakeSolid("airRegion")
        fake_named = [NamedSolid(solid=solid, name="airRegion", source="assembly")]

        def _fake_classify(s, point, tol):
            return STATE_IN

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
        ]):
            result = main()

        assert result == 0


# ---------------------------------------------------------------------------
# Error-handling tests (no OCC needed)
# ---------------------------------------------------------------------------

class TestCLIErrors:

    def test_cli_stp_path_missing(self, tmp_path: Path):
        """Non-existent --stp-path must exit 1."""
        _write_bmd(tmp_path)
        nonexistent = tmp_path / "missing.stp"

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(nonexistent),
        ]):
            main()
        assert exc_info.value.code == 1

    def test_cli_no_step_in_geometry(self, tmp_path: Path):
        """Empty geometry dir â†' exit 1."""
        _write_bmd(tmp_path)
        (tmp_path / "constant" / "geometry").mkdir(parents=True, exist_ok=True)

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
        ]):
            main()
        assert exc_info.value.code == 1

    def test_cli_multiple_step_in_geometry(self, tmp_path: Path):
        """Multiple STEP files without --stp-path â†' exit 1."""
        _write_bmd(tmp_path)
        geometry_dir = tmp_path / "constant" / "geometry"
        _write_stp(geometry_dir, "first.stp")
        _write_stp(geometry_dir, "second.stp")

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
        ]):
            main()
        assert exc_info.value.code == 1

    def test_cli_strict_ambiguous_exits_2(self, tmp_path: Path):
        """--strict with ambiguous assignment â†' exit 2.

        Both solids return IN for 5/9 samples (tie), ensuring no early-exit,
        which requires in_count < total (9) for the first solid.
        """
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        solid_a = _FakeSolid("A")
        solid_b = _FakeSolid("B")
        fake_named = [
            NamedSolid(solid=solid_a, name="fluid", source="assembly"),
            NamedSolid(solid=solid_b, name="solid", source="assembly"),
        ]

        # Both solids return IN for exactly 5 of 9 samples â†' tie, ambiguous
        call_counts: dict = {}

        def _fake_classify(s, point, tol):
            key = id(s)
            call_counts[key] = call_counts.get(key, 0) + 1
            return STATE_IN if call_counts[key] <= 5 else STATE_OUT

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--strict",
        ]):
            main()
        assert exc_info.value.code == 2

    def test_cli_strict_no_match_succeeds(self, tmp_path: Path):
        """--strict with no match â†' exit 0, block stays unzoned."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        solid = _FakeSolid("fluid")
        fake_named = [NamedSolid(solid=solid, name="fluid", source="assembly")]

        # No match: OUT for everything
        def _fake_classify(s, point, tol):
            return STATE_OUT

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--strict",
        ]):
            result = main()

        assert result == 0

        # Block must remain unzoned (no zone token in output)
        bmd_text = (tmp_path / "system" / "blockMeshDict").read_text(encoding="utf-8")
        # Ensure the zone name "fluid" does not appear inside the blocks section
        # We check the blocks section only (simple heuristic: count occurrences
        # of "fluid" in the blocks() section â€” should be zero).
        blocks_start = bmd_text.find("blocks")
        blocks_end = bmd_text.find(");", blocks_start)
        blocks_section = bmd_text[blocks_start:blocks_end]
        assert "fluid" not in blocks_section


# ---------------------------------------------------------------------------
# Option pass-through tests
# ---------------------------------------------------------------------------

class TestCLIOptions:

    def test_cli_no_backup_flag(self, tmp_path: Path):
        """--noBackup â†' .bak must not be created."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        fake_named: list = []

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--noBackup",
        ]):
            main()

        bak = tmp_path / "system" / "blockMeshDict.bak"
        assert not bak.exists()

    def test_cli_tolerance_passed_through(self, tmp_path: Path):
        """--tolerance 1e-5 must be forwarded to assign_cell_zones."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}

        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["tol"] = tol
            captured["strict"] = strict
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--tolerance", "1e-5",
        ]):
            main()

        assert captured.get("tol") == pytest.approx(1e-5)

    def test_cli_log_level_debug(self, tmp_path: Path, caplog):
        """--logLevel DEBUG must be accepted without error."""
        _write_bmd(tmp_path)
        (tmp_path / "constant" / "geometry").mkdir(parents=True, exist_ok=True)

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--logLevel", "DEBUG",
        ]):
            main()
        # Exits 1 because no STP found â€” but must not crash on the log-level itself
        assert exc_info.value.code == 1

    def test_cli_sampling_strategy_default_is_inset(self, tmp_path: Path):
        """Default sampling strategy must be 'inset' with factor=0.5 and majority vote."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["sampling_strategy"] = sampling_strategy
            captured["inset_factor"] = inset_factor
            captured["vote_policy"] = vote_policy
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
        ]):
            main()

        assert captured.get("sampling_strategy") == "inset"
        assert captured.get("inset_factor") == pytest.approx(0.5)
        assert captured.get("vote_policy") == "majority"

    def test_cli_sampling_strategy_centroid_passthrough(self, tmp_path: Path):
        """--sampling-strategy centroid must be forwarded."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["sampling_strategy"] = sampling_strategy
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--samplingStrategy", "centroid",
        ]):
            main()

        assert captured.get("sampling_strategy") == "centroid"

    def test_cli_inset_factor_passthrough(self, tmp_path: Path):
        """--inset-factor 0.3 must be forwarded to assign_cell_zones."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["inset_factor"] = inset_factor
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--insetFactor", "0.3",
        ]):
            main()

        assert captured.get("inset_factor") == pytest.approx(0.3)

    def test_cli_inset_factor_invalid_value(self, tmp_path: Path):
        """--inset-factor 1.5 triggers ValueError â†' exit 1."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        fake_named: list = []

        def _fake_classify(s, point, tol):
            return STATE_OUT

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--insetFactor", "1.5",
        ]):
            main()
        assert exc_info.value.code == 1

    def test_cli_vote_policy_invalid_choice(self, tmp_path: Path):
        """--vote-policy random â†' argparse rejects with exit 2."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--votePolicy", "random",
        ]):
            main()
        assert exc_info.value.code == 2

    def test_cli_invalid_sampling_strategy(self, tmp_path: Path):
        """--sampling-strategy foo â†' argparse rejects with exit 2."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--samplingStrategy", "foo",
        ]):
            main()
        assert exc_info.value.code == 2

    def test_cli_defaults(self, tmp_path: Path):
        """Without flags: tol=1e-7, strict=False, .bak is created."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["tol"] = tol
            captured["strict"] = strict
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
        ]):
            main()

        assert captured.get("tol") == pytest.approx(1e-7)
        assert captured.get("strict") is False

        bak = tmp_path / "system" / "blockMeshDict.bak"
        assert bak.exists()


# ---------------------------------------------------------------------------
# TestNoAabbFilterFlag
# ---------------------------------------------------------------------------

class TestNoAabbFilterFlag:
    """Tests for the --no-aabb-filter CLI flag."""

    def test_default_invocation_enables_aabb_filter(self, tmp_path: Path):
        """Without --no-aabb-filter, use_aabb_filter=True must be passed."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["use_aabb_filter"] = use_aabb_filter
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
        ]):
            main()

        assert captured.get("use_aabb_filter") is True

    def test_no_aabb_filter_flag_disables_filter(self, tmp_path: Path):
        """With --no-aabb-filter, use_aabb_filter=False must be passed."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        captured: dict = {}
        fake_named: list = []

        def _fake_assign(
            bmd, pairs, *, tol, strict, epsilon=None,
            sampling_strategy="inset", inset_factor=0.5, vote_policy="majority",
            use_aabb_filter=True,
        ):
            captured["use_aabb_filter"] = use_aabb_filter
            return {}

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ), patch(
            "meshing_utils.cli.extract_cell_zones.assign_cell_zones",
            side_effect=_fake_assign,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--noAabbFilter",
        ]):
            main()

        assert captured.get("use_aabb_filter") is False

    def test_no_aabb_filter_help_text_present(self, tmp_path: Path):
        """--help output must contain the --no-aabb-filter flag description."""
        import contextlib
        import io

        help_output = io.StringIO()
        with (
            pytest.raises(SystemExit) as exc_info,
            contextlib.redirect_stdout(help_output),
            patch.object(sys, "argv", ["extractCZones", "--help"]),
        ):
            main()
        # argparse exits with 0 for --help
        assert exc_info.value.code == 0
        help_text = help_output.getvalue()
        assert "--noAabbFilter" in help_text


# ---------------------------------------------------------------------------
# --naming flag tests
# ---------------------------------------------------------------------------

class TestNamingFlag:
    """Tests for the --naming {auto,generic} CLI flag."""

    def test_naming_auto_uses_load_solids_with_names(self, tmp_path: Path):
        """--naming auto (default) must call load_solids_with_names and use real names."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        solid = _FakeSolid("zone_aussen")
        fake_named = [NamedSolid(solid=solid, name="zone_aussen", source="assembly")]

        def _fake_classify(s, point, tol):
            return STATE_IN

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_solids_with_names",
            return_value=fake_named,
        ) as mock_lswn, patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--naming", "auto",
        ]):
            result = main()

        assert result == 0
        mock_lswn.assert_called_once()
        bmd_text = (tmp_path / "system" / "blockMeshDict").read_text(encoding="utf-8")
        assert "zone_aussen" in bmd_text

    def test_naming_generic_uses_zone_i_names(self, tmp_path: Path):
        """--naming generic must produce zone0, zone1, ... names from load_step_solids."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        solid = _FakeSolid("whatever_occ_name")
        # load_step_solids returns (solid, label) pairs
        fake_pairs = [(solid, "whatever_occ_name")]

        def _fake_classify(s, point, tol):
            return STATE_IN

        with patch(
            "meshing_utils.cli.extract_cell_zones.load_step_solids",
            return_value=fake_pairs,
        ) as mock_lss, patch(
            "meshing_utils.operations.cell_zones.classification.classify_point_in_solid",
            side_effect=_fake_classify,
        ), patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--naming", "generic",
        ]):
            result = main()

        assert result == 0
        mock_lss.assert_called_once()
        bmd_text = (tmp_path / "system" / "blockMeshDict").read_text(encoding="utf-8")
        # With --naming generic, the block should be named "zone0"
        assert "zone0" in bmd_text

    def test_naming_invalid_choice_exits_2(self, tmp_path: Path):
        """--naming invalid_choice â†' argparse rejects with exit 2."""
        _write_bmd(tmp_path)
        stp_path = tmp_path / "m.stp"
        stp_path.write_text("dummy", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", [
            "extractCZones",
            "--caseDir", str(tmp_path),
            "--stpPath", str(stp_path),
            "--naming", "invalid",
        ]):
            main()
        assert exc_info.value.code == 2

    def test_naming_flag_in_help_text(self):
        """--help output must contain the --naming flag description."""
        import contextlib
        import io

        help_output = io.StringIO()
        with (
            pytest.raises(SystemExit),
            contextlib.redirect_stdout(help_output),
            patch.object(sys, "argv", ["extractCZones", "--help"]),
        ):
            main()
        assert "--naming" in help_output.getvalue()


# ---------------------------------------------------------------------------
# Optional integration test (skipped when file not present)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration tests that require the real STEP fixture file."""

    _STP_PATH = (
        Path(__file__).parent.parent.parent
        / "cad_files"
        / "fluidvolumen_komplett_cell_zones.stp"
    )

    def test_extract_expected_zone_names(self):
        """Real STEP file must yield zone_aussen, zone_innen, zone_mitte."""
        pytest.importorskip("OCP")
        if not self._STP_PATH.exists():
            pytest.skip(f"STEP fixture not found: {self._STP_PATH}")

        from meshing_utils.cad.step_loader import load_solids_with_names

        named_solids = load_solids_with_names(self._STP_PATH)
        names = {ns.name for ns in named_solids}
        expected = {"zone_aussen", "zone_innen", "zone_mitte"}
        assert expected <= names, (
            f"Expected names {expected} to be a subset of found names {names}"
        )
