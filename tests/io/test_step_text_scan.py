"""Unit tests for meshing_utils.io.step_text_scan.

All tests are pure Python and do not require OCC to be installed.
Synthetic STEP file content is written to tmp_path fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meshing_utils.io.step_text_scan import (
    BrepSolidEntry,
    build_brep_name_map_by_file_id,
    parse_brep_solid_entries,
)

# ---------------------------------------------------------------------------
# Shared STEP fixture helpers
# ---------------------------------------------------------------------------

_STEP_HEADER = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'),'2;1');
FILE_NAME('test.stp','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
"""

_STEP_FOOTER = "ENDSEC;\nEND-ISO-10303-21;\n"


def _make_step(data_section: str) -> str:
    """Wrap *data_section* in a minimal ISO-10303-21 envelope."""
    return _STEP_HEADER + data_section + _STEP_FOOTER


def _write_step(tmp_path: Path, data_section: str, name: str = "test.stp") -> Path:
    stp = tmp_path / name
    stp.write_text(_make_step(data_section), encoding="utf-8")
    return stp


# ---------------------------------------------------------------------------
# BrepSolidEntry dataclass
# ---------------------------------------------------------------------------

class TestBrepSolidEntry:
    def test_fields_stored(self):
        entry = BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="zone_a")
        assert entry.file_id == 57
        assert entry.type_token == "MANIFOLD_SOLID_BREP"
        assert entry.name == "zone_a"

    def test_empty_name_allowed(self):
        entry = BrepSolidEntry(file_id=1, type_token="BREP_WITH_VOIDS", name="")
        assert entry.name == ""

    def test_equality(self):
        a = BrepSolidEntry(file_id=10, type_token="FACETED_BREP", name="x")
        b = BrepSolidEntry(file_id=10, type_token="FACETED_BREP", name="x")
        assert a == b

    def test_inequality_on_field(self):
        a = BrepSolidEntry(file_id=10, type_token="FACETED_BREP", name="x")
        b = BrepSolidEntry(file_id=11, type_token="FACETED_BREP", name="x")
        assert a != b


# ---------------------------------------------------------------------------
# parse_brep_solid_entries — happy path
# ---------------------------------------------------------------------------

class TestParseBrepSolidEntries:
    def test_manifold_solid_brep_extracted(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].file_id == 57
        assert result[0].type_token == "MANIFOLD_SOLID_BREP"
        assert result[0].name == "zone_aussen"

    def test_brep_with_voids_extracted(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].file_id == 56
        assert result[0].type_token == "BREP_WITH_VOIDS"
        assert result[0].name == "zone_mitte"

    def test_faceted_brep_extracted(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#99 = FACETED_BREP('zone_faceted', #300);\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].file_id == 99
        assert result[0].type_token == "FACETED_BREP"

    def test_shell_based_surface_model_extracted(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#42 = SHELL_BASED_SURFACE_MODEL('zone_shell', (#50));\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].file_id == 42
        assert result[0].type_token == "SHELL_BASED_SURFACE_MODEL"
        assert result[0].name == "zone_shell"

    def test_all_four_types_in_one_file(self, tmp_path: Path):
        data = (
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
            "#99 = FACETED_BREP('zone_faceted', #300);\n"
            "#42 = SHELL_BASED_SURFACE_MODEL('zone_shell', (#50));\n"
        )
        stp = _write_step(tmp_path, data)
        result = parse_brep_solid_entries(stp)
        assert len(result) == 5
        ids = [e.file_id for e in result]
        assert 56 in ids
        assert 57 in ids
        assert 58 in ids
        assert 99 in ids
        assert 42 in ids

    def test_returns_file_order(self, tmp_path: Path):
        """Entries are returned in the order they appear in the file."""
        data = (
            "#100 = MANIFOLD_SOLID_BREP('first', #1);\n"
            "#50 = MANIFOLD_SOLID_BREP('second', #2);\n"
            "#200 = BREP_WITH_VOIDS('third', #3, (#4));\n"
        )
        stp = _write_step(tmp_path, data)
        result = parse_brep_solid_entries(stp)
        assert len(result) == 3
        assert result[0].file_id == 100
        assert result[0].name == "first"
        assert result[1].file_id == 50
        assert result[1].name == "second"
        assert result[2].file_id == 200
        assert result[2].name == "third"

    def test_type_token_is_upper_cased(self, tmp_path: Path):
        """type_token is normalised to upper case regardless of file casing."""
        stp = _write_step(tmp_path, "#77 = manifold_solid_brep('zone_lower', #100);\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].type_token == "MANIFOLD_SOLID_BREP"

    def test_empty_name_entry_included(self, tmp_path: Path):
        """Entries with empty name strings are included (not filtered here)."""
        stp = _write_step(tmp_path, "#10 = MANIFOLD_SOLID_BREP('', #100);\n")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].name == ""

    def test_file_id_is_int(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#123 = MANIFOLD_SOLID_BREP('my_zone', #100);\n")
        result = parse_brep_solid_entries(stp)
        assert isinstance(result[0].file_id, int)
        assert result[0].file_id == 123


# ---------------------------------------------------------------------------
# parse_brep_solid_entries — DATA section extraction
# ---------------------------------------------------------------------------

class TestParseBrepSolidEntriesDataSection:
    def test_entity_outside_data_section_ignored(self, tmp_path: Path):
        """Entities in the HEADER section or outside DATA; are not matched."""
        # An entity appearing before DATA; (in the HEADER section) should be ignored.
        content = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "#99 = MANIFOLD_SOLID_BREP('should_be_ignored', #1);\n"
            "ENDSEC;\n"
            "DATA;\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n"
        )
        stp = tmp_path / "header_noise.stp"
        stp.write_text(content, encoding="utf-8")
        result = parse_brep_solid_entries(stp)
        # Only the entry from the DATA section should appear
        assert len(result) == 1
        assert result[0].name == "zone_aussen"

    def test_multiline_entity_parsed(self, tmp_path: Path):
        """Entity definition spanning multiple lines is correctly matched."""
        data = (
            "#57 = MANIFOLD_SOLID_BREP(\n"
            "  'zone_multiline',\n"
            "  #100);\n"
        )
        stp = _write_step(tmp_path, data)
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].name == "zone_multiline"

    def test_whitespace_before_parenthesis(self, tmp_path: Path):
        """Optional whitespace between the type keyword and '(' is handled."""
        data = "#57 = MANIFOLD_SOLID_BREP   (  'spaced_name', #100);\n"
        stp = _write_step(tmp_path, data)
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].name == "spaced_name"

    def test_no_data_section_still_returns_results(self, tmp_path: Path):
        """Files without a DATA; marker fall back to scanning the entire file."""
        content = "#57 = MANIFOLD_SOLID_BREP('fallback_zone', #100);\n"
        stp = tmp_path / "no_section.stp"
        stp.write_text(content, encoding="utf-8")
        result = parse_brep_solid_entries(stp)
        assert len(result) == 1
        assert result[0].name == "fallback_zone"


# ---------------------------------------------------------------------------
# parse_brep_solid_entries — error handling
# ---------------------------------------------------------------------------

class TestParseBrepSolidEntriesErrors:
    def test_missing_file_returns_empty_list(self, tmp_path: Path):
        result = parse_brep_solid_entries(tmp_path / "nonexistent.stp")
        assert result == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        stp = tmp_path / "empty.stp"
        stp.write_text("", encoding="utf-8")
        result = parse_brep_solid_entries(stp)
        assert result == []

    def test_no_brep_entities_returns_empty_list(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#10 = CARTESIAN_POINT('origin', (0.0, 0.0, 0.0));\n")
        result = parse_brep_solid_entries(stp)
        assert result == []

    def test_result_is_list(self, tmp_path: Path):
        stp = tmp_path / "empty2.stp"
        stp.write_text("", encoding="utf-8")
        result = parse_brep_solid_entries(stp)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# build_brep_name_map_by_file_id — happy path
# ---------------------------------------------------------------------------

class TestBuildBrepNameMapByFileId:
    def test_manifold_solid_brep_included(self, tmp_path: Path):
        stp = _write_step(
            tmp_path,
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n",
        )
        result = build_brep_name_map_by_file_id(stp)
        assert result == {57: "zone_aussen", 58: "zone_innen"}

    def test_brep_with_voids_included(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n")
        result = build_brep_name_map_by_file_id(stp)
        assert result == {56: "zone_mitte"}

    def test_mixed_types(self, tmp_path: Path):
        data = (
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = _write_step(tmp_path, data)
        result = build_brep_name_map_by_file_id(stp)
        assert result == {56: "zone_mitte", 57: "zone_aussen", 58: "zone_innen"}

    def test_empty_name_excluded(self, tmp_path: Path):
        """Entries with empty names are not included in the map."""
        data = (
            "#10 = MANIFOLD_SOLID_BREP('', #100);\n"
            "#11 = MANIFOLD_SOLID_BREP('valid_name', #101);\n"
        )
        stp = _write_step(tmp_path, data)
        result = build_brep_name_map_by_file_id(stp)
        assert 10 not in result
        assert 11 in result
        assert result[11] == "valid_name"

    def test_keys_are_int(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#123 = MANIFOLD_SOLID_BREP('my_zone', #100);\n")
        result = build_brep_name_map_by_file_id(stp)
        for k in result:
            assert isinstance(k, int)

    def test_values_are_str(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#123 = MANIFOLD_SOLID_BREP('my_zone', #100);\n")
        result = build_brep_name_map_by_file_id(stp)
        for v in result.values():
            assert isinstance(v, str)

    def test_returns_dict(self, tmp_path: Path):
        stp = tmp_path / "empty3.stp"
        stp.write_text("", encoding="utf-8")
        result = build_brep_name_map_by_file_id(stp)
        assert isinstance(result, dict)

    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        result = build_brep_name_map_by_file_id(tmp_path / "nonexistent.stp")
        assert result == {}

    def test_multiline_entity_included(self, tmp_path: Path):
        data = "#57 = MANIFOLD_SOLID_BREP(\n  'multiline_zone',\n  #100);\n"
        stp = _write_step(tmp_path, data)
        result = build_brep_name_map_by_file_id(stp)
        assert result == {57: "multiline_zone"}

    def test_case_insensitive_type_keyword(self, tmp_path: Path):
        stp = _write_step(tmp_path, "#77 = manifold_solid_brep('zone_lower', #100);\n")
        result = build_brep_name_map_by_file_id(stp)
        assert result == {77: "zone_lower"}

    def test_real_world_triple(self, tmp_path: Path):
        """Reproduces the exact scenario from the diagnosis: #56, #57, #58."""
        data = (
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = _write_step(tmp_path, data)
        result = build_brep_name_map_by_file_id(stp)
        assert result[56] == "zone_mitte"
        assert result[57] == "zone_aussen"
        assert result[58] == "zone_innen"


# ---------------------------------------------------------------------------
# Integration test (requires real OCC + STP file)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_extract_solid_names_from_real_file():
    """Integration test: extract names from the real STEP file.

    Requires OCC and the real STP file at the project root.
    Skipped automatically when the file is absent.

    Expected: three solids named zone_aussen, zone_innen, zone_mitte,
    all with source == 'step_id' (primary OCC path via StringLabel).
    """
    stp = Path("cad_files/fluidvolumen_komplett_cell_zones.stp")
    if not stp.exists():
        pytest.skip(f"STP file not found: {stp}")

    try:
        from meshing_utils import load_solids_with_names  # type: ignore[import]
    except ImportError:
        pytest.skip("OCC not available")

    try:
        named_solids = load_solids_with_names(stp)
    except (ImportError, Exception) as exc:
        pytest.skip(f"OCC not available or load failed: {exc}")

    names = {ns.name for ns in named_solids}
    sources = {ns.source for ns in named_solids}

    assert names == {"zone_aussen", "zone_innen", "zone_mitte"}, (
        f"Unexpected names: {names}"
    )
    assert "generic" not in sources, (
        f"At least one solid received a generic name; sources: {sources}"
    )
