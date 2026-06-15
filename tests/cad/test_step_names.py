"""Unit tests for meshing_utils.cad.step_names.

All tests are pure-Python and do not require OCC to be installed.
OCC-dependent paths are exercised via MagicMock.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meshing_utils.cad.step_names import (
    _build_assembly_map,
    _build_brep_name_map,
    _dedupe,
    _diagnose_occ_mapping,
    _extract_via_entity_name,
    _extract_via_step_id,
    _get_external_file_id_from_internal,
    _is_unusable,
    _map_solids_via_model_iteration,
    _ordered_match_by_type,
    _read_entity_name,
    _sanitize,
    _strip_occurrence_index,
    extract_solid_names,
)
from meshing_utils.io.step_text_scan import BrepSolidEntry

# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_simple_ascii(self):
        assert _sanitize("zone_aussen") == "zone_aussen"

    def test_spaces_replaced(self):
        assert _sanitize("my zone") == "my_zone"

    def test_special_characters_replaced(self):
        # trailing underscore from "!" is stripped by the strip("_") step
        assert _sanitize("fluid-region!") == "fluid_region"

    def test_leading_digit_gets_prefix(self):
        result = _sanitize("1solid")
        assert result == "z_1solid"

    def test_leading_underscore_stripped(self):
        result = _sanitize("_mysolid")
        assert result == "mysolid"

    def test_trailing_underscore_stripped(self):
        result = _sanitize("mysolid_")
        assert result == "mysolid"

    def test_empty_string_returns_solid(self):
        assert _sanitize("") == "solid"

    def test_only_special_chars_returns_solid(self):
        assert _sanitize("!@#$%") == "solid"

    def test_unicode_replaced(self):
        # "Flüssigkeit": ü is replaced by _, consecutive underscores collapsed,
        # then the result is "Fl_ssigkeit" (ü → single _)
        result = _sanitize("Flüssigkeit")
        assert "ssigkeit" in result
        assert "F" in result
        assert "l" in result

    def test_consecutive_underscores_collapsed(self):
        result = _sanitize("a  b  c")
        assert result == "a_b_c"

    def test_mixed_case_preserved(self):
        assert _sanitize("ZoneInnen") == "ZoneInnen"

    def test_digits_in_middle_allowed(self):
        assert _sanitize("zone1a") == "zone1a"

    def test_whitespace_only_returns_solid(self):
        assert _sanitize("   ") == "solid"

    def test_leading_digit_after_cleanup(self):
        # "!1abc" → clean → "1abc" → needs prefix
        result = _sanitize("!1abc")
        assert result == "z_1abc"


# ---------------------------------------------------------------------------
# _is_unusable
# ---------------------------------------------------------------------------

class TestIsUnusable:
    def test_none_is_unusable(self):
        assert _is_unusable(None) is True

    def test_empty_string_is_unusable(self):
        assert _is_unusable("") is True

    def test_whitespace_only_is_unusable(self):
        assert _is_unusable("   ") is True

    def test_none_string_is_unusable(self):
        assert _is_unusable("NONE") is True

    def test_none_lowercase_is_unusable(self):
        assert _is_unusable("none") is True

    def test_part_is_unusable(self):
        assert _is_unusable("Part") is True

    def test_solid_is_unusable(self):
        assert _is_unusable("Solid") is True

    def test_open_cascade_is_unusable(self):
        assert _is_unusable("Open CASCADE STEP processor") is True

    def test_open_cascade_case_insensitive(self):
        assert _is_unusable("open cascade something") is True

    def test_valid_name_not_unusable(self):
        assert _is_unusable("zone_aussen") is False

    def test_valid_name_with_digits(self):
        assert _is_unusable("zone1") is False

    def test_valid_name_mixed_case(self):
        assert _is_unusable("FluidRegion") is False

    def test_solid_prefix_allowed(self):
        # "Solid1" is NOT in the unusable set (only exact "Solid" is filtered)
        assert _is_unusable("Solid1") is False


# ---------------------------------------------------------------------------
# _dedupe
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_no_duplicates_unchanged(self):
        names = ["a", "b", "c"]
        assert _dedupe(names) == ["a", "b", "c"]

    def test_single_duplicate_gets_suffix(self):
        result = _dedupe(["zone", "zone"])
        assert result == ["zone", "zone_1"]

    def test_triple_duplicate(self):
        result = _dedupe(["x", "x", "x"])
        assert result == ["x", "x_1", "x_2"]

    def test_non_adjacent_duplicates(self):
        result = _dedupe(["a", "b", "a"])
        assert result == ["a", "b", "a_1"]

    def test_empty_list(self):
        assert _dedupe([]) == []

    def test_single_element(self):
        assert _dedupe(["zone"]) == ["zone"]

    def test_duplicate_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="meshing_utils.cad.step_names"):
            _dedupe(["zone", "zone"])
        assert any("zone" in rec.message for rec in caplog.records)

    def test_original_order_preserved(self):
        names = ["c", "a", "b", "a", "c"]
        result = _dedupe(names)
        assert result[0] == "c"
        assert result[1] == "a"
        assert result[2] == "b"
        assert result[3] == "a_1"
        assert result[4] == "c_1"


# ---------------------------------------------------------------------------
# _strip_occurrence_index
# ---------------------------------------------------------------------------

class TestStripOccurrenceIndex:
    def test_single_occurrence_stripped(self):
        assert _strip_occurrence_index("Aussenring:1") == "Aussenring"

    def test_double_digit_occurrence_stripped(self):
        assert _strip_occurrence_index("zyl_1:14") == "zyl_1"

    def test_name_without_index_unchanged(self):
        assert _strip_occurrence_index("hex_1") == "hex_1"

    def test_underscore_name_unchanged(self):
        assert _strip_occurrence_index("zone_aussen") == "zone_aussen"

    def test_only_trailing_index_stripped(self):
        # An interior colon is not an occurrence index and stays put.
        assert _strip_occurrence_index("a:1:2") == "a:1"

    def test_empty_string(self):
        assert _strip_occurrence_index("") == ""


# ---------------------------------------------------------------------------
# _read_entity_name / _extract_via_entity_name  (Path A0)
# ---------------------------------------------------------------------------

class TestReadEntityName:
    def test_to_cstring_handle(self):
        entity = MagicMock()
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = "hex_1"
        entity.Name.return_value = handle
        assert _read_entity_name(entity) == "hex_1"

    def test_plain_str_name(self):
        entity = MagicMock()
        entity.Name.return_value = "Aussenring:1"
        assert _read_entity_name(entity) == "Aussenring:1"

    def test_none_entity_returns_none(self):
        assert _read_entity_name(None) is None

    def test_entity_without_name_method_returns_none(self):
        entity = MagicMock(spec=[])  # no Name attribute
        assert _read_entity_name(entity) is None

    def test_null_handle_returns_none(self):
        entity = MagicMock()
        handle = MagicMock()
        handle.IsNull.return_value = True
        entity.Name.return_value = handle
        assert _read_entity_name(entity) is None

    def test_empty_name_returns_none(self):
        entity = MagicMock()
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = "   "
        entity.Name.return_value = handle
        assert _read_entity_name(entity) is None

    def test_name_raises_returns_none(self):
        entity = MagicMock()
        entity.Name.side_effect = RuntimeError("OCC error")
        assert _read_entity_name(entity) is None


class TestExtractViaEntityName:
    def _entity(self, name_str: str):
        entity = MagicMock()
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = name_str
        entity.Name.return_value = handle
        return entity

    def test_per_solid_names_in_order(self):
        solids = [_FakeSolid("s0"), _FakeSolid("s1"), _FakeSolid("s2")]
        entities = {
            id(solids[0]): self._entity("hex_1"),
            id(solids[1]): self._entity("hex_2"),
            id(solids[2]): self._entity("hex_3"),
        }
        transfer_reader = MagicMock()
        transfer_reader.EntityFromShapeResult.side_effect = (
            lambda shape, mode=1: entities.get(id(shape))
        )

        result = _extract_via_entity_name(transfer_reader, solids)
        assert result == ["hex_1", "hex_2", "hex_3"]

    def test_missing_entity_yields_none(self):
        solids = [_FakeSolid("s0"), _FakeSolid("s1")]
        entities = {id(solids[0]): self._entity("hex_1")}
        transfer_reader = MagicMock()
        transfer_reader.EntityFromShapeResult.side_effect = (
            lambda shape, mode=1: entities.get(id(shape))
        )

        result = _extract_via_entity_name(transfer_reader, solids)
        assert result == ["hex_1", None]

    def test_entity_from_shape_result_raises_yields_none(self):
        solids = [_FakeSolid("s0")]
        transfer_reader = MagicMock()
        transfer_reader.EntityFromShapeResult.side_effect = RuntimeError("boom")

        result = _extract_via_entity_name(transfer_reader, solids)
        assert result == [None]


# ---------------------------------------------------------------------------
# _build_brep_name_map  (synthetic STEP strings via tmp_path)
# ---------------------------------------------------------------------------

# Minimal ISO-10303-21 header used in synthetic STEP test fixtures.
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
    return _STEP_HEADER + data_section + _STEP_FOOTER


class TestBuildBrepNameMap:
    def test_manifold_solid_brep_extracted(self, tmp_path: Path):
        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {57: "zone_aussen", 58: "zone_innen"}

    def test_brep_with_voids_extracted(self, tmp_path: Path):
        content = _make_step(
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {56: "zone_mitte"}

    def test_faceted_brep_extracted(self, tmp_path: Path):
        content = _make_step(
            "#99 = FACETED_BREP('zone_faceted', #300);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {99: "zone_faceted"}

    def test_shell_based_surface_model_extracted(self, tmp_path: Path):
        content = _make_step(
            "#42 = SHELL_BASED_SURFACE_MODEL('zone_shell', (#50));\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {42: "zone_shell"}

    def test_mixed_brep_types(self, tmp_path: Path):
        content = _make_step(
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {56: "zone_mitte", 57: "zone_aussen", 58: "zone_innen"}

    def test_empty_name_excluded(self, tmp_path: Path):
        content = _make_step(
            "#10 = MANIFOLD_SOLID_BREP('', #100);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert 10 not in result

    def test_case_insensitive_matching(self, tmp_path: Path):
        content = _make_step(
            "#77 = manifold_solid_brep('zone_lower', #100);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert result == {77: "zone_lower"}

    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        result = _build_brep_name_map(tmp_path / "nonexistent.stp")
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path: Path):
        stp = tmp_path / "empty.stp"
        stp.write_text("", encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_returns_int_keys(self, tmp_path: Path):
        content = _make_step(
            "#123 = MANIFOLD_SOLID_BREP('my_zone', #100);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_brep_name_map(stp)
        assert isinstance(next(iter(result.keys())), int)


# ---------------------------------------------------------------------------
# _build_assembly_map  (synthetic STEP strings via tmp_path)
# ---------------------------------------------------------------------------

class TestBuildAssemblyMap:
    def test_extracts_product_names(self, tmp_path: Path):
        content = _make_step(
            "#10 = PRODUCT('zone_aussen','zone_aussen','',());\n"
            "#11 = PRODUCT('zone_innen','zone_innen','',());\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_assembly_map(stp)
        assert "#10" in result
        assert result["#10"] == "zone_aussen"
        assert "#11" in result
        assert result["#11"] == "zone_innen"

    def test_extracts_nauo_component_names(self, tmp_path: Path):
        content = _make_step(
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','zone_mitte','',#3,#4,$);\n"
            "#6 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','zone_aussen','',#3,#7,$);\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_assembly_map(stp)
        # NAUO names should appear as values
        assert "zone_mitte" in result.values() or "zone_aussen" in result.values()

    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        result = _build_assembly_map(tmp_path / "nonexistent.stp")
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path: Path):
        stp = tmp_path / "empty.stp"
        stp.write_text("", encoding="utf-8")
        result = _build_assembly_map(stp)
        assert isinstance(result, dict)

    def test_multiline_entity_normalised(self, tmp_path: Path):
        # Entity split across lines should still be found after normalisation
        content = _make_step(
            "#10 = PRODUCT(\n  'zone_aussen',\n  'zone_aussen',\n  '',\n  ());\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_assembly_map(stp)
        assert "#10" in result
        assert result["#10"] == "zone_aussen"

    def test_empty_product_name_excluded(self, tmp_path: Path):
        content = _make_step(
            "#10 = PRODUCT('','empty','',());\n"
        )
        stp = tmp_path / "test.stp"
        stp.write_text(content, encoding="utf-8")
        result = _build_assembly_map(stp)
        # Empty name should not be stored
        assert "#10" not in result


# ---------------------------------------------------------------------------
# _get_external_file_id_from_internal
# ---------------------------------------------------------------------------

class TestGetExternalFileIdFromInternal:
    """Tests for the StringLabel-based external id helper."""

    def _make_model(self, label_return) -> MagicMock:
        """Build a mock OCC model where StringLabel returns *label_return*."""
        model = MagicMock()
        model.StringLabel.return_value = label_return
        return model

    def _make_model_str(self, label_str: str) -> MagicMock:
        """Build a mock OCC model whose StringLabel returns a ToCString mock."""
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = label_str
        return self._make_model(handle)

    # --- Happy path ---

    def test_hash_prefix_parsed(self):
        model = self._make_model_str("#56")
        assert _get_external_file_id_from_internal(model, 1) == 56

    def test_whitespace_around_hash(self):
        model = self._make_model_str("  #56  ")
        assert _get_external_file_id_from_internal(model, 1) == 56

    def test_no_hash_prefix(self):
        model = self._make_model_str("56")
        assert _get_external_file_id_from_internal(model, 1) == 56

    def test_plain_string_label(self):
        """When StringLabel returns a plain str (not a handle), it must be accepted."""
        model = MagicMock()
        model.StringLabel.return_value = "#99"
        assert _get_external_file_id_from_internal(model, 1) == 99

    # --- Edge cases: None / IsNull ---

    def test_empty_string_returns_none(self):
        model = self._make_model_str("")
        assert _get_external_file_id_from_internal(model, 1) is None

    def test_non_numeric_returns_none(self):
        model = self._make_model_str("foo")
        assert _get_external_file_id_from_internal(model, 1) is None

    def test_string_label_returns_none_handle(self):
        model = MagicMock()
        model.StringLabel.return_value = None
        assert _get_external_file_id_from_internal(model, 1) is None

    def test_is_null_true_returns_none(self):
        handle = MagicMock()
        handle.IsNull.return_value = True
        model = self._make_model(handle)
        assert _get_external_file_id_from_internal(model, 1) is None

    def test_string_label_raises_returns_none(self):
        model = MagicMock()
        model.StringLabel.side_effect = RuntimeError("OCC error")
        assert _get_external_file_id_from_internal(model, 1) is None

    # --- Guard conditions ---

    def test_none_model_returns_none(self):
        assert _get_external_file_id_from_internal(None, 1) is None

    def test_none_internal_number_returns_none(self):
        model = MagicMock()
        assert _get_external_file_id_from_internal(model, None) is None

    def test_zero_internal_number_returns_none(self):
        model = MagicMock()
        assert _get_external_file_id_from_internal(model, 0) is None

    def test_negative_internal_number_returns_none(self):
        model = MagicMock()
        assert _get_external_file_id_from_internal(model, -5) is None

    def test_handle_without_to_cstring_returns_none(self):
        """A handle with no ToCString and not a str returns None."""
        handle = MagicMock(spec=[])  # no ToCString, not a str
        model = self._make_model(handle)
        assert _get_external_file_id_from_internal(model, 1) is None


# ---------------------------------------------------------------------------
# _extract_via_step_id
# ---------------------------------------------------------------------------

class _FakeSolid:
    """Sentinel object standing in for a TopoDS_Solid."""
    def __init__(self, label: str = ""):
        self.label = label


def _make_transfer_reader_and_model(
    solids: list,
    internal_numbers: list[int | None],
    string_labels: dict[int, str],
) -> tuple[MagicMock, MagicMock]:
    """Build mocks for transfer_reader and model for _extract_via_step_id tests.

    Parameters
    ----------
    solids:
        List of solid objects (used as EntityFromShapeResult keys).
    internal_numbers:
        OCC-internal number returned by model.Number() for each solid's entity.
        ``None`` means the entity itself is None (no entity found).
    string_labels:
        Mapping {internal_number: label_str} for model.StringLabel().
    """
    transfer_reader = MagicMock()
    model = MagicMock()

    # Map solids -> entities by identity so EntityFromShapeResult is idempotent
    # (the entity_name primary path and Path A' both query it per solid).
    solid_to_entity: dict[int, object] = {}
    entity_to_number: dict[int, int | None] = {}
    for solid, num in zip(solids, internal_numbers, strict=False):
        if num is None:
            solid_to_entity[id(solid)] = None
        else:
            e = MagicMock()
            solid_to_entity[id(solid)] = e
            entity_to_number[id(e)] = num

    def efsr_side_effect(shape, mode=1):
        return solid_to_entity.get(id(shape))

    transfer_reader.EntityFromShapeResult.side_effect = efsr_side_effect

    def number_side_effect(entity):
        return entity_to_number.get(id(entity))

    model.Number.side_effect = number_side_effect

    def string_label_side_effect(internal: int):
        label_str = string_labels.get(internal)
        if label_str is None:
            return None
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = label_str
        return handle

    model.StringLabel.side_effect = string_label_side_effect

    return transfer_reader, model


class TestExtractViaStepId:
    """Tests for Path A' (_extract_via_step_id)."""

    def test_happy_path_three_solids(self):
        """All three solids resolved via StringLabel → brep_name_map lookup."""
        solids = [_FakeSolid(), _FakeSolid(), _FakeSolid()]
        brep_name_map = {56: "zone_mitte", 57: "zone_aussen", 58: "zone_innen"}

        transfer_reader, model = _make_transfer_reader_and_model(
            solids=solids,
            internal_numbers=[10, 11, 12],
            string_labels={10: "#56", 11: "#57", 12: "#58"},
        )

        results = _extract_via_step_id(transfer_reader, model, solids, brep_name_map)

        assert len(results) == 3
        assert results[0] == ("zone_mitte", 56)
        assert results[1] == ("zone_aussen", 57)
        assert results[2] == ("zone_innen", 58)

    def test_entity_not_found_returns_none(self):
        """When EntityFromShapeResult returns None, result is (None, None)."""
        solids = [_FakeSolid()]
        brep_name_map = {56: "zone_mitte"}

        transfer_reader, model = _make_transfer_reader_and_model(
            solids=solids,
            internal_numbers=[None],  # entity is None
            string_labels={},
        )

        results = _extract_via_step_id(transfer_reader, model, solids, brep_name_map)

        assert len(results) == 1
        assert results[0] == (None, None)

    def test_file_id_not_in_map_returns_none_name(self):
        """File id found but not in brep_name_map → name is None."""
        solids = [_FakeSolid()]
        brep_name_map = {99: "other"}  # 56 not present

        transfer_reader, model = _make_transfer_reader_and_model(
            solids=solids,
            internal_numbers=[10],
            string_labels={10: "#56"},
        )

        results = _extract_via_step_id(transfer_reader, model, solids, brep_name_map)

        assert results[0][0] is None  # name is None
        assert results[0][1] == 56   # file_id still resolved

    def test_entity_from_shape_result_raises_returns_none(self):
        """If EntityFromShapeResult raises, result is (None, None)."""
        solids = [_FakeSolid()]
        brep_name_map = {56: "zone_mitte"}

        transfer_reader = MagicMock()
        transfer_reader.EntityFromShapeResult.side_effect = RuntimeError("OCC crash")
        model = MagicMock()

        results = _extract_via_step_id(transfer_reader, model, solids, brep_name_map)

        assert len(results) == 1
        assert results[0] == (None, None)

    def test_model_number_raises_skips_gracefully(self):
        """If model.Number raises, file_id cannot be resolved; result is (None, None)."""
        solids = [_FakeSolid()]
        brep_name_map = {56: "zone_mitte"}

        transfer_reader = MagicMock()
        entity = MagicMock()
        transfer_reader.EntityFromShapeResult.return_value = entity

        model = MagicMock()
        model.Number.side_effect = RuntimeError("Number error")

        results = _extract_via_step_id(transfer_reader, model, solids, brep_name_map)

        assert len(results) == 1
        assert results[0][0] is None


# ---------------------------------------------------------------------------
# extract_solid_names  (mock-based, new path order)
# ---------------------------------------------------------------------------

def _make_reader_mock_step_id(
    solids: list,
    internal_numbers: list[int | None],
    string_labels: dict[int, str],
) -> MagicMock:
    """Build a mock STEPControl_Reader for Path A' (step_id) tests."""
    mock_reader = MagicMock()
    transfer_reader, model = _make_transfer_reader_and_model(
        solids, internal_numbers, string_labels
    )
    mock_reader.WS.return_value.TransferReader.return_value = transfer_reader
    mock_reader.WS.return_value.Model.return_value = model
    # Make TransientProcess accessible (needed by Path A'' and diagnose)
    mock_reader.WS.return_value.TransferReader.return_value.TransientProcess.return_value = (
        MagicMock()
    )
    return mock_reader


class TestExtractSolidNames:
    """Tests for the extract_solid_names orchestrator."""

    # --- Path A0 (entity_name): primary path ---

    def test_path_a0_entity_name_is_primary(self, tmp_path: Path):
        """entity.Name() resolves names directly and wins over later paths.

        Occurrence indices (``:1``) are stripped so unique parts stay
        suffix-free.
        """
        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('brep_fallback_name', #100);\n"
        )
        stp = tmp_path / "a0.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid("s0"), _FakeSolid("s1")]

        def _named_entity(name_str: str) -> MagicMock:
            entity = MagicMock()
            handle = MagicMock()
            handle.IsNull.return_value = False
            handle.ToCString.return_value = name_str
            entity.Name.return_value = handle
            return entity

        entities = {
            id(solids[0]): _named_entity("Aussenring:1"),
            id(solids[1]): _named_entity("Innenring:1"),
        }
        mock_reader = MagicMock()
        tr = mock_reader.WS.return_value.TransferReader.return_value
        tr.EntityFromShapeResult.side_effect = (
            lambda shape, mode=1: entities.get(id(shape))
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert [ns.name for ns in result] == ["Aussenring", "Innenring"]
        assert all(ns.source == "entity_name" for ns in result)

    def test_path_a0_duplicates_are_deduped(self, tmp_path: Path):
        """Genuine duplicate instances are numbered by _dedupe."""
        stp = tmp_path / "a0_dup.stp"
        stp.write_text(_make_step(""), encoding="utf-8")

        solids = [_FakeSolid("s0"), _FakeSolid("s1"), _FakeSolid("s2")]

        def _named_entity(name_str: str) -> MagicMock:
            entity = MagicMock()
            handle = MagicMock()
            handle.IsNull.return_value = False
            handle.ToCString.return_value = name_str
            entity.Name.return_value = handle
            return entity

        entities = {
            id(solids[0]): _named_entity("zyl_1:1"),
            id(solids[1]): _named_entity("zyl_1:2"),
            id(solids[2]): _named_entity("zyl_1:3"),
        }
        mock_reader = MagicMock()
        tr = mock_reader.WS.return_value.TransferReader.return_value
        tr.EntityFromShapeResult.side_effect = (
            lambda shape, mode=1: entities.get(id(shape))
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert [ns.name for ns in result] == ["zyl_1", "zyl_1_1", "zyl_1_2"]
        assert all(ns.source == "entity_name" for ns in result)

    # --- Path A' (step_id): primary path ---

    def test_path_a_prime_step_id_resolves_names(self, tmp_path: Path):
        """Names resolved via StringLabel + brep_name_map → source == 'step_id'."""
        content = _make_step(
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = tmp_path / "a_prime.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid("s0"), _FakeSolid("s1"), _FakeSolid("s2")]
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[10, 11, 12],
            string_labels={10: "#56", 11: "#57", 12: "#58"},
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert len(result) == 3
        assert result[0].name == "zone_mitte"
        assert result[0].source == "step_id"
        assert result[1].name == "zone_aussen"
        assert result[1].source == "step_id"
        assert result[2].name == "zone_innen"
        assert result[2].source == "step_id"

    def test_path_a_prime_wins_over_model_scan(self, tmp_path: Path, monkeypatch):
        """When Path A' succeeds, Path A'' is not consulted."""
        from meshing_utils.cad import step_names as _mod

        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
        )
        stp = tmp_path / "a_prime_wins.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid()]
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[10],
            string_labels={10: "#57"},
        )

        scan_called = []

        def fake_scan(reader, sols, brep_name_map):
            scan_called.append(True)
            return {}

        monkeypatch.setattr(_mod, "_map_solids_via_model_iteration", fake_scan)

        result = extract_solid_names(mock_reader, solids, stp)

        assert result[0].source == "step_id"
        assert result[0].name == "zone_aussen"
        assert scan_called == []

    # --- Path A'' (model_scan): fallback after A' ---

    def test_path_a_double_prime_used_when_a_prime_fails(
        self, tmp_path: Path, monkeypatch
    ):
        """When Path A' returns nothing, Path A'' geometry-match provides names."""
        from meshing_utils.cad import step_names as _mod

        content = _make_step(
            "#56 = BREP_WITH_VOIDS('zone_mitte', #200, (#201));\n"
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
            "#58 = MANIFOLD_SOLID_BREP('zone_innen', #101);\n"
        )
        stp = tmp_path / "scan.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid("s0"), _FakeSolid("s1"), _FakeSolid("s2")]

        # Path A' returns no entities (StringLabel not found)
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[None, None, None],
            string_labels={},
        )

        # Path A'' injected via monkeypatch
        scan_result = {
            id(solids[0]): "zone_aussen",
            id(solids[1]): "zone_innen",
            id(solids[2]): "zone_mitte",
        }

        monkeypatch.setattr(
            _mod,
            "_map_solids_via_model_iteration",
            lambda reader, sols, bnm: {k: v for k, v in scan_result.items()
                                       if any(id(s) == k for s in sols)},
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert len(result) == 3
        assert result[0].name == "zone_aussen"
        assert result[0].source == "model_scan"
        assert result[1].name == "zone_innen"
        assert result[1].source == "model_scan"
        assert result[2].name == "zone_mitte"
        assert result[2].source == "model_scan"

    # --- Path A''' (ordered_match): failsafe ---

    def test_path_a_triple_prime_used_when_a_prime_and_scan_fail(
        self, tmp_path: Path, monkeypatch
    ):
        """When Paths A' and A'' fail, ordered_match provides names."""
        from meshing_utils.cad import step_names as _mod

        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('zone_aussen', #100);\n"
        )
        stp = tmp_path / "ordered.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid()]
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[None],
            string_labels={},
        )

        monkeypatch.setattr(
            _mod, "_map_solids_via_model_iteration", lambda r, s, bnm: {}
        )
        ordered_result = {id(solids[0]): "zone_aussen"}
        monkeypatch.setattr(
            _mod, "_ordered_match_by_type", lambda r, s, e: ordered_result
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert result[0].source == "ordered_match"
        assert result[0].name == "zone_aussen"

    # --- Path B: assembly map via NAUO ---

    def test_path_b_assembly(self, tmp_path: Path):
        """Names resolved via NAUO regex when Paths A'/A''/A''' return None.

        In that case source == 'assembly'.
        """
        content = _make_step(
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','zone_mitte','',#3,#4,$);\n"
            "#6 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','zone_aussen','',#3,#7,$);\n"
        )
        stp = tmp_path / "c.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid(), _FakeSolid()]

        # reader=None skips Paths A', A'', A''' entirely
        result = extract_solid_names(None, solids, stp)

        assert len(result) == 2
        assert result[0].source == "assembly"
        assert result[1].source == "assembly"
        assert result[0].name == "zone_mitte"
        assert result[1].name == "zone_aussen"

    # --- Path C: generic fallback ---

    def test_path_c_generic_fallback(self, tmp_path: Path, caplog):
        """Names fall back to 'solid{i}' when all other paths fail."""
        stp = tmp_path / "d.stp"
        stp.write_text("no useful data", encoding="utf-8")

        solids = [_FakeSolid(), _FakeSolid(), _FakeSolid()]

        with caplog.at_level(logging.WARNING, logger="meshing_utils.cad.step_names"):
            result = extract_solid_names(None, solids, stp)

        assert len(result) == 3
        for i, ns in enumerate(result):
            assert ns.source == "generic"
            assert ns.name == f"solid{i}"

        # Warning must have been emitted
        assert any("generic" in rec.message for rec in caplog.records)

    # --- reader=None skips OCC paths ---

    def test_reader_none_skips_occ_paths(self, tmp_path: Path):
        """Passing reader=None must skip Paths A', A'', A''' entirely."""
        content = _make_step(
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','fluidZone','',#3,#4,$);\n"
        )
        stp = tmp_path / "f.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid()]
        result = extract_solid_names(None, solids, stp)

        assert len(result) == 1
        assert result[0].name == "fluidZone"
        assert result[0].source == "assembly"

    # --- Sanitisation applied ---

    def test_names_are_sanitised(self, tmp_path: Path):
        """Returned names must be OpenFOAM-compatible."""
        content = _make_step(
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','my zone!','',#3,#4,$);\n"
        )
        stp = tmp_path / "g.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid()]
        result = extract_solid_names(None, solids, stp)

        # "my zone!" → "my_zone_" after replace, then strip → "my_zone"
        assert result[0].name == "my_zone"

    # --- De-duplication applied ---

    def test_duplicate_names_deduped(self, tmp_path: Path):
        """Duplicate names across solids must be de-duplicated."""
        content = _make_step(
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','zone','',#3,#4,$);\n"
            "#6 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','zone','',#3,#7,$);\n"
        )
        stp = tmp_path / "h.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid(), _FakeSolid()]
        result = extract_solid_names(None, solids, stp)

        assert result[0].name == "zone"
        assert result[1].name == "zone_1"

    # --- NamedSolid.solid reference preserved ---

    def test_solid_reference_preserved(self, tmp_path: Path):
        """The solid object in NamedSolid must be the original object."""
        stp = tmp_path / "i.stp"
        stp.write_text("no data", encoding="utf-8")

        s = _FakeSolid()
        result = extract_solid_names(None, [s], stp)
        assert result[0].solid is s

    # --- Full three-path chain: A' → B → C ---

    def test_three_path_order_a_prime_then_b_then_c(self, tmp_path: Path, monkeypatch):
        """Demonstrates the A' → B → C fallback chain.

        solid[0]: Path A' succeeds (StringLabel found in brep_name_map)
        solid[1]: Path A' fails, A''/A''' patched to return nothing,
                  Path B succeeds (NAUO)
        solid[2]: All paths fail → Path C (generic)
        """
        from meshing_utils.cad import step_names as _mod

        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('path_a_prime_name', #100);\n"
            # NAUO for solid[1] only; 1 NAUO entry matches 1 missing solid
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','path_b_name','',#3,#4,$);\n"
        )
        stp = tmp_path / "three_paths.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid(), _FakeSolid(), _FakeSolid()]

        # solid[0]: entity found, StringLabel → #57 → in map
        # solid[1], solid[2]: no entity
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[10, None, None],
            string_labels={10: "#57"},
        )
        # Patch A'' and A''' to return nothing
        monkeypatch.setattr(_mod, "_map_solids_via_model_iteration", lambda r, s, bnm: {})
        monkeypatch.setattr(_mod, "_ordered_match_by_type", lambda r, s, e: {})

        result = extract_solid_names(mock_reader, solids, stp)

        assert len(result) == 3
        assert result[0].name == "path_a_prime_name"
        assert result[0].source == "step_id"
        assert result[1].name == "path_b_name"
        assert result[1].source == "assembly"
        assert result[2].source == "generic"
        assert result[2].name == "solid2"

    # --- Full five-path chain: A' → A'' → A''' → B → C ---

    def test_five_path_full_chain(self, tmp_path: Path, monkeypatch):
        """Full 5-path chain across five solids.

        solid[0]: Path A'   succeeds (step_id)
        solid[1]: Path A''  succeeds (model_scan)
        solid[2]: Path A''' succeeds (ordered_match)
        solid[3]: Path B    succeeds (assembly)
        solid[4]: All fail  → Path C (generic)
        """
        from meshing_utils.cad import step_names as _mod

        content = _make_step(
            "#57 = MANIFOLD_SOLID_BREP('name_a_prime', #100);\n"
            "#5 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','name_b','',#3,#4,$);\n"
        )
        stp = tmp_path / "five_paths.stp"
        stp.write_text(content, encoding="utf-8")

        solids = [_FakeSolid() for _ in range(5)]

        # solid[0]: resolved via Path A'
        mock_reader = _make_reader_mock_step_id(
            solids=solids,
            internal_numbers=[10, None, None, None, None],
            string_labels={10: "#57"},
        )

        # solid[1]: resolved via Path A''
        scan_result = {id(solids[1]): "name_a_double_prime"}
        monkeypatch.setattr(
            _mod, "_map_solids_via_model_iteration",
            lambda r, s, bnm: {k: v for k, v in scan_result.items()
                                if any(id(x) == k for x in s)},
        )

        # solid[2]: resolved via Path A'''
        ordered_result = {id(solids[2]): "name_a_triple_prime"}
        monkeypatch.setattr(
            _mod, "_ordered_match_by_type",
            lambda r, s, e: {k: v for k, v in ordered_result.items()
                             if any(id(x) == k for x in s)},
        )

        result = extract_solid_names(mock_reader, solids, stp)

        assert len(result) == 5
        assert result[0].name == "name_a_prime"
        assert result[0].source == "step_id"
        assert result[1].name == "name_a_double_prime"
        assert result[1].source == "model_scan"
        assert result[2].name == "name_a_triple_prime"
        assert result[2].source == "ordered_match"
        assert result[3].name == "name_b"
        assert result[3].source == "assembly"
        assert result[4].source == "generic"
        assert result[4].name == "solid4"


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------

def _make_model_scan_reader(
    nb_entities: int,
    entities: list,  # list of (mock_entity | None) in model order (1-based)
    tp_results: dict,  # {entity_mock: shape_mock | None}
    string_labels: dict[int, str] | None = None,
) -> MagicMock:
    """Build a mock STEPControl_Reader for _map_solids_via_model_iteration tests.

    Parameters
    ----------
    nb_entities:
        Value returned by ``model.NbEntities()``.
    entities:
        List of length *nb_entities*.  ``model.Value(i)`` returns
        ``entities[i-1]`` (1-based indexing).
    tp_results:
        Mapping from entity mock → shape mock (or ``None``) returned by
        ``tp.Find(entity).Result()``.
    string_labels:
        Optional mapping {1-based index: label_str} for model.StringLabel().
        When ``None``, StringLabel always returns ``None``.
    """
    mock_reader = MagicMock()
    mock_model = MagicMock()
    mock_tp = MagicMock()

    mock_reader.WS.return_value.Model.return_value = mock_model
    mock_reader.WS.return_value.TransferReader.return_value.TransientProcess.return_value = mock_tp

    mock_model.NbEntities.return_value = nb_entities

    def model_value(i: int):
        if 1 <= i <= len(entities):
            return entities[i - 1]
        return None

    mock_model.Value.side_effect = model_value

    # StringLabel mock
    _labels = string_labels or {}

    def string_label_side_effect(internal: int):
        label_str = _labels.get(internal)
        if label_str is None:
            return None
        handle = MagicMock()
        handle.IsNull.return_value = False
        handle.ToCString.return_value = label_str
        return handle

    mock_model.StringLabel.side_effect = string_label_side_effect

    def tp_find(entity):
        shape = tp_results.get(id(entity))
        if shape is None:
            return None
        binder = MagicMock()
        binder.Result.return_value = shape
        return binder

    mock_tp.Find.side_effect = tp_find
    # Ensure FindTransient also returns None for entities not in tp_results
    mock_tp.FindTransient.return_value = None

    return mock_reader


def _make_entity(
    type_name: str,
    name_str: str = "",
) -> MagicMock:
    """Create a mock OCC STEP entity with the given dynamic type and optional name."""
    entity = MagicMock()
    entity.DynamicType.return_value.Name.return_value = type_name
    entity.Name.return_value.ToCString.return_value = name_str
    return entity


def _make_shape_matching(solid) -> MagicMock:
    """Create a mock TopoDS_Shape that matches *solid* via IsSame."""
    shape = MagicMock()
    shape.IsSame.side_effect = lambda s: s is solid
    shape.IsPartner.return_value = False
    return shape


# ---------------------------------------------------------------------------
# _map_solids_via_model_iteration (updated signature with brep_name_map)
# ---------------------------------------------------------------------------

class TestMapSolidsViaModelIteration:
    """Tests for the Path A'' geometry-matching scan."""

    def test_single_solid_matched_via_brep_name_map(self):
        """Happy path: entity at index 1, StringLabel=#57 → brep_name_map[57]."""
        solid = _FakeSolid("s0")
        shape = _make_shape_matching(solid)
        entity = _make_entity("StepShape_ManifoldSolidBrep")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={id(entity): shape},
            string_labels={1: "#57"},
        )

        brep_name_map = {57: "zone_aussen"}
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)

        assert id(solid) in result
        assert result[id(solid)] == "zone_aussen"

    def test_brep_with_voids_type_matched(self):
        """BREP_WITH_VOIDS entity type is recognised and name resolved from map."""
        solid = _FakeSolid("s0")
        shape = _make_shape_matching(solid)
        entity = _make_entity("StepShape_BrepWithVoids")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={id(entity): shape},
            string_labels={1: "#56"},
        )

        brep_name_map = {56: "zone_mitte"}
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)

        assert id(solid) in result
        assert result[id(solid)] == "zone_mitte"

    def test_non_brep_entity_ignored(self):
        """Entities with non-BREP type names are skipped silently."""
        solid = _FakeSolid("s0")
        shape = _make_shape_matching(solid)
        entity = _make_entity("StepShape_EdgeCurve")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={id(entity): shape},
            string_labels={1: "#10"},
        )

        brep_name_map = {10: "should_not_appear"}
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)
        assert result == {}

    def test_multiple_solids_matched_independently(self):
        """Each solid is matched to its correct entity."""
        s0 = _FakeSolid("s0")
        s1 = _FakeSolid("s1")
        s2 = _FakeSolid("s2")

        shape0 = _make_shape_matching(s0)
        shape1 = _make_shape_matching(s1)
        shape2 = _make_shape_matching(s2)

        e0 = _make_entity("StepShape_ManifoldSolidBrep")
        e1 = _make_entity("StepShape_ManifoldSolidBrep")
        e2 = _make_entity("StepShape_BrepWithVoids")

        reader = _make_model_scan_reader(
            nb_entities=3,
            entities=[e0, e1, e2],
            tp_results={id(e0): shape0, id(e1): shape1, id(e2): shape2},
            string_labels={1: "#57", 2: "#58", 3: "#56"},
        )

        brep_name_map = {57: "zone_aussen", 58: "zone_innen", 56: "zone_mitte"}
        result = _map_solids_via_model_iteration(reader, [s0, s1, s2], brep_name_map)

        assert result.get(id(s0)) == "zone_aussen"
        assert result.get(id(s1)) == "zone_innen"
        assert result.get(id(s2)) == "zone_mitte"

    def test_entity_with_no_binder_skipped(self):
        """When tp.Find returns None, that entity is skipped."""
        solid = _FakeSolid("s0")
        entity = _make_entity("StepShape_ManifoldSolidBrep")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={},  # no binder
            string_labels={1: "#57"},
        )

        brep_name_map = {57: "zone_aussen"}
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)
        assert result == {}

    def test_file_id_not_in_map_skips_solid(self):
        """When StringLabel resolves an id not in brep_name_map, solid is not added."""
        solid = _FakeSolid("s0")
        shape = _make_shape_matching(solid)
        entity = _make_entity("StepShape_ManifoldSolidBrep")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={id(entity): shape},
            string_labels={1: "#99"},
        )

        brep_name_map = {57: "zone_aussen"}  # 99 not present
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)
        assert result == {}

    def test_model_value_raises_exception_skipped(self):
        """If model.Value() raises, that index is skipped without crashing."""
        solid = _FakeSolid("s0")

        reader = MagicMock()
        mock_model = MagicMock()
        reader.WS.return_value.Model.return_value = mock_model
        transfer_reader = reader.WS.return_value.TransferReader.return_value
        transfer_reader.TransientProcess.return_value = MagicMock()

        mock_model.NbEntities.return_value = 1
        mock_model.Value.side_effect = RuntimeError("OCC crash")

        result = _map_solids_via_model_iteration(reader, [solid], {})
        assert result == {}

    def test_ws_access_failure_returns_empty_dict(self):
        """If reader.WS() raises, an empty dict is returned immediately."""
        solid = _FakeSolid("s0")

        reader = MagicMock()
        reader.WS.side_effect = RuntimeError("no WS")

        result = _map_solids_via_model_iteration(reader, [solid], {})
        assert result == {}

    def test_is_same_raises_falls_back_to_is_partner(self):
        """When IsSame raises, IsPartner is tried as a fallback."""
        solid = _FakeSolid("s0")

        shape = MagicMock()
        shape.IsSame.side_effect = RuntimeError("IsSame error")
        shape.IsPartner.side_effect = lambda s: s is solid

        entity = _make_entity("StepShape_ManifoldSolidBrep")

        reader = _make_model_scan_reader(
            nb_entities=1,
            entities=[entity],
            tp_results={id(entity): shape},
            string_labels={1: "#57"},
        )

        brep_name_map = {57: "zone_partner"}
        result = _map_solids_via_model_iteration(reader, [solid], brep_name_map)
        assert id(solid) in result
        assert result[id(solid)] == "zone_partner"


# ---------------------------------------------------------------------------
# _ordered_match_by_type
# ---------------------------------------------------------------------------

class TestOrderedMatchByType:
    """Tests for Path A''' (_ordered_match_by_type)."""

    def _make_reader_for_ordered(
        self,
        solids: list,
        entity_types: list[str],
        occ_numbers: list[int],
    ) -> MagicMock:
        """Build a mock reader where EntityFromShapeResult returns entities
        with the given types and model.Number returns the given occ_numbers."""
        reader = MagicMock()
        model = MagicMock()
        transfer_reader = MagicMock()

        reader.WS.return_value.Model.return_value = model
        reader.WS.return_value.TransferReader.return_value = transfer_reader

        entities = []
        entity_to_number: dict[int, int] = {}
        for typ, num in zip(entity_types, occ_numbers, strict=False):
            e = MagicMock()
            e.DynamicType.return_value.Name.return_value = typ
            entities.append(e)
            entity_to_number[id(e)] = num

        transfer_reader.EntityFromShapeResult.side_effect = entities

        def number_side_effect(entity):
            return entity_to_number.get(id(entity))

        model.Number.side_effect = number_side_effect

        return reader

    def test_happy_path_single_type(self):
        """1 BrepWithVoids + 2 ManifoldSolidBrep, correctly correlated."""
        s0 = _FakeSolid()  # ManifoldSolidBrep occ_number=10
        s1 = _FakeSolid()  # ManifoldSolidBrep occ_number=11
        s2 = _FakeSolid()  # BrepWithVoids     occ_number=5

        reader = self._make_reader_for_ordered(
            solids=[s0, s1, s2],
            entity_types=[
                "StepShape_ManifoldSolidBrep",
                "StepShape_ManifoldSolidBrep",
                "StepShape_BrepWithVoids",
            ],
            occ_numbers=[10, 11, 5],
        )

        brep_entries = [
            BrepSolidEntry(file_id=56, type_token="BREP_WITH_VOIDS", name="zone_mitte"),
            BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="zone_aussen"),
            BrepSolidEntry(file_id=58, type_token="MANIFOLD_SOLID_BREP", name="zone_innen"),
        ]

        result = _ordered_match_by_type(reader, [s0, s1, s2], brep_entries)

        # ManifoldSolidBrep sorted by occ_number: s0(10) < s1(11)
        # File order for MANIFOLD_SOLID_BREP: zone_aussen(57), zone_innen(58)
        assert result.get(id(s0)) == "zone_aussen"
        assert result.get(id(s1)) == "zone_innen"
        # BrepWithVoids: s2(5), one file entry → zone_mitte
        assert result.get(id(s2)) == "zone_mitte"

    def test_count_mismatch_emits_warning_and_skips(self, caplog):
        """When file count != OCC count for a type, warning is emitted and type skipped."""
        s0 = _FakeSolid()

        reader = self._make_reader_for_ordered(
            solids=[s0],
            entity_types=["StepShape_ManifoldSolidBrep"],
            occ_numbers=[10],
        )

        # Two file entries but only one OCC solid → mismatch
        brep_entries = [
            BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="zone_a"),
            BrepSolidEntry(file_id=58, type_token="MANIFOLD_SOLID_BREP", name="zone_b"),
        ]

        with caplog.at_level(logging.WARNING, logger="meshing_utils.cad.step_names"):
            result = _ordered_match_by_type(reader, [s0], brep_entries)

        assert result == {}
        assert any("mismatch" in rec.message.lower() for rec in caplog.records)

    def test_sorting_by_occ_number(self):
        """Solids with higher OCC numbers map to later file entries."""
        s0 = _FakeSolid()  # occ_number=20 (second in file)
        s1 = _FakeSolid()  # occ_number=10 (first in file)

        reader = self._make_reader_for_ordered(
            solids=[s0, s1],
            entity_types=[
                "StepShape_ManifoldSolidBrep",
                "StepShape_ManifoldSolidBrep",
            ],
            occ_numbers=[20, 10],
        )

        brep_entries = [
            BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="first_in_file"),
            BrepSolidEntry(file_id=58, type_token="MANIFOLD_SOLID_BREP", name="second_in_file"),
        ]

        result = _ordered_match_by_type(reader, [s0, s1], brep_entries)

        # s1 has occ_number=10 (lower) → first_in_file
        # s0 has occ_number=20 (higher) → second_in_file
        assert result.get(id(s1)) == "first_in_file"
        assert result.get(id(s0)) == "second_in_file"

    def test_ws_failure_returns_empty_dict(self):
        """If reader.WS() raises, empty dict is returned."""
        reader = MagicMock()
        reader.WS.side_effect = RuntimeError("WS error")

        result = _ordered_match_by_type(reader, [_FakeSolid()], [])
        assert result == {}

    def test_unknown_type_token_ignored(self):
        """Entries with unknown type tokens are silently ignored."""
        s0 = _FakeSolid()

        reader = self._make_reader_for_ordered(
            solids=[s0],
            entity_types=["StepShape_ManifoldSolidBrep"],
            occ_numbers=[10],
        )

        brep_entries = [
            BrepSolidEntry(file_id=99, type_token="UNKNOWN_TYPE", name="should_not_appear"),
            BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="zone_aussen"),
        ]

        result = _ordered_match_by_type(reader, [s0], brep_entries)

        # Only the MANIFOLD_SOLID_BREP entry is used; count matches (1==1)
        assert result.get(id(s0)) == "zone_aussen"

    def test_entity_not_in_brep_types_ignored(self):
        """OCC entities whose DynamicType is not a BREP type are ignored."""
        s0 = _FakeSolid()

        reader = self._make_reader_for_ordered(
            solids=[s0],
            entity_types=["StepShape_EdgeCurve"],  # not a BREP type
            occ_numbers=[10],
        )

        brep_entries = [
            BrepSolidEntry(file_id=57, type_token="MANIFOLD_SOLID_BREP", name="zone_aussen"),
        ]

        result = _ordered_match_by_type(reader, [s0], brep_entries)
        # OCC has 0 BREP solids, file has 1 → mismatch → empty
        assert result == {}

    def test_empty_solids_returns_empty(self):
        """Empty solids list produces empty result."""
        reader = MagicMock()
        reader.WS.return_value.Model.return_value = MagicMock()
        reader.WS.return_value.TransferReader.return_value.EntityFromShapeResult.return_value = None

        result = _ordered_match_by_type(reader, [], [])
        assert result == {}

    def test_empty_brep_entries_returns_empty(self):
        """Empty brep_entries list produces empty result."""
        s0 = _FakeSolid()
        reader = self._make_reader_for_ordered(
            solids=[s0],
            entity_types=["StepShape_ManifoldSolidBrep"],
            occ_numbers=[10],
        )

        result = _ordered_match_by_type(reader, [s0], [])
        assert result == {}


# ---------------------------------------------------------------------------
# _diagnose_occ_mapping
# ---------------------------------------------------------------------------

class TestDiagnoseOccMapping:
    """Tests for the debug diagnostics helper."""

    def test_returns_empty_list_when_reader_is_none(self, tmp_path: Path):
        stp = tmp_path / "diag.stp"
        stp.write_text("", encoding="utf-8")
        result = _diagnose_occ_mapping(None, [_FakeSolid()], stp)
        assert result == []

    def test_returns_one_entry_per_solid(self, tmp_path: Path):
        stp = tmp_path / "diag2.stp"
        stp.write_text("", encoding="utf-8")

        solids = [_FakeSolid("s0"), _FakeSolid("s1")]

        mock_reader = MagicMock()
        mock_transfer_reader = MagicMock()
        mock_model = MagicMock()
        mock_reader.WS.return_value.TransferReader.return_value = mock_transfer_reader
        mock_reader.WS.return_value.Model.return_value = mock_model

        # EntityFromShapeResult returns None for both solids
        mock_transfer_reader.EntityFromShapeResult.return_value = None

        result = _diagnose_occ_mapping(mock_reader, solids, stp)

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    def test_captures_entity_type_and_name(self, tmp_path: Path):
        stp = tmp_path / "diag3.stp"
        stp.write_text("", encoding="utf-8")

        solid = _FakeSolid("s0")

        entity = MagicMock()
        entity.DynamicType.return_value.Name.return_value = "StepShape_ManifoldSolidBrep"
        entity.Name.return_value.ToCString.return_value = "my_zone"

        mock_reader = MagicMock()
        mock_model = MagicMock()
        transfer_reader = mock_reader.WS.return_value.TransferReader.return_value
        transfer_reader.EntityFromShapeResult.return_value = entity
        mock_reader.WS.return_value.Model.return_value = mock_model
        mock_model.Number.return_value = 42

        result = _diagnose_occ_mapping(mock_reader, [solid], stp)

        assert len(result) == 1
        entry = result[0]
        assert entry["entity_type"] == "StepShape_ManifoldSolidBrep"
        assert entry["entity_number"] == 42
        assert entry["has_name_method"] is True
        assert entry["name_value"] == "my_zone"

    def test_ws_failure_returns_empty_list(self, tmp_path: Path):
        stp = tmp_path / "diag4.stp"
        stp.write_text("", encoding="utf-8")

        mock_reader = MagicMock()
        mock_reader.WS.side_effect = RuntimeError("WS error")

        result = _diagnose_occ_mapping(mock_reader, [_FakeSolid()], stp)
        assert result == []

    def test_includes_step_path_in_each_entry(self, tmp_path: Path):
        stp = tmp_path / "diag5.stp"
        stp.write_text("", encoding="utf-8")

        mock_reader = MagicMock()
        transfer_reader = mock_reader.WS.return_value.TransferReader.return_value
        transfer_reader.EntityFromShapeResult.return_value = None
        mock_reader.WS.return_value.Model.return_value = MagicMock()

        result = _diagnose_occ_mapping(mock_reader, [_FakeSolid()], stp)

        assert len(result) == 1
        assert result[0]["step_path"] == str(stp)


# ---------------------------------------------------------------------------
# Integration tests against real STEP fixtures (require OCP)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestExtractSolidNamesWithOCP:
    """End-to-end naming against real STEP files. Skipped without OCP."""

    def _load(self, rel_path: str) -> list:
        pytest.importorskip("OCP")
        from meshing_utils.cad.step_loader import load_solids_with_names

        stp = _REPO_ROOT / rel_path
        if not stp.exists():
            pytest.skip(f"fixture missing: {stp}")
        return load_solids_with_names(stp)

    def test_single_part_brep_names_not_swapped(self):
        """A single part with named BREP solids -> hex_1/hex_2/hex_3, no suffix."""
        named = self._load("test_blocks.stp")
        names = [ns.name for ns in named]
        assert set(names) == {"hex_1", "hex_2", "hex_3"}
        assert len(names) == len(set(names))  # no spurious _dedupe suffixes
        assert all(ns.source == "entity_name" for ns in named)

    def test_assembly_occurrence_names(self):
        """Assembly -> clean component names; repeated instances numbered."""
        named = self._load("cad_files/patches_lager_komplett.stp")
        names = [ns.name for ns in named]
        # Unique components keep clean, suffix-free names.
        for expected in ("Aussenring", "Innenring", "kaefig", "seite_1", "seite_2"):
            assert expected in names
        # The 14 repeated 'zyl_1' instances are numbered, not collapsed.
        zyl = [nm for nm in names if nm == "zyl_1" or nm.startswith("zyl_1_")]
        assert len(zyl) == 14
        assert len(names) == len(set(names))  # all names unique
        assert all(ns.source == "entity_name" for ns in named)
