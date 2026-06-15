"""Tests for meshing_utils.operations.cell_zones."""

from __future__ import annotations

import pytest

# Phase 3.4: monkeypatch helpers must target the submodule that imports them.
import meshing_utils.operations.cell_zones.classification as cz_classification_mod
import meshing_utils.operations.cell_zones.core as cz_core_mod
from meshing_utils import Block, BlockMeshDict, Vertex
from meshing_utils.geometry.containment import (
    STATE_IN,
    STATE_ON,
    STATE_OUT,
)
from meshing_utils.operations.cell_zones import (
    SAMPLING_CENTROID,
    SAMPLING_INSET,
    STAGE_A_NOT_APPLICABLE,
    BlockSolidCounts,
    _assign_unique_zone_names,
    _resolve_vertex_dominant,
    _sanitize_zone_name,
    assign_cell_zones,
)

# ---------------------------------------------------------------------------
# Sentinel solid objects (no OCC needed)
# ---------------------------------------------------------------------------

class _Solid:
    """Sentinel object to represent an OCC solid without OCC."""

    def __init__(self, name: str = ""):
        self._name = name

    def __repr__(self) -> str:
        return f"<Solid {self._name!r}>"


SOLID_A = _Solid("A")
SOLID_B = _Solid("B")
SOLID_C = _Solid("C")


# ---------------------------------------------------------------------------
# Minimal BlockMeshDict factory
# ---------------------------------------------------------------------------

def _make_unit_cube_bmd(block_name: str = "b0") -> tuple[BlockMeshDict, Block]:
    """Return (bmd, block) with a single unit-cube block."""
    bmd = BlockMeshDict()
    coords = [
        ("v0", [0.0, 0.0, 0.0]),
        ("v1", [1.0, 0.0, 0.0]),
        ("v2", [1.0, 1.0, 0.0]),
        ("v3", [0.0, 1.0, 0.0]),
        ("v4", [0.0, 0.0, 1.0]),
        ("v5", [1.0, 0.0, 1.0]),
        ("v6", [1.0, 1.0, 1.0]),
        ("v7", [0.0, 1.0, 1.0]),
    ]
    for name, c in coords:
        bmd.vertices.add(Vertex(name, c))
    block = Block(
        block_name,
        vertices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        cells=[1, 1, 1],
    )
    bmd.blocks.add(block)
    return bmd, block


# ---------------------------------------------------------------------------
# _sanitize_zone_name
# ---------------------------------------------------------------------------

class TestSanitizeZoneName:

    def test_happy_path(self):
        assert _sanitize_zone_name("fluid", 0) == "fluid"

    def test_special_chars_replaced(self):
        assert _sanitize_zone_name("my-zone.1", 0) == "my_zone_1"

    def test_leading_digit_gets_prefix(self):
        result = _sanitize_zone_name("1zone", 0)
        assert result.startswith("z_")

    def test_none_falls_back(self):
        assert _sanitize_zone_name(None, 3) == "solid3"

    def test_empty_string_falls_back(self):
        assert _sanitize_zone_name("", 5) == "solid5"

    def test_whitespace_only_falls_back(self):
        # Whitespace becomes underscores, so the result is non-empty —
        # but pure-whitespace is not the same as empty after replacement.
        # The plan only mandates None/empty → fallback; whitespace-only
        # strings become underscore strings (valid identifiers).
        result = _sanitize_zone_name("   ", 2)
        assert result == "___"


# ---------------------------------------------------------------------------
# _assign_unique_zone_names
# ---------------------------------------------------------------------------

class TestAssignUniqueZoneNames:

    def test_unique_labels(self):
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        names = _assign_unique_zone_names(pairs)
        assert names == ["fluid", "solid"]

    def test_duplicate_labels_get_suffix(self):
        pairs = [(SOLID_A, "region"), (SOLID_B, "region")]
        names = _assign_unique_zone_names(pairs)
        assert names[0] == "region"
        assert names[1] == "region_2"

    def test_triple_duplicate(self):
        pairs = [(SOLID_A, "zone"), (SOLID_B, "zone"), (SOLID_C, "zone")]
        names = _assign_unique_zone_names(pairs)
        assert names[0] == "zone"
        assert names[1] == "zone_2"
        assert names[2] == "zone_3"

    def test_mixed_none_and_labels(self):
        pairs = [(SOLID_A, None), (SOLID_B, "fluid"), (SOLID_C, None)]
        names = _assign_unique_zone_names(pairs)
        assert names[0] == "solid0"
        assert names[1] == "fluid"
        assert names[2] == "solid2"


# ---------------------------------------------------------------------------
# assign_cell_zones — mock classify_point_in_solid for centroid strategy
# ---------------------------------------------------------------------------

class TestAssignCellZones:
    """Tests using centroid strategy (Stage A always skipped)."""

    @staticmethod
    def _patch(monkeypatch, return_map: dict):
        """Replace classify_point_in_solid with a function that returns
        return_map[(solid, point)] or STATE_OUT as default."""
        import meshing_utils.geometry.containment as cont_mod

        def _fake_classify(solid, point, tol):
            return return_map.get((id(solid), point), STATE_OUT)

        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", _fake_classify)

    # -----------------------------------------------------------------------
    # single IN hit
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_single_in_hit(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        self._patch(monkeypatch, {(id(SOLID_A), centroid): STATE_IN})
        pairs = [(SOLID_A, "fluid")]
        mapping = assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    # -----------------------------------------------------------------------
    # no hit → unzoned, no exception even with strict=True
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_no_hit_stays_unzoned(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        self._patch(monkeypatch, {})  # everything returns OUT
        pairs = [(SOLID_A, "fluid")]
        mapping = assign_cell_zones(bmd, pairs, strict=True, sampling_strategy=SAMPLING_CENTROID)
        assert "b0" not in mapping
        assert block.zone is None

    # -----------------------------------------------------------------------
    # ambiguous IN, strict → RuntimeError
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_ambiguous_in_strict(self, monkeypatch):
        bmd, _ = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        self._patch(monkeypatch, {
            (id(SOLID_A), centroid): STATE_IN,
            (id(SOLID_B), centroid): STATE_IN,
        })
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        with pytest.raises(RuntimeError):
            assign_cell_zones(bmd, pairs, strict=True, sampling_strategy=SAMPLING_CENTROID)

    # -----------------------------------------------------------------------
    # ambiguous IN, non-strict → uses first, logs warning
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_ambiguous_in_nonstrict(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        self._patch(monkeypatch, {
            (id(SOLID_A), centroid): STATE_IN,
            (id(SOLID_B), centroid): STATE_IN,
        })
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, strict=False, sampling_strategy=SAMPLING_CENTROID)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    # -----------------------------------------------------------------------
    # single ON hit → accepted
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_single_on_accepted(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        self._patch(monkeypatch, {(id(SOLID_A), centroid): STATE_ON})
        pairs = [(SOLID_A, "fluid")]
        mapping = assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    # -----------------------------------------------------------------------
    # multiple ON hits, perturbation resolves to 1 IN
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_multiple_on_perturbation_resolves(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        epsilon = max(bmd_bbox_diagonal_value(bmd) * 1e-9, 1e-12)
        # With centroid strategy, samples[0] == centroid → perturbation shifts to (+eps,+eps,+eps)
        perturbed = (
            centroid[0] + epsilon,
            centroid[1] + epsilon,
            centroid[2] + epsilon,
        )

        def _fake_classify(solid, point, tol):
            if solid is SOLID_A and point == centroid:
                return STATE_ON
            if solid is SOLID_B and point == centroid:
                return STATE_ON
            if solid is SOLID_A and point == perturbed:
                return STATE_IN
            return STATE_OUT

        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", _fake_classify)

        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    # -----------------------------------------------------------------------
    # multiple ON hits, perturbation inconclusive → strict raises
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_multiple_on_perturbation_inconclusive_strict(
        self, monkeypatch
    ):
        bmd, _ = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)

        def _fake_classify(solid, point, tol):
            if point == centroid:
                return STATE_ON
            return STATE_OUT

        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", _fake_classify)

        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        with pytest.raises(RuntimeError):
            assign_cell_zones(bmd, pairs, strict=True, sampling_strategy=SAMPLING_CENTROID)

    # -----------------------------------------------------------------------
    # multiple ON hits, perturbation inconclusive → non-strict uses first ON
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_multiple_on_perturbation_inconclusive_nonstrict(
        self, monkeypatch
    ):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)

        def _fake_classify(solid, point, tol):
            if point == centroid:
                return STATE_ON
            return STATE_OUT

        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", _fake_classify)

        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, strict=False, sampling_strategy=SAMPLING_CENTROID)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    # -----------------------------------------------------------------------
    # skips blocks with wrong vertex count
    # -----------------------------------------------------------------------

    def test_assign_cell_zones_skips_non_hex_blocks(self, monkeypatch):
        bmd = BlockMeshDict()
        for name, c in [("v0", [0.0, 0.0, 0.0]), ("v1", [1.0, 0.0, 0.0]),
                        ("v2", [1.0, 1.0, 0.0]), ("v3", [0.0, 1.0, 0.0])]:
            bmd.vertices.add(Vertex(name, c))
        bad_block = Block("bad", vertices=["v0", "v1", "v2", "v3"], cells=[1, 1, 1])
        bmd.blocks.add(bad_block)
        self._patch(monkeypatch, {})
        pairs = [(SOLID_A, "fluid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {}
        assert bad_block.zone is None

    # -----------------------------------------------------------------------
    # block.zone attribute is set directly
    # -----------------------------------------------------------------------

    def test_block_zone_attribute_set(self, monkeypatch):
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        self._patch(monkeypatch, {(id(SOLID_A), centroid): STATE_IN})
        pairs = [(SOLID_A, "waterRegion")]
        assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID)
        assert block.zone == "waterRegion"


# ---------------------------------------------------------------------------
# Helper to compute bbox diagonal (reuses production code)
# ---------------------------------------------------------------------------

def bmd_bbox_diagonal_value(bmd) -> float:
    from meshing_utils.geometry.containment import bmd_bbox_diagonal
    return bmd_bbox_diagonal(bmd)


# ---------------------------------------------------------------------------
# TestAssignCellZonesInsetSampling
# ---------------------------------------------------------------------------

class TestAssignCellZonesInsetSampling:
    """Tests for the default 'inset' sampling strategy."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        """Patch classify_point_in_solid in both modules."""
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_inset_default_strategy_active(self, monkeypatch):
        """Inset strategy calls classify 17 times per solid (9 interior + 8 vertex,
        no early-exit since all return OUT)."""
        bmd, _ = _make_unit_cube_bmd("b0")
        call_count = {"n": 0}

        def _counting_classify(solid, point, tol):
            call_count["n"] += 1
            return STATE_OUT

        self._patch_classify(monkeypatch, _counting_classify)
        pairs = [(SOLID_A, "fluid")]
        assign_cell_zones(bmd, pairs)  # default = inset
        # 9 interior + 8 vertex samples x 1 solid = 17 calls
        assert call_count["n"] == 17

    def test_inset_all_samples_in_one_solid(self, monkeypatch):
        """All samples IN for SOLID_A, OUT for SOLID_B → SOLID_A wins via Stage A."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_inset_centroid_out_but_majority_in(self, monkeypatch):
        """Centroid OUT but 8/9 interior IN → majority selects SOLID_A (Stage B).

        Vertices all OUT to keep Stage A inactive.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)

        # Unit-cube vertex coords (raw, not inset)
        raw_vertex_coords = {
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        }

        def _fake_classify(solid, point, tol):
            if solid is not SOLID_A:
                return STATE_OUT
            # Raw vertices → OUT (keeps Stage A inactive: v_hits = 0 < 5)
            if point in raw_vertex_coords:
                return STATE_OUT
            # centroid returns OUT; all inset samples return IN
            if point == centroid:
                return STATE_OUT
            return STATE_IN

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        # 8/9 IN for SOLID_A interior (centroid OUT, 8 inset IN); 0 vertex hits → Stage B
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_inset_majority_tie_strict_raises(self, monkeypatch):
        """Stage B tie: 5/9 IN for A and 5/9 IN for B → strict=True raises RuntimeError.

        Vertices all OUT so Stage A does not activate.
        """
        bmd, _ = _make_unit_cube_bmd("b0")
        # Raw vertex coords for unit cube
        raw_vertex_coords = {
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        }
        # Interior call counters per solid (only count non-vertex points)
        interior_counts: dict = {id(SOLID_A): 0, id(SOLID_B): 0}

        def _fake_classify(solid, point, tol):
            # Raw vertices → always OUT (Stage A stays inactive)
            if point in raw_vertex_coords:
                return STATE_OUT
            # Interior points: return IN for first 5 calls per solid
            interior_counts[id(solid)] += 1
            return STATE_IN if interior_counts[id(solid)] <= 5 else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        with pytest.raises(RuntimeError):
            assign_cell_zones(bmd, pairs, strict=True)

    def test_inset_majority_tie_nonstrict_picks_first(self, monkeypatch):
        """Stage B tie: 5/9 IN for both → non-strict picks first (SOLID_A).

        Vertices all OUT so Stage A does not activate.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        raw_vertex_coords = {
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        }
        interior_counts: dict = {id(SOLID_A): 0, id(SOLID_B): 0}

        def _fake_classify(solid, point, tol):
            if point in raw_vertex_coords:
                return STATE_OUT
            interior_counts[id(solid)] += 1
            return STATE_IN if interior_counts[id(solid)] <= 5 else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, strict=False)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_inset_clear_winner_no_tie(self, monkeypatch):
        """SOLID_A gets all 9 interior IN + 8 vertex IN → Stage A wins trivially."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_inset_zero_in_falls_back_to_on_majority(self, monkeypatch):
        """No IN; SOLID_A 6/9 interior ON, SOLID_B 2/9 interior ON.

        Vertices all OUT → Stage A inactive; Stage B ON-majority selects A.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        raw_vertex_coords = {
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        }
        interior_counts: dict = {"a": 0, "b": 0}

        def _fake_classify(solid, point, tol):
            if point in raw_vertex_coords:
                return STATE_OUT
            if solid is SOLID_A:
                interior_counts["a"] += 1
                return STATE_ON if interior_counts["a"] <= 6 else STATE_OUT
            else:
                interior_counts["b"] += 1
                return STATE_ON if interior_counts["b"] <= 2 else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"


# ---------------------------------------------------------------------------
# TestEarlyExit
# ---------------------------------------------------------------------------

class TestEarlyExit:
    """Tests for the early-exit optimisation in _classify_block_two_pass."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_early_exit_skips_remaining_solids(self, monkeypatch):
        """When SOLID_A gets 9/9 interior IN + 8/8 vertex IN/ON, SOLID_B must not
        be classified at all (sharpened early-exit condition)."""
        bmd, _ = _make_unit_cube_bmd("b0")
        solid_b_calls = {"n": 0}

        def _fake_classify(solid, point, tol):
            if solid is SOLID_B:
                solid_b_calls["n"] += 1
            # SOLID_A: always IN (covers both interior and vertex samples)
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        assign_cell_zones(bmd, pairs)
        assert solid_b_calls["n"] == 0

    def test_no_early_exit_when_partial_in(self, monkeypatch):
        """SOLID_A gets 5/9 interior IN (not all) → SOLID_B is still classified."""
        bmd, _ = _make_unit_cube_bmd("b0")
        solid_b_calls = {"n": 0}
        solid_a_calls = {"n": 0}

        raw_vertex_coords = {
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        }

        def _fake_classify(solid, point, tol):
            if solid is SOLID_A:
                if point in raw_vertex_coords:
                    return STATE_OUT
                solid_a_calls["n"] += 1
                # Return IN for the first 5 interior calls, OUT for the rest
                return STATE_IN if solid_a_calls["n"] <= 5 else STATE_OUT
            if solid is SOLID_B:
                solid_b_calls["n"] += 1
            return STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        assign_cell_zones(bmd, pairs)
        # SOLID_B must have been classified (no early exit)
        assert solid_b_calls["n"] > 0

    def test_early_exit_inactive_for_centroid_strategy(self, monkeypatch):
        """With centroid strategy (interior_total=1), early-exit does NOT trigger
        (interior_total > 1 guard), so all solids are classified."""
        bmd, _ = _make_unit_cube_bmd("b0")
        solid_b_calls = {"n": 0}

        def _fake_classify(solid, point, tol):
            if solid is SOLID_B:
                solid_b_calls["n"] += 1
            return STATE_IN  # both solids return IN

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        # non-strict to avoid RuntimeError
        assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID, strict=False)
        # SOLID_B must still have been classified
        assert solid_b_calls["n"] > 0


# ---------------------------------------------------------------------------
# TestSamplingStrategy
# ---------------------------------------------------------------------------

class TestSamplingStrategy:
    """Tests for sampling strategy selection and parameter validation."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_centroid_strategy_uses_one_sample(self, monkeypatch):
        """centroid strategy makes exactly 1 classify call per solid."""
        bmd, _ = _make_unit_cube_bmd("b0")
        call_count = {"n": 0}

        def _counting_classify(solid, point, tol):
            call_count["n"] += 1
            return STATE_OUT

        self._patch_classify(monkeypatch, _counting_classify)
        pairs = [(SOLID_A, "fluid")]
        assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_CENTROID)
        assert call_count["n"] == 1

    def test_inset_strategy_uses_seventeen_samples(self, monkeypatch):
        """inset strategy makes exactly 17 classify calls per solid (9 interior + 8
        vertex, no early-exit since all return OUT)."""
        bmd, _ = _make_unit_cube_bmd("b0")
        call_count = {"n": 0}

        def _counting_classify(solid, point, tol):
            call_count["n"] += 1
            return STATE_OUT  # ensure no early-exit

        self._patch_classify(monkeypatch, _counting_classify)
        pairs = [(SOLID_A, "fluid")]
        assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_INSET)
        assert call_count["n"] == 17

    def test_invalid_sampling_strategy_raises(self, monkeypatch):
        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "fluid")]
        with pytest.raises(ValueError, match="sampling_strategy"):
            assign_cell_zones(bmd, pairs, sampling_strategy="foo")

    def test_inset_factor_propagated(self, monkeypatch):
        """inset_factor=0.3 must be forwarded to compute_block_sample_sets."""
        captured_factor = {}

        original_fn = cz_classification_mod.compute_block_sample_sets

        def _capturing_sample_sets(coords, inset_factor=0.5):
            captured_factor["f"] = inset_factor
            return original_fn(coords, inset_factor)

        monkeypatch.setattr(
            cz_classification_mod, "compute_block_sample_sets", _capturing_sample_sets
        )

        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(
            cz_classification_mod, "classify_point_in_solid", lambda s, p, t: STATE_OUT
        )
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", lambda s, p, t: STATE_OUT)

        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "fluid")]
        assign_cell_zones(bmd, pairs, sampling_strategy=SAMPLING_INSET, inset_factor=0.3)
        assert captured_factor.get("f") == pytest.approx(0.3)

    def test_invalid_vote_policy_raises(self, monkeypatch):
        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "fluid")]
        with pytest.raises(ValueError, match="vote_policy"):
            assign_cell_zones(bmd, pairs, vote_policy="random")


# ---------------------------------------------------------------------------
# TestPerturbationFallback (with inset strategy)
# ---------------------------------------------------------------------------

class TestPerturbationFallback:
    """Tests for the radial perturbation fallback with inset sampling.

    Vertices are all OUT to ensure Stage A does not activate, so Stage B
    (including perturbation) is exercised.
    """

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    # Unit-cube raw vertex coordinates (set for fast membership test)
    _RAW_VERTS = frozenset({
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    })

    def test_perturbation_resolves_on_tie_with_inset(self, monkeypatch):
        """Multiple ON interior candidates: perturbation resolves to SOLID_A.

        Vertices all OUT → Stage A inactive; Stage B perturbation runs.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS
        first_pass_points: set = set()
        first_pass_done = {"v": False}

        def _fake_classify(solid, point, tol):
            # Raw vertices → OUT (keeps Stage A inactive)
            if point in raw_verts:
                return STATE_OUT
            if not first_pass_done["v"]:
                first_pass_points.add(point)
                # Mark first pass done once we have seen 9 interior points
                if len(first_pass_points) >= 9:
                    first_pass_done["v"] = True
                return STATE_ON
            # Second pass (perturbation interior points)
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_perturbation_inconclusive_strict_raises_inset(self, monkeypatch):
        """All interior ON, perturbation also IN for both → strict raises."""
        bmd, _ = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS
        call_count = {"n": 0}

        def _fake_classify2(solid, point, tol):
            # Raw vertices → OUT (Stage A inactive)
            if point in raw_verts:
                return STATE_OUT
            call_count["n"] += 1
            # First 18 interior calls (9 interior x 2 solids) → ON
            if call_count["n"] <= 18:
                return STATE_ON
            # Perturbation pass: return IN for both → inconclusive
            return STATE_IN

        self._patch_classify(monkeypatch, _fake_classify2)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        with pytest.raises(RuntimeError):
            assign_cell_zones(bmd, pairs, strict=True)

    def test_perturbation_direction_uses_centroid_vector(self, monkeypatch):
        """Perturbation samples are shifted radially from the centroid."""
        import math

        import meshing_utils.geometry.containment as cont_mod

        bmd, _ = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        raw_verts = self._RAW_VERTS

        first_pass_done = {"v": False}
        first_pass_count = {"n": 0}
        perturbation_points: list = []

        def _fake_classify(solid, point, tol):
            if point in raw_verts:
                return STATE_OUT
            if not first_pass_done["v"]:
                first_pass_count["n"] += 1
                # 9 interior x 2 solids = 18 calls for first pass
                if first_pass_count["n"] >= 18:
                    first_pass_done["v"] = True
                return STATE_ON
            # Perturbation pass: record points for SOLID_A
            if solid is SOLID_A:
                perturbation_points.append(point)
            return STATE_IN if solid is SOLID_A else STATE_OUT

        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", _fake_classify)

        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        assign_cell_zones(bmd, pairs)

        # Verify the perturbed centroid is at (M+eps, M+eps, M+eps)
        assert len(perturbation_points) >= 1
        # The first perturbed point is the centroid shifted by (+eps,+eps,+eps)
        pert_centroid = perturbation_points[0]
        # It must be close to but not equal to the original centroid
        dx = pert_centroid[0] - centroid[0]
        dy = pert_centroid[1] - centroid[1]
        dz = pert_centroid[2] - centroid[2]
        assert dx > 0 and dy > 0 and dz > 0

        # Verify non-centroid inset points are shifted radially
        for pt in perturbation_points[1:]:
            dx_v = pt[0] - centroid[0]
            dy_v = pt[1] - centroid[1]
            dz_v = pt[2] - centroid[2]
            # The perturbation direction must point away from centroid
            L = math.sqrt(dx_v ** 2 + dy_v ** 2 + dz_v ** 2)
            assert L > 0


# ---------------------------------------------------------------------------
# TestSagittaEdgeCases
# ---------------------------------------------------------------------------

class TestSagittaEdgeCases:
    """Mock-based sagitta / curved-boundary scenarios."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    _RAW_VERTS = frozenset({
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    })

    def test_thin_annulus_30deg_centroid_outside_inset_inside(self, monkeypatch):
        """Centroid is OUT, 8 inset samples IN, 8 raw vertices ON → Stage A wins
        (v_on=8 >= 5 threshold) and assigns annulus.

        This is the thin-annulus regression case: without vertex-priority, the
        old code would use interior 8/9 IN and might still work, but the key
        regression is that vertices ON ensures Stage A activates.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        centroid = (0.5, 0.5, 0.5)
        raw_verts = self._RAW_VERTS

        def _fake_classify(solid, point, tol):
            if solid is not SOLID_A:
                return STATE_OUT
            if point in raw_verts:
                return STATE_ON  # 8 vertices ON → vertex_hits=8 ≥ 5 → Stage A
            if point == centroid:
                return STATE_OUT
            return STATE_IN  # 8 inset samples IN

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "annulus")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "annulus"}
        assert block.zone == "annulus"

    def test_convex_outer_shell_centroid_inside_inset_inside(self, monkeypatch):
        """All 9 interior + 8 vertex IN → Stage A wins immediately."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "outer")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "outer"}
        assert block.zone == "outer"

    def test_block_on_curved_boundary_face(self, monkeypatch):
        """All 8 raw vertices ON, centroid IN, 8 inset-samples IN → Stage A wins
        (v_on=8 >= 5) with SOLID_A."""
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS

        def _fake_classify(solid, point, tol):
            if solid is not SOLID_A:
                return STATE_OUT
            if point in raw_verts:
                return STATE_ON  # 8 vertices ON → vertex_hits=8 ≥ 5 → Stage A
            return STATE_IN  # centroid + inset samples all IN

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "curved")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "curved"}
        assert block.zone == "curved"


# ---------------------------------------------------------------------------
# TestStageAVertexDominant
# ---------------------------------------------------------------------------

class TestStageAVertexDominant:
    """Direct tests for Stage A vertex-dominant resolution."""

    @staticmethod
    def _make_counts(
        i_in=0, i_on=0, i_out=9,
        v_in=0, v_on=0, v_out=8,
    ) -> BlockSolidCounts:
        return BlockSolidCounts(i_in, i_on, i_out, v_in, v_on, v_out)

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_case1_pure_interior_all_vertices_in_solid_a_wins(self, monkeypatch):
        """8 vertices IN + all 9 interior IN → SOLID_A wins via Stage A."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_case2_thin_annulus_vertices_on_a_insets_in_b(self, monkeypatch):
        """Thin-annulus regression: v_hits(A)=8 via ON, all interior IN B.

        Without Stage A: interior vote would pick B.
        With Stage A: vertex_hits(A)=8 >= 5 → A wins.
        """
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = frozenset({
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        })

        def _fake_classify(solid, point, tol):
            if point in raw_verts:
                # A gets ON (boundary), B gets OUT
                return STATE_ON if solid is SOLID_A else STATE_OUT
            # Interior (centroid + insets): B gets IN, A gets OUT
            return STATE_IN if solid is SOLID_B else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "annulus"), (SOLID_B, "inner_cylinder")]
        mapping = assign_cell_zones(bmd, pairs)
        # Stage A: A has vertex_hits=8 (v_on=8), B has 0 → A wins
        assert mapping == {"b0": "annulus"}
        assert block.zone == "annulus"

    def test_case3_shared_boundary_tiebreak_via_interior_in(self, monkeypatch):
        """A and B both v_on=8, v_in=0; A interior_in=9, B interior_in=0 → A via TB3."""
        _bmd, _block = _make_unit_cube_bmd("b0")

        # Build BlockSolidCounts directly and test _resolve_vertex_dominant
        counts_a = BlockSolidCounts(9, 0, 0, 0, 8, 0)   # v_hits=8, v_on=8, i_in=9
        counts_b = BlockSolidCounts(0, 0, 9, 0, 8, 0)   # v_hits=8, v_on=8, i_in=0
        counts = [counts_a, counts_b]
        zone_names = ["A", "B"]

        # Dummy block with a name attribute
        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(_FakeBlock(), counts, zone_names, strict=False)
        assert result == 0  # index 0 = A

    def test_case5_asymmetric_5on_a_3on_b(self, monkeypatch):
        """A has vertex_on=5, B has vertex_on=3 → A wins (5 ≥ threshold=5)."""
        counts_a = BlockSolidCounts(0, 0, 9, 0, 5, 3)   # v_hits=5
        counts_b = BlockSolidCounts(0, 0, 9, 0, 3, 5)   # v_hits=3

        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(
            _FakeBlock(), [counts_a, counts_b], ["A", "B"], strict=False
        )
        assert result == 0  # A

    def test_threshold_boundary_4_hits_falls_through_to_stage_b(self):
        """max_hits=4 → Stage A returns STAGE_A_NOT_APPLICABLE."""
        counts = [BlockSolidCounts(0, 0, 9, 0, 4, 4)]  # v_hits=4 < threshold

        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(_FakeBlock(), counts, ["A"], strict=False)
        assert result is STAGE_A_NOT_APPLICABLE

    def test_threshold_boundary_5_hits_activates_stage_a(self):
        """max_hits=5 → Stage A activates and returns index 0."""
        counts = [BlockSolidCounts(0, 0, 9, 0, 5, 3)]  # v_hits=5 == threshold

        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(_FakeBlock(), counts, ["A"], strict=False)
        assert result == 0

    def test_tiebreak_v_on_wins_over_v_in_count(self):
        """Same v_hits; A has more v_on → A wins via TB1."""
        # v_hits(A) = 8, v_hits(B) = 8; A v_on=8, B v_on=3
        counts_a = BlockSolidCounts(0, 0, 9, 0, 8, 0)   # v_on=8
        counts_b = BlockSolidCounts(0, 0, 9, 5, 3, 0)   # v_on=3, v_in=5

        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(
            _FakeBlock(), [counts_a, counts_b], ["A", "B"], strict=False
        )
        assert result == 0  # A wins via v_on tiebreak

    def test_tiebreak_v_in_resolves_when_v_on_equal(self):
        """Three solids where two are eliminated by the v_hits gate, leaving A.

        Note: a pure v_in tie-break (Stage-A TB2) is unreachable for two solids
        by construction -- equal v_hits and equal v_on force equal v_in -- so
        this exercises the realistic path where only A survives the v_hits
        comparison.
        """
        class _FakeBlock:
            name = "b0"

        cnt_a = BlockSolidCounts(0, 0, 9, 4, 4, 0)  # v_hits=8, v_on=4, v_in=4
        cnt_b = BlockSolidCounts(0, 0, 9, 2, 4, 2)  # v_hits=6; eliminated by v_hits gate
        cnt_c = BlockSolidCounts(0, 0, 9, 2, 4, 2)  # v_hits=6; eliminated by v_hits gate

        result = _resolve_vertex_dominant(
            _FakeBlock(), [cnt_a, cnt_b, cnt_c], ["A", "B", "C"], strict=False
        )
        assert result == 0  # A is the only candidate with maximum v_hits

    def test_stage_a_strict_unresolvable_raises(self):
        """All tie-breaks exhausted + strict=True → RuntimeError."""
        # Equal v_hits, equal v_on, equal v_in, equal i_in → complete tie
        cnt_a = BlockSolidCounts(5, 0, 4, 4, 4, 0)  # v_hits=8, v_on=4, v_in=4, i_in=5
        cnt_b = BlockSolidCounts(5, 0, 4, 4, 4, 0)  # identical

        class _FakeBlock:
            name = "b0"

        with pytest.raises(RuntimeError, match="Stage-A ambiguous"):
            _resolve_vertex_dominant(
                _FakeBlock(), [cnt_a, cnt_b], ["A", "B"], strict=True
            )

    def test_stage_a_nonstrict_unresolvable_picks_first(self):
        """All tie-breaks exhausted + strict=False → first candidate, no exception."""
        cnt_a = BlockSolidCounts(5, 0, 4, 4, 4, 0)
        cnt_b = BlockSolidCounts(5, 0, 4, 4, 4, 0)

        class _FakeBlock:
            name = "b0"

        result = _resolve_vertex_dominant(
            _FakeBlock(), [cnt_a, cnt_b], ["A", "B"], strict=False
        )
        assert result == 0  # first candidate


# ---------------------------------------------------------------------------
# TestStageBInteriorFallback
# ---------------------------------------------------------------------------

class TestStageBInteriorFallback:
    """Tests that verify Stage B is exercised when Stage A does not activate."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    _RAW_VERTS = frozenset({
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    })

    def test_low_vertex_hits_falls_through_to_stage_b(self, monkeypatch):
        """Only 3 vertex hits (< threshold=5) → Stage A skips → Stage B resolves."""
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS
        vertex_count = {"n": 0}

        def _fake_classify(solid, point, tol):
            if point in raw_verts:
                if solid is SOLID_A:
                    vertex_count["n"] += 1
                    # Only 3 vertices IN for A (< threshold=5)
                    return STATE_IN if vertex_count["n"] <= 3 else STATE_OUT
                return STATE_OUT
            # Interior: SOLID_A wins clearly
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        # Stage A: max v_hits=3 < 5 → skip; Stage B: A wins via interior majority
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_stage_b_perturbation_still_works(self, monkeypatch):
        """Stage B perturbation resolves ON-tie when Stage A is inactive."""
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS
        interior_pass_done = {"v": False}
        interior_points_seen = {"n": 0}

        def _fake_classify(solid, point, tol):
            # Raw vertices → OUT → Stage A inactive
            if point in raw_verts:
                return STATE_OUT
            if not interior_pass_done["v"]:
                interior_points_seen["n"] += 1
                if interior_points_seen["n"] >= 18:  # 9 interior x 2 solids
                    interior_pass_done["v"] = True
                return STATE_ON  # all interior ON → perturbation needed
            # Perturbation pass: SOLID_A wins
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_stage_b_no_match_returns_unzoned_in_strict(self, monkeypatch):
        """All 17 samples OUT → unzoned, NO RuntimeError even with strict=True."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid")]
        # strict=True must NOT raise for no-match case
        mapping = assign_cell_zones(bmd, pairs, strict=True)
        assert "b0" not in mapping
        assert block.zone is None


# ---------------------------------------------------------------------------
# TestThinAnnulusRegression
# ---------------------------------------------------------------------------

class TestThinAnnulusRegression:
    """End-to-end regression tests for the thin-annulus scenario."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    _RAW_VERTS = frozenset({
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    })

    def test_thin_annulus_block_assigned_to_annulus_not_inner_cylinder(
        self, monkeypatch
    ):
        """Thin-annulus block: vertices ON annulus, interior samples IN inner_cylinder.

        Expected: Stage A activates → block assigned to annulus (not inner_cylinder).
        """
        bmd, block = _make_unit_cube_bmd("b0")
        raw_verts = self._RAW_VERTS

        def _fake_classify(solid, point, tol):
            if point in raw_verts:
                # Annulus solid: vertices ON (they sit on the curved boundary)
                # Inner cylinder solid: vertices OUT
                return STATE_ON if solid is SOLID_A else STATE_OUT
            else:
                # Interior samples (centroid + insets) are IN the inner cylinder
                # due to sagitta — the arc-chord effect pushes them inside
                return STATE_IN if solid is SOLID_B else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "annulus"), (SOLID_B, "inner_cylinder")]
        mapping = assign_cell_zones(bmd, pairs)
        # Stage A: annulus v_hits=8 (v_on=8) vs inner_cylinder v_hits=0 → annulus wins
        assert mapping == {"b0": "annulus"}
        assert block.zone == "annulus"

    def test_outer_shell_block_case6(self, monkeypatch):
        """All 9 interior IN + 8 vertex IN → SOLID_A wins via Stage A immediately."""
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "outer_shell"), (SOLID_B, "inner")]
        mapping = assign_cell_zones(bmd, pairs)
        assert mapping == {"b0": "outer_shell"}
        assert block.zone == "outer_shell"


# ---------------------------------------------------------------------------
# TestAssignCellZonesAABBFilter
# ---------------------------------------------------------------------------

class TestAssignCellZonesAABBFilter:
    """Tests for the AABB pre-filter in assign_cell_zones."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_default_uses_aabb_filter(self, monkeypatch):
        """Default path (use_aabb_filter=True) produces correct zone assignment.

        AABB filter may be disabled silently when OCC is unavailable (fallback),
        but the result must still be correct.
        """
        bmd, block = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, use_aabb_filter=True)
        assert mapping == {"b0": "fluid"}
        assert block.zone == "fluid"

    def test_disabled_aabb_filter_same_result(self, monkeypatch):
        """use_aabb_filter=False must produce the same zone mapping as the default."""
        bmd_on, _block_on = _make_unit_cube_bmd("b0")
        bmd_off, _block_off = _make_unit_cube_bmd("b0")

        def _fake_classify(solid, point, tol):
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]

        mapping_on = assign_cell_zones(bmd_on, pairs, use_aabb_filter=True)
        mapping_off = assign_cell_zones(bmd_off, pairs, use_aabb_filter=False)
        assert mapping_on == mapping_off

    def test_aabb_filter_skips_classifier_for_disjoint_solid(self, monkeypatch):
        """When solid AABB is disjoint from the block, classify_point_in_solid must
        not be called for that solid.

        We mock compute_solid_aabb to return a far-away AABB for SOLID_B so that
        it never overlaps the unit-cube block AABB, then verify SOLID_B is not
        classified.
        """

        bmd, _block = _make_unit_cube_bmd("b0")

        classify_calls: dict = {id(SOLID_A): 0, id(SOLID_B): 0}

        def _fake_classify(solid, point, tol):
            classify_calls[id(solid)] = classify_calls.get(id(solid), 0) + 1
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)

        # Return a far-away AABB for SOLID_B (100..200 range, far from unit cube)
        def _fake_compute_solid_aabb(solid, tol):
            if solid is SOLID_B:
                return (100.0, 100.0, 100.0, 200.0, 200.0, 200.0)
            # SOLID_A: return a normal AABB that overlaps the unit cube
            return (-0.1, -0.1, -0.1, 1.1, 1.1, 1.1)

        monkeypatch.setattr(cz_core_mod, "compute_solid_aabb", _fake_compute_solid_aabb)

        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        mapping = assign_cell_zones(bmd, pairs, use_aabb_filter=True)

        # SOLID_A must still be classified (AABB overlaps)
        assert classify_calls.get(id(SOLID_A), 0) > 0
        # SOLID_B must not be classified at all (AABB disjoint)
        assert classify_calls.get(id(SOLID_B), 0) == 0
        assert mapping == {"b0": "fluid"}

    def test_aabb_overlapping_still_classifies(self, monkeypatch):
        """When solid AABB overlaps the block AABB, classification must run normally."""

        bmd, _block = _make_unit_cube_bmd("b0")
        classify_calls: dict = {id(SOLID_A): 0}

        def _fake_classify(solid, point, tol):
            classify_calls[id(solid)] = classify_calls.get(id(solid), 0) + 1
            return STATE_IN if solid is SOLID_A else STATE_OUT

        self._patch_classify(monkeypatch, _fake_classify)

        # Return an overlapping AABB for SOLID_A
        def _fake_compute_solid_aabb(solid, tol):
            return (-0.1, -0.1, -0.1, 1.1, 1.1, 1.1)

        monkeypatch.setattr(cz_core_mod, "compute_solid_aabb", _fake_compute_solid_aabb)

        pairs = [(SOLID_A, "fluid")]
        mapping = assign_cell_zones(bmd, pairs, use_aabb_filter=True)
        # SOLID_A must have been classified (AABB overlaps)
        assert classify_calls.get(id(SOLID_A), 0) > 0
        assert mapping == {"b0": "fluid"}


# ---------------------------------------------------------------------------
# TestAssignCellZonesClassifierCache
# ---------------------------------------------------------------------------

class TestAssignCellZonesClassifierCache:
    """Tests for the per-solid classifier cache in assign_cell_zones."""

    @staticmethod
    def _patch_classify(monkeypatch, side_effect):
        import meshing_utils.geometry.containment as cont_mod
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", side_effect)
        monkeypatch.setattr(cont_mod, "classify_point_in_solid", side_effect)

    def test_make_solid_classifier_called_once_per_solid(self, monkeypatch):
        """make_solid_classifier must be called exactly M times (once per solid),
        regardless of the number of blocks N."""

        # Build a BMD with 2 blocks
        bmd = BlockMeshDict()
        offset_pairs = [(0.0, "b0"), (2.0, "b1")]
        for off, bname in offset_pairs:
            for _i, (name, c) in enumerate([
                (f"{bname}_v0", [0.0 + off, 0.0, 0.0]),
                (f"{bname}_v1", [1.0 + off, 0.0, 0.0]),
                (f"{bname}_v2", [1.0 + off, 1.0, 0.0]),
                (f"{bname}_v3", [0.0 + off, 1.0, 0.0]),
                (f"{bname}_v4", [0.0 + off, 0.0, 1.0]),
                (f"{bname}_v5", [1.0 + off, 0.0, 1.0]),
                (f"{bname}_v6", [1.0 + off, 1.0, 1.0]),
                (f"{bname}_v7", [0.0 + off, 1.0, 1.0]),
            ]):
                bmd.vertices.add(Vertex(name, c))
            block = Block(
                bname,
                vertices=[
                    f"{bname}_v0", f"{bname}_v1", f"{bname}_v2", f"{bname}_v3",
                    f"{bname}_v4", f"{bname}_v5", f"{bname}_v6", f"{bname}_v7",
                ],
                cells=[1, 1, 1],
            )
            bmd.blocks.add(block)

        make_classifier_calls = {"n": 0}
        fake_classifier = object()  # sentinel

        def _fake_make_solid_classifier(solid):
            make_classifier_calls["n"] += 1
            return fake_classifier

        monkeypatch.setattr(cz_core_mod, "make_solid_classifier", _fake_make_solid_classifier)

        def _fake_classify_with_classifier(classifier, point, tol):
            return STATE_OUT

        monkeypatch.setattr(
            cz_classification_mod, "classify_point_with_classifier", _fake_classify_with_classifier
        )

        # 2 blocks, 2 solids → make_solid_classifier must be called exactly 2 times
        pairs = [(SOLID_A, "fluid"), (SOLID_B, "solid")]
        assign_cell_zones(bmd, pairs, use_aabb_filter=False)
        assert make_classifier_calls["n"] == 2

    def test_classifier_passed_to_classification_function(self, monkeypatch):
        """When a classifier is available, classify_point_with_classifier must be
        called instead of classify_point_in_solid."""

        bmd, _block = _make_unit_cube_bmd("b0")
        fake_clf = object()

        def _fake_make_solid_classifier(solid):
            return fake_clf

        monkeypatch.setattr(cz_core_mod, "make_solid_classifier", _fake_make_solid_classifier)

        classify_with_calls = {"n": 0}
        classify_direct_calls = {"n": 0}

        def _fake_classify_with_classifier(clf, point, tol):
            assert clf is fake_clf, "Wrong classifier object passed"
            classify_with_calls["n"] += 1
            return STATE_IN if classify_with_calls["n"] <= 9 else STATE_OUT

        def _fake_classify_direct(solid, point, tol):
            classify_direct_calls["n"] += 1
            return STATE_OUT

        monkeypatch.setattr(
            cz_classification_mod, "classify_point_with_classifier", _fake_classify_with_classifier
        )
        monkeypatch.setattr(cz_classification_mod, "classify_point_in_solid", _fake_classify_direct)

        pairs = [(SOLID_A, "fluid")]
        assign_cell_zones(bmd, pairs, use_aabb_filter=False)

        # classify_point_with_classifier must have been used
        assert classify_with_calls["n"] > 0
        # classify_point_in_solid (direct) must NOT have been used
        assert classify_direct_calls["n"] == 0


# ---------------------------------------------------------------------------
# TestFailFastOnOCCFailure
# ---------------------------------------------------------------------------

class TestFailFastOnOCCFailure:
    """Verify that exceptions from compute_solid_aabb and make_solid_classifier
    propagate to the caller instead of being silently swallowed."""

    @pytest.mark.no_mock_occ
    def test_compute_solid_aabb_exception_propagates(self, monkeypatch):
        """compute_solid_aabb raising must propagate when use_aabb_filter=True."""
        monkeypatch.setattr(
            "meshing_utils.operations.cell_zones.core.make_solid_classifier",
            lambda s: None,
        )

        def _raise(solid, tol):
            raise RuntimeError("forced AABB failure")

        monkeypatch.setattr(
            "meshing_utils.operations.cell_zones.core.compute_solid_aabb",
            _raise,
        )
        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "myzone")]
        with pytest.raises(RuntimeError, match="forced AABB failure"):
            assign_cell_zones(bmd, pairs, use_aabb_filter=True)

    @pytest.mark.no_mock_occ
    def test_make_solid_classifier_exception_propagates(self, monkeypatch):
        """make_solid_classifier raising must propagate regardless of AABB filter."""
        monkeypatch.setattr(
            "meshing_utils.operations.cell_zones.core.compute_solid_aabb",
            lambda s, t: (-1e10, -1e10, -1e10, 1e10, 1e10, 1e10),
        )

        def _raise(solid):
            raise RuntimeError("forced classifier failure")

        monkeypatch.setattr(
            "meshing_utils.operations.cell_zones.core.make_solid_classifier",
            _raise,
        )
        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "myzone")]
        with pytest.raises(RuntimeError, match="forced classifier failure"):
            assign_cell_zones(bmd, pairs, use_aabb_filter=True)

    @pytest.mark.no_mock_occ
    def test_classifier_exception_propagates_when_aabb_filter_disabled(self, monkeypatch):
        """Even with use_aabb_filter=False, classifier-build still runs and must fail-fast."""

        def _raise(solid):
            raise RuntimeError("forced classifier failure")

        monkeypatch.setattr(
            "meshing_utils.operations.cell_zones.core.make_solid_classifier",
            _raise,
        )
        bmd, _ = _make_unit_cube_bmd("b0")
        pairs = [(SOLID_A, "myzone")]
        with pytest.raises(RuntimeError, match="forced classifier failure"):
            assign_cell_zones(bmd, pairs, use_aabb_filter=False)
