# Public API for meshing_utils
# Re-exports all publicly used symbols from submodules.

# --- exceptions ---
# --- foam/elements (replaces common/models) ---
# --- face_geometry (migrated to cad/face_matching) ---
from meshing_utils.cad.face_matching import (
    BlockFace,
    compute_outward_normal,
    effective_normal_tolerance,
    extract_block_faces,
    find_dominant_face,
    nearest_face_within_tol,
    normals_consistent,
    surface_type_of,
)

# --- step_loader (migrated to cad/step_loader) ---
from meshing_utils.cad.step_loader import (
    explore_solids,
    find_single_step_file,
    load_solids_with_names,
    load_step_solids,
    read_step_solid_names,
    read_step_unit,
    read_step_xcaf,
)

# --- step_names ---
from meshing_utils.cad.step_names import NamedSolid
from meshing_utils.exceptions import MeshingUtilsError

# --- block_mesh_dict (migrated to foam/dict_file) ---
from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import (
    Block,
    Blocks,
    Boundary,
    DefaultPatch,
    Edge,
    Edges,
    Face,
    Markable,
    Patch,
    Vertex,
    Vertices,
)

# --- cell_count_strategy (migrated to geometry/cell_count) ---
from meshing_utils.geometry.cell_count import (
    CellCountStrategy,
    EuclideanProjectedCellCountStrategy,
    PropagatedCellCountStrategy,
    compute_cell_counts_new,
)

# --- geometry/containment ---
from meshing_utils.geometry.containment import (
    DEFAULT_INSET_FACTOR,
    VERTEX_DOMINANT_THRESHOLD,
    aabbs_overlap,
    bmd_bbox_diagonal,
    classify_point_in_solid,
    classify_point_with_classifier,
    compute_block_centroid,
    compute_block_sample_sets,
    compute_inset_sample_points,
    compute_points_aabb,
    compute_solid_aabb,
    make_solid_classifier,
    point_inside_solid,
    resolve_block_coords,
)

# --- block_axis (migrated to geometry/hex_axes) ---
from meshing_utils.geometry.hex_axes import (
    AxisEquivalenceClasses,
    BlockAxis,
    TopologyError,
)

# --- geometry/hex_topology (replaces common/geom_utils) ---
from meshing_utils.geometry.hex_topology import (
    HEX_FACE_INDICES,
    HEX_FACE_NAMES,
    CurveInfo,
    HexCandidate,
    HexValidationError,
    OrderingConsistencyError,
    PointPool,
    assert_block_face_normals_outward,
    assert_hex_outward_from_coords,
    check_global_face_consistency,
    enforce_openfoam_face_convention,
    ensure_right_handed,
    order_hex_vertices,
    validate_hex,
)

# --- io/step_text_scan ---
from meshing_utils.io.step_text_scan import (
    BrepSolidEntry,
    build_brep_name_map_by_file_id,
    parse_brep_solid_entries,
)

# --- operations/cell_zones ---
from meshing_utils.operations.cell_zones import (
    SAMPLING_CENTROID,
    SAMPLING_INSET,
    VOTE_MAJORITY,
    BlockSolidCounts,
    assign_cell_zones,
)

# --- operations/combine ---
from meshing_utils.operations.combine import (
    combine_blockmeshdicts,
    discover_source_files,
)

# --- extrusion (migrated to operations/) ---
from meshing_utils.operations.extrusion import (
    AmbiguousFaceError,
    ExtrusionError,
    LayerStep,
    NoMarkersFoundError,
    extrude,
    extrude_with_steps,
    parse_layer_steps,
    parse_offsets,
)

# --- revolve (migrated to operations/) ---
from meshing_utils.operations.revolve import (
    RevolveConfig,
    revolve,
)

# --- scale (migrated to operations/) ---
from meshing_utils.operations.scale import (
    scale,
    validate_factors,
)

# --- operations/split_by_zones ---
from meshing_utils.operations.split_by_zones import split_blockmeshdict_by_zones

# Curated public API surface.
#
# Only the symbols a downstream user is meant to call appear in __all__:
# core data models, high-level operations, and public exception base
# classes. Every other symbol imported above is internal — it remains
# importable via ``from meshing_utils import X`` for transitional reasons
# but may be moved, renamed, or removed without notice. Internal code
# (CLI modules, operations, tests) must import from the defining
# submodule rather than routing through this facade.
#
# Intentional changes to this list must also update
# tests/test_public_api.py in the same commit.
__all__ = [
    "Block",
    # data models
    "BlockMeshDict",
    "Boundary",
    "Edge",
    # public exception base classes
    "ExtrusionError",
    "Face",
    "HexValidationError",
    "MeshingUtilsError",
    "Patch",
    "RevolveConfig",
    "TopologyError",
    "Vertex",
    # high-level operations
    "assign_cell_zones",
    "combine_blockmeshdicts",
    "extrude",
    "extrude_with_steps",
    "revolve",
    "scale",
    "split_blockmeshdict_by_zones",
]
