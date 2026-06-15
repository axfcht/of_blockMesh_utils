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
from meshing_utils.operations.revolve import RevolveConfig, revolve
from meshing_utils.operations.scale import scale, validate_factors

# stp_pipeline is intentionally NOT imported here to avoid pulling the
# heavy OCP-dependent code path eagerly. Import it directly from
# meshing_utils.operations.stp_pipeline when needed.

__all__ = [
    "AmbiguousFaceError",
    "ExtrusionError",
    "LayerStep",
    "NoMarkersFoundError",
    "RevolveConfig",
    "extrude",
    "extrude_with_steps",
    "parse_layer_steps",
    "parse_offsets",
    "revolve",
    "scale",
    "validate_factors",
]
