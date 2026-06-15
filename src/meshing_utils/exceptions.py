"""Root exception type for meshing_utils.

This module has no internal imports so any layer may import it without
creating an upward dependency.
"""
from __future__ import annotations


class MeshingUtilsError(Exception):
    """Base class for all meshing_utils-specific errors."""
