# Re-export all element classes from their individual submodules.
# This is the single source of truth for all FOAM element classes.
from meshing_utils.foam.elements.block import Block, Blocks
from meshing_utils.foam.elements.edge import Edge, Edges
from meshing_utils.foam.elements.face import Face
from meshing_utils.foam.elements.markable import Markable
from meshing_utils.foam.elements.patch import Boundary, DefaultPatch, Patch
from meshing_utils.foam.elements.vertex import Vertex, Vertices

__all__ = [
    "Block",
    "Blocks",
    "Boundary",
    "DefaultPatch",
    "Edge",
    "Edges",
    "Face",
    "Markable",
    "Patch",
    "Vertex",
    "Vertices",
]
