"""OpenFOAM dictionary file header / footer templates.

Centralises the textual boilerplate that wraps every OpenFOAM dict file
so that a future second dict type (``meshQualityDict``, ``decomposeParDict``,
...) can render its own header without duplicating the banner.
"""

from __future__ import annotations

_HEADER_TEMPLATE = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
\tformat      ascii;
\tclass       {foam_class};
\tobject      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""

FOOTER = "// ************************************************************************* //"


def render_header(object_name: str, foam_class: str = "dictionary") -> str:
    """Return the OpenFOAM v13 file header for the given dict file.

    ``object_name`` is the ``FoamFile.object`` value (e.g.
    ``"blockMeshDict"``); ``foam_class`` is the ``FoamFile.class`` value
    (``"dictionary"`` by default).
    """
    return _HEADER_TEMPLATE.format(object_name=object_name, foam_class=foam_class)


# Backwards-compatible: BlockMeshDict.write() historically appended a
# pre-rendered constant for the blockMeshDict header.
BLOCKMESHDICT_HEADER = render_header("blockMeshDict")
