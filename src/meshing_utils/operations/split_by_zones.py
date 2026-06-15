"""Split a BlockMeshDict into multiple files grouped by block zone.

Each output file contains the subset of blocks belonging to one zone (or a
collected ``_rest`` / ``_no_zone`` bucket) together with all vertices, edges,
and patch faces that are exclusively referenced by those blocks.

Public API
----------
split_blockmeshdict_by_zones(bmd, output_dir, ...)
"""

from __future__ import annotations

import copy
from pathlib import Path

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import (
    Block,
    Blocks,
    Boundary,
    Edge,
    Edges,
    Face,
    Patch,
    Vertex,
    Vertices,
)
from meshing_utils.operations.cell_zones.naming import _sanitize_zone_name

# ---------------------------------------------------------------------------
# Reserved bucket names that must not clash with real zone names
# ---------------------------------------------------------------------------

_RESERVED_NAMES = frozenset({"no_zone", "rest"})

# Internal sentinel used as dict key for blocks that carry no zone
_NO_ZONE_KEY = "no_zone"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_blocks_by_zone(bmd: BlockMeshDict) -> dict[str, list[Block]]:
    """Return a mapping from zone bucket key to list of Block objects.

    Blocks whose ``zone`` attribute is ``None`` or an empty string are placed
    in the ``"no_zone"`` bucket.

    Parameters
    ----------
    bmd:
        Source :class:`~meshing_utils.foam.dict_file.BlockMeshDict`.

    Returns
    -------
    dict[str, list[Block]]
        Ordered dict preserving insertion order; ``"no_zone"`` key is only
        present when at least one un-zoned block exists.

    Raises
    ------
    ValueError
        When any block carries a zone name that matches a reserved bucket name
        (``"no_zone"`` or ``"rest"``).
    """
    groups: dict[str, list[Block]] = {}
    for block in bmd.blocks:
        zone = block.zone if block.zone else None
        if zone is not None and zone in _RESERVED_NAMES:
            raise ValueError(
                f"Block {block.name!r} has zone name {zone!r}, which is a "
                f"reserved bucket name. Reserved names: {sorted(_RESERVED_NAMES)}"
            )
        key = zone if zone is not None else _NO_ZONE_KEY
        groups.setdefault(key, []).append(block)
    return groups


def _collect_vertex_names(blocks: list[Block]) -> set[str]:
    """Return the union of all vertex names referenced by the given blocks."""
    names: set[str] = set()
    for block in blocks:
        names.update(block.vertices)
    return names


def _build_subset_bmd(
    source: BlockMeshDict,
    blocks: list[Block],
    reindex_vertices: bool,
    keep_empty_patches: bool,
) -> BlockMeshDict:
    """Construct a new :class:`BlockMeshDict` containing only the given blocks.

    The new BMD contains:

    - All vertices referenced by *blocks* (in original order).
    - All edges whose both endpoint vertices are in the vertex subset.
    - All patches, filtered to faces whose 4 vertices are all in the subset;
      empty patches are dropped unless *keep_empty_patches* is ``True``.
    - ``convertToMeters``, ``geometry_body``, and ``default_patch`` copied
      verbatim from *source* (deep-copied where mutable).

    When *reindex_vertices* is ``True``, vertices are compactly renamed
    ``v0 … vN-1`` and all references in blocks, edges, and faces are remapped
    consistently.

    Parameters
    ----------
    source:
        Original :class:`~meshing_utils.foam.dict_file.BlockMeshDict`.
    blocks:
        The subset of blocks for this output file.  Must not be empty.
    reindex_vertices:
        Whether to rename vertices to a compact ``v0…vN-1`` sequence.
    keep_empty_patches:
        Whether to keep patches that end up with zero faces after filtering.

    Returns
    -------
    BlockMeshDict
        A new, independent BMD (no shared state with *source*).
    """
    vertex_name_set = _collect_vertex_names(blocks)

    # Collect vertices in original order (preserves spatial layout)
    subset_vertices: list[Vertex] = [
        copy.deepcopy(v) for v in source.vertices if v.name in vertex_name_set
    ]

    # Build rename map for reindex mode
    rename: dict[str, str] = {}
    if reindex_vertices:
        for new_idx, v in enumerate(subset_vertices):
            new_name = f"v{new_idx}"
            rename[v.name] = new_name
            v.name = new_name

    def _remap(name: str) -> str:
        return rename.get(name, name)

    # Deep-copy and optionally remap blocks
    subset_blocks: list[Block] = []
    for block in blocks:
        b = copy.deepcopy(block)
        if reindex_vertices:
            b.vertices = [_remap(vn) for vn in b.vertices]
        subset_blocks.append(b)

    # Filter edges: both endpoints must be in the subset vertex set
    effective_vertex_names = {_remap(n) for n in vertex_name_set}
    subset_edges: list[Edge] = []
    for edge in source.edges:
        e_start = _remap(edge.v_start) if reindex_vertices else edge.v_start
        e_end = _remap(edge.v_end) if reindex_vertices else edge.v_end
        if e_start in effective_vertex_names and e_end in effective_vertex_names:
            e = copy.deepcopy(edge)
            if reindex_vertices:
                e.v_start = e_start
                e.v_end = e_end
            subset_edges.append(e)

    # Filter patches and faces
    subset_patches: list[Patch] = []
    for patch in source.boundary:
        filtered_faces: list[Face] = []
        for face in patch.faces:
            remapped_face_verts = (
                [_remap(vn) for vn in face.vertices]
                if reindex_vertices
                else face.vertices
            )
            if all(vn in effective_vertex_names for vn in remapped_face_verts):
                f = copy.deepcopy(face)
                if reindex_vertices:
                    f.vertices = remapped_face_verts
                filtered_faces.append(f)
        if filtered_faces or keep_empty_patches:
            p = copy.deepcopy(patch)
            p.faces = filtered_faces
            subset_patches.append(p)

    # Assemble the new BMD
    out = BlockMeshDict()
    out.convertToMeters = source.convertToMeters
    out.geometry_body = source.geometry_body  # immutable string — no deepcopy needed
    out.default_patch = copy.deepcopy(source.default_patch)
    out.vertices = Vertices(subset_vertices)
    out.edges = Edges(subset_edges)
    out.blocks = Blocks(subset_blocks)
    out.boundary = Boundary(subset_patches)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_blockmeshdict_by_zones(
    bmd: BlockMeshDict,
    output_dir: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    reindex_vertices: bool = False,
    keep_empty_patches: bool = False,
) -> list[Path]:
    """Split *bmd* into per-zone ``blockMeshDict`` files in *output_dir*.

    Parameters
    ----------
    bmd:
        Source :class:`~meshing_utils.foam.dict_file.BlockMeshDict`.  Must
        not be mutated by this function.
    output_dir:
        Directory where the split files are written.  Created automatically
        if it does not yet exist.
    include:
        When given, only these zone names get their own output file.  All
        other blocks (including un-zoned ones) are collected into
        ``blockMeshDict_rest``.  Mutually exclusive with *exclude*.
    exclude:
        When given, these zone names are excluded from their own files and
        collected into ``blockMeshDict_rest``.  All other zone buckets get
        their own file.  Mutually exclusive with *include*.
    reindex_vertices:
        When ``True``, vertices in each output file are renamed to a compact
        ``v0 … vN-1`` sequence and all references are updated consistently.
    keep_empty_patches:
        When ``True``, patches that end up with zero faces are still written
        to the output files.

    Returns
    -------
    list[Path]
        Sorted list of paths that were written.

    Raises
    ------
    ValueError
        - When both *include* and *exclude* are given.
        - When *include* or *exclude* contains a zone name not present in *bmd*.
        - When any block carries a zone name equal to a reserved bucket name
          (``"no_zone"`` or ``"rest"``).
    """
    if include is not None and exclude is not None:
        raise ValueError(
            "--include and --exclude are mutually exclusive; provide at most one."
        )

    output_dir = Path(output_dir)

    # Group all blocks by zone
    zone_groups = _group_blocks_by_zone(bmd)
    available_zones = set(zone_groups.keys())

    # Validate include/exclude against known zones
    if include is not None:
        # Validate against real zone names (not the internal _NO_ZONE_KEY)
        real_zones = available_zones - {_NO_ZONE_KEY}
        unknown = set(include) - real_zones
        if unknown:
            raise ValueError(
                f"Unknown zone(s) in --include: {sorted(unknown)}. "
                f"Available zones: {sorted(real_zones)}"
            )

    if exclude is not None:
        real_zones = available_zones - {_NO_ZONE_KEY}
        unknown = set(exclude) - real_zones
        if unknown:
            raise ValueError(
                f"Unknown zone(s) in --exclude: {sorted(unknown)}. "
                f"Available zones: {sorted(real_zones)}"
            )

    # Determine which buckets get their own file and which go to _rest
    # Returns: list of (output_suffix, list_of_blocks)
    output_specs: list[tuple[str, list[Block]]] = []

    if include is None and exclude is None:
        # Default mode: one file per bucket
        for key, blocks in zone_groups.items():
            output_specs.append((key, blocks))

    elif include is not None:
        # Named zones get own files; everything else → _rest
        rest_blocks: list[Block] = []
        for key, blocks in zone_groups.items():
            if key in include:
                output_specs.append((key, blocks))
            else:
                rest_blocks.extend(blocks)
        if rest_blocks:
            output_specs.append(("rest", rest_blocks))

    else:  # exclude is not None
        # All non-excluded buckets get own files; excluded → _rest
        rest_blocks = []
        for key, blocks in zone_groups.items():
            if key in exclude:
                rest_blocks.extend(blocks)
            else:
                output_specs.append((key, blocks))
        if rest_blocks:
            output_specs.append(("rest", rest_blocks))

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Track suffixes already used so distinct zones that sanitise to the same
    # identifier (e.g. ``inlet-1`` and ``inlet.1`` both → ``inlet_1``) get
    # deterministic numeric suffixes (``_2``, ``_3``, …) instead of silently
    # overwriting one another.
    used_suffixes: set[str] = set()

    for idx, (suffix, blocks) in enumerate(output_specs):
        subset_bmd = _build_subset_bmd(
            bmd, blocks, reindex_vertices, keep_empty_patches
        )
        if suffix in _RESERVED_NAMES:
            # Reserved buckets are already valid identifiers and must stay
            # verbatim; they are guaranteed unique by construction.
            safe_suffix = suffix
        else:
            base = _sanitize_zone_name(suffix, idx)
            safe_suffix = base
            counter = 2
            while safe_suffix in used_suffixes:
                safe_suffix = f"{base}_{counter}"
                counter += 1
        used_suffixes.add(safe_suffix)

        out_path = output_dir / f"blockMeshDict_{safe_suffix}"
        resolved = out_path.resolve()
        if output_dir.resolve() not in resolved.parents:
            raise ValueError(f"Refusing to write outside output dir: {resolved}")
        subset_bmd.write(out_path)
        written.append(out_path)

    return sorted(written)
