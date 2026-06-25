"""Combine multiple BlockMeshDict fragments into a single BlockMeshDict.

This is the counterpart to :mod:`meshing_utils.operations.split_by_zones`.

Public API
----------
combine_blockmeshdicts(sources, ...)
discover_source_files(system_dir, ...)
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import (
    Edge,
    Patch,
    Vertex,
)
from meshing_utils.operations.stp_pipeline import resolve_block_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_source_files(
    system_dir: Path,
    excludes: list[str] | None = None,
) -> list[Path]:
    """Discover blockMeshDict fragment files in *system_dir* via substring match.

    A file is a candidate when all of the following hold:

    - ``path.is_file()``
    - ``"blockMeshDict"`` is contained in ``path.name``
    - ``path.name != "blockMeshDict"``  (main file is never a source)
    - ``path.name`` is not in *excludes*

    Results are sorted alphabetically by file name.

    Parameters
    ----------
    system_dir:
        Directory to search (typically ``<case-dir>/system/``).
    excludes:
        File names to exclude from discovery.

    Returns
    -------
    list[Path]
        Sorted list of discovered source paths.

    Raises
    ------
    ValueError
        When no source files are found.
    """
    system_dir = Path(system_dir)
    exclude_set: set[str] = set(excludes) if excludes else set()

    candidates: list[Path] = []
    for p in system_dir.iterdir():
        if not p.is_file():
            continue
        if "blockMeshDict" not in p.name:
            continue
        if p.name == "blockMeshDict":
            continue
        if p.name in exclude_set:
            continue
        candidates.append(p)

    if not candidates:
        raise ValueError(
            f"No blockMeshDict fragment files found in {system_dir}. "
            "Expected files whose names contain 'blockMeshDict' "
            "(excluding the main 'blockMeshDict' file itself)."
        )

    return sorted(candidates, key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Internal merge helpers
# ---------------------------------------------------------------------------


def _coord_distance(
    a: list[float],
    b: list[float],
) -> float:
    """Return the Euclidean distance between two 3-D coordinate lists."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=False)))


def resolve_vertex_name(out: BlockMeshDict, preferred: str) -> str:
    """Return a vertex name not already present in *out*.

    Mirrors resolve_block_name's "suffix" policy: returns *preferred*
    when free, else preferred_2, preferred_3, ... Internal helper; not
    part of the public API.
    """
    if preferred not in out.vertices:   # Vertices.__contains__(name)
        return preferred
    counter = 2
    while True:
        candidate = f"{preferred}_{counter}"
        if candidate not in out.vertices:
            return candidate
        counter += 1


def _merge_scalars(
    out: BlockMeshDict,
    src: BlockMeshDict,
    label: str,
    strict: bool,
) -> None:
    """Merge scalar fields (convertToMeters, geometry_body, default_patch) from *src* into *out*.

    First-source wins.  Conflicts are logged as warnings (or raised when *strict*
    is ``True``).

    Parameters
    ----------
    out:
        Target BlockMeshDict (mutated in place).
    src:
        Source BlockMeshDict to merge from.
    label:
        Human-readable label for diagnostic messages (typically the file name).
    strict:
        If ``True``, conflicts raise ``ValueError`` instead of being warned.
    """
    # convertToMeters — compare by value
    if abs(out.convertToMeters - src.convertToMeters) > 1e-12:
        msg = (
            f"[{label}] convertToMeters conflict: "
            f"output has {out.convertToMeters}, source has {src.convertToMeters}. "
            "Keeping existing value."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    # geometry_body — compare by string equality (stripped)
    out_geo = out.geometry_body.strip()
    src_geo = src.geometry_body.strip()
    if src_geo and out_geo != src_geo:
        if not out_geo:
            out.geometry_body = src.geometry_body
        else:
            msg = (
                f"[{label}] geometry_body conflict: output already has geometry content "
                "that differs from source. Keeping existing value."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

    # default_patch — compare name + type
    if src.default_patch.name or src.default_patch.type:
        out_name = out.default_patch.name
        out_type = out.default_patch.type
        src_name = src.default_patch.name
        src_type = src.default_patch.type
        if out_name != src_name or out_type != src_type:
            # Only warn if the output already has non-default values
            default_name = "defaultFaces"
            default_type = "empty"
            if out_name != default_name or out_type != default_type:
                msg = (
                    f"[{label}] default_patch conflict: "
                    f"output has name={out_name!r} type={out_type!r}, "
                    f"source has name={src_name!r} type={src_type!r}. "
                    "Keeping existing value."
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
            else:
                # Output still has defaults — adopt source values
                out.default_patch.name = src_name
                out.default_patch.type = src_type


def _merge_vertices(
    out: BlockMeshDict,
    src: BlockMeshDict,
    label: str,
    vertex_tol: float,
) -> dict[str, str]:
    """Merge vertices from *src* into *out*, returning a name remap table.

    Rules (per incoming vertex ``v``):

    R4 — Same name, coordinate difference ≤ *vertex_tol* → identity remap,
         skip add (duplicate).
    R2 — Same name, coords differ, but the incoming coordinate matches an
         existing vertex under a *different* name → collapse: remap to that
         existing name, do not add.
    R3 — Same name, coords differ, no coordinate match in output → unique
         rename via ``resolve_vertex_name`` (``<name>_2``, ``_3``, ...);
         add a deep copy with the new name.
    R5 — New name → add deep copy as-is; identity remap entry so downstream
         lookups never KeyError.

    Parameters
    ----------
    out:
        Target BlockMeshDict (mutated in place).
    src:
        Source BlockMeshDict to merge from.
    label:
        Human-readable label for diagnostic messages.
    vertex_tol:
        Tolerance for coordinate conflict detection (Euclidean distance).

    Returns
    -------
    dict[str, str]
        Mapping from every incoming vertex name to the name it resolves to
        in the output.  Identity where unchanged.
    """
    existing_by_name: dict[str, Vertex] = {v.name: v for v in out.vertices}
    remap: dict[str, str] = {}
    for v in src.vertices:
        if v.name in existing_by_name:
            existing_v = existing_by_name[v.name]
            if _coord_distance(existing_v.coords, v.coords) <= vertex_tol:
                # Same name + same coords -> dedup (R4)
                remap[v.name] = v.name
                logger.debug("Skipping duplicate vertex %r from %s", v.name, label)
                continue
            # Name collision + differing coords:
            # 1) does this coordinate already exist under a DIFFERENT name?
            match = None
            for ov in out.vertices:
                if ov.name == v.name:
                    continue
                if _coord_distance(ov.coords, v.coords) <= vertex_tol:
                    match = ov
                    break
            if match is not None:
                # Collapse onto existing vertex (R2)
                remap[v.name] = match.name
                logger.info(
                    "[%s] Vertex %r collapsed onto existing vertex %r (coords match within tol)",
                    label, v.name, match.name,
                )
                continue
            # 2) genuinely new coordinate -> unique rename (R3)
            new_name = resolve_vertex_name(out, v.name)
            new_v = copy.deepcopy(v)
            new_v.name = new_name
            out.vertices.add(new_v)
            existing_by_name[new_name] = new_v
            remap[v.name] = new_name
            logger.info(
                "[%s] Vertex name collision: %r renamed -> %r (coords differ)",
                label, v.name, new_name,
            )
        else:
            new_v = copy.deepcopy(v)
            out.vertices.add(new_v)
            existing_by_name[v.name] = new_v
            remap[v.name] = v.name   # identity (R5 uniformity)
    return remap


def _edges_equal(a: Edge, b: Edge) -> bool:
    """Return ``True`` when two edges are geometrically identical.

    Edges are equal when they have the same type and the same support points
    (within floating-point representation).  The vertex order is compared
    after normalisation.
    """
    if a.type != b.type:
        return False
    if len(a.points) != len(b.points):
        return False
    # Compare point lists element by element
    for pa, pb in zip(a.points, b.points, strict=False):
        if len(pa) != len(pb):
            return False
        if any(abs(x - y) > 1e-12 for x, y in zip(pa, pb, strict=False)):
            return False
    return True


def _merge_edges(
    out: BlockMeshDict,
    src: BlockMeshDict,
    label: str,
    strict: bool,
    vertex_remap: dict[str, str],
) -> None:
    """Merge edges from *src* into *out*.

    Uses an unordered vertex-pair key for lookup.

    - Same endpoints + same type + same geometry → skip (duplicate).
    - Same endpoints + different type or geometry → warning (existing kept) or
      ``ValueError`` if *strict*.
    - New endpoints → deepcopy into output.

    Parameters
    ----------
    out:
        Target BlockMeshDict (mutated in place).
    src:
        Source BlockMeshDict to merge from.
    label:
        Human-readable label for diagnostic messages.
    strict:
        If ``True``, conflicts raise ``ValueError``.
    vertex_remap:
        Name remap table returned by ``_merge_vertices`` for this source.
        Applied to edge endpoints before dedup/conflict detection.
    """
    # Build lookup: frozenset({v_start, v_end}) -> Edge
    existing_edges: dict[frozenset[str], Edge] = {}
    for e in out.edges:
        key = frozenset({e.v_start, e.v_end})
        existing_edges[key] = e

    for e in src.edges:
        new_edge = copy.deepcopy(e)
        new_edge.v_start = vertex_remap.get(new_edge.v_start, new_edge.v_start)
        new_edge.v_end = vertex_remap.get(new_edge.v_end, new_edge.v_end)
        key = frozenset({new_edge.v_start, new_edge.v_end})
        if key in existing_edges:
            existing = existing_edges[key]
            if _edges_equal(existing, new_edge):
                logger.debug(
                    "Skipping duplicate edge (%s, %s) from %s",
                    new_edge.v_start, new_edge.v_end, label,
                )
            else:
                msg = (
                    f"[{label}] Edge conflict at ({new_edge.v_start}, {new_edge.v_end}): "
                    f"existing type={existing.type!r}, new type={new_edge.type!r}. "
                    "Keeping existing edge."
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
        else:
            out.edges.add(new_edge)
            existing_edges[key] = new_edge


def _merge_blocks(
    out: BlockMeshDict,
    src: BlockMeshDict,
    label: str,
    combine_cell_zones: str | None,
    vertex_remap: dict[str, str],
) -> None:
    """Merge blocks from *src* into *out*.

    Block name collisions are resolved via ``resolve_block_name`` (policy:
    ``"suffix"``).  Vertex references are remapped via *vertex_remap*
    (collapse/rename from ``_merge_vertices``).

    When *combine_cell_zones* is not ``None``, every merged block's zone is
    overridden to that value (including blocks whose original zone was ``None``).

    Parameters
    ----------
    out:
        Target BlockMeshDict (mutated in place).
    src:
        Source BlockMeshDict to merge from.
    label:
        Human-readable label for diagnostic messages.
    combine_cell_zones:
        Zone name to assign to ALL merged blocks, or ``None`` to preserve
        each block's own zone.
    vertex_remap:
        Name remap table returned by ``_merge_vertices`` for this source.
        Applied to block vertex references before adding to output.
    """
    for block in src.blocks:
        new_block = copy.deepcopy(block)
        new_block.vertices = [vertex_remap.get(n, n) for n in new_block.vertices]

        # Resolve name collision
        resolved_name = resolve_block_name(out, new_block.name, policy="suffix")
        if resolved_name != new_block.name:
            logger.info(
                "[%s] Block name collision: renamed %r -> %r",
                label, new_block.name, resolved_name,
            )
            new_block.name = resolved_name

        # Override zone if requested
        if combine_cell_zones is not None:
            new_block.zone = combine_cell_zones

        out.blocks.add(new_block)


def _merge_patches(
    out: BlockMeshDict,
    src: BlockMeshDict,
    label: str,
    strict: bool,
    vertex_remap: dict[str, str],
) -> None:
    """Merge patches and faces from *src* into *out*.

    Rules:

    - Patches with the same name are merged: faces are unioned and deduplicated.
    - Type conflict on a same-named patch → warning (first wins) or ``ValueError``
      if *strict*.
    - A face already present in *any* output patch → skip (if in same patch) or
      warning (if in different patch) or ``ValueError`` if *strict*.

    Face identity is determined by ``tuple(sorted(face.vertices))``.

    Parameters
    ----------
    out:
        Target BlockMeshDict (mutated in place).
    src:
        Source BlockMeshDict to merge from.
    label:
        Human-readable label for diagnostic messages.
    strict:
        If ``True``, conflicts raise ``ValueError``.
    vertex_remap:
        Name remap table returned by ``_merge_vertices`` for this source.
        Applied to face vertex lists before dedup/conflict detection.
    """
    # Build global face set across ALL output patches: face_key -> patch_name
    global_face_registry: dict[tuple[str, ...], str] = {}
    for p in out.boundary:
        for f in p.faces:
            fkey = tuple(sorted(f.vertices))
            global_face_registry[fkey] = p.name

    for src_patch in src.boundary:
        # Find or create the target patch in output
        existing_patch: Patch | None = None
        for p in out.boundary:
            if p.name == src_patch.name:
                existing_patch = p
                break

        if existing_patch is None:
            # New patch — add it with all its faces (deduplicated against global set)
            new_patch = Patch(src_patch.name, type=src_patch.type, faces=[])
            for f in src_patch.faces:
                new_face = copy.deepcopy(f)
                new_face.vertices = [vertex_remap.get(n, n) for n in new_face.vertices]
                fkey = tuple(sorted(new_face.vertices))
                if fkey in global_face_registry:
                    owner = global_face_registry[fkey]
                    msg = (
                        f"[{label}] Face {list(new_face.vertices)} already present in patch "
                        f"{owner!r}; skipping duplicate."
                    )
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                else:
                    new_patch.faces.append(new_face)
                    global_face_registry[fkey] = new_patch.name
            out.boundary.add(new_patch)
        else:
            # Existing patch — check type compatibility first
            if existing_patch.type != src_patch.type:
                msg = (
                    f"[{label}] Patch type conflict for patch {src_patch.name!r}: "
                    f"existing type={existing_patch.type!r}, "
                    f"source type={src_patch.type!r}. Keeping existing type."
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)

            # Build per-patch face set for same-patch duplicate check
            patch_face_keys: set[tuple[str, ...]] = {
                tuple(sorted(f.vertices)) for f in existing_patch.faces
            }

            for f in src_patch.faces:
                new_face = copy.deepcopy(f)
                new_face.vertices = [vertex_remap.get(n, n) for n in new_face.vertices]
                fkey = tuple(sorted(new_face.vertices))
                if fkey in patch_face_keys:
                    logger.debug(
                        "Skipping duplicate face %s in patch %r from %s",
                        list(new_face.vertices), src_patch.name, label,
                    )
                elif fkey in global_face_registry:
                    # Face exists in a *different* patch — warn
                    owner = global_face_registry[fkey]
                    msg = (
                        f"[{label}] Face {list(new_face.vertices)} is already present in "
                        f"patch {owner!r} (target patch: {src_patch.name!r}). "
                        "Skipping."
                    )
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                else:
                    existing_patch.faces.append(new_face)
                    patch_face_keys.add(fkey)
                    global_face_registry[fkey] = existing_patch.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combine_blockmeshdicts(
    sources: list[BlockMeshDict],
    source_labels: list[str] | None = None,
    combine_cell_zones: str | None = None,
    vertex_tol: float = 1e-9,
    strict: bool = False,
) -> BlockMeshDict:
    """Combine multiple BlockMeshDict objects into a single one.

    Parameters
    ----------
    sources:
        List of parsed BlockMeshDict objects (already loaded by caller).
    source_labels:
        File names or labels for error messages, parallel to *sources*.
        When ``None``, labels default to ``"source[i]"``.
    combine_cell_zones:
        When set, overrides the zone of ALL blocks in the output to this
        value (including blocks whose original zone was ``None``).
    vertex_tol:
        Tolerance for vertex coordinate conflict detection (Euclidean
        distance between two vertices with the same name).
    strict:
        When ``True``, edge / patch / face conflicts and scalar conflicts
        raise ``ValueError`` instead of being logged as warnings.

    Returns
    -------
    BlockMeshDict
        A new BlockMeshDict; sources are not mutated.

    Raises
    ------
    ValueError
        When *sources* is empty, or when an unresolvable conflict is
        detected (e.g. any conflict when *strict* is ``True``).

    Notes
    -----
    Vertex name collisions with differing coordinates are handled
    gracefully: if the incoming coordinate already exists in the output
    under a *different* name the incoming vertex is collapsed onto that
    name (no duplicate added); otherwise the vertex is added under a
    unique name generated by the ``<name>_2``, ``<name>_3``, ... suffix
    scheme.  Downstream references (blocks, edges, faces) are updated
    via the per-source remap table produced by ``_merge_vertices``.
    """
    if not sources:
        raise ValueError(
            "combine_blockmeshdicts requires at least one source BlockMeshDict."
        )

    if source_labels is None:
        source_labels = [f"source[{i}]" for i in range(len(sources))]

    # Initialise the output from the first source (deep copy to avoid mutation)
    first = sources[0]
    first_label = source_labels[0]

    out = BlockMeshDict()
    out.convertToMeters = first.convertToMeters
    out.geometry_body = first.geometry_body
    out.default_patch = copy.deepcopy(first.default_patch)

    # Vertices from the first source
    for v in first.vertices:
        out.vertices.add(copy.deepcopy(v))

    # Edges from the first source
    for e in first.edges:
        out.edges.add(copy.deepcopy(e))

    # Blocks from the first source (with optional zone override)
    for block in first.blocks:
        new_block = copy.deepcopy(block)
        if combine_cell_zones is not None:
            new_block.zone = combine_cell_zones
        out.blocks.add(new_block)

    # Patches from the first source
    for patch in first.boundary:
        new_patch = copy.deepcopy(patch)
        out.boundary.add(new_patch)

    logger.info(
        "[%s] Loaded %d vertex(ices), %d block(s), %d edge(s), %d patch(es)",
        first_label,
        len(first.vertices),
        len(first.blocks),
        len(first.edges),
        len(list(first.boundary)),
    )

    # Merge remaining sources
    for src, label in zip(sources[1:], source_labels[1:], strict=False):
        logger.info(
            "[%s] Merging %d vertex(ices), %d block(s), %d edge(s), %d patch(es)",
            label,
            len(src.vertices),
            len(src.blocks),
            len(src.edges),
            len(list(src.boundary)),
        )
        _merge_scalars(out, src, label, strict)
        vertex_remap = _merge_vertices(out, src, label, vertex_tol)
        _merge_edges(out, src, label, strict, vertex_remap)
        _merge_blocks(out, src, label, combine_cell_zones, vertex_remap)
        _merge_patches(out, src, label, strict, vertex_remap)

    logger.info(
        "Combined result: %d vertex(ices), %d block(s), %d edge(s), %d patch(es)",
        len(out.vertices),
        len(out.blocks),
        len(out.edges),
        len(list(out.boundary)),
    )
    return out
