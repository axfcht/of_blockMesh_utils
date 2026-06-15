"""Block-naming and ordering helpers used during ``BlockMeshDict`` assembly."""

from __future__ import annotations

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.geometry.hex_topology import HEX_FACE_INDICES


def resolve_block_name(
    bmd: BlockMeshDict,
    preferred: str,
    policy: str,
) -> str:
    """Return a block name that does not conflict with existing blocks.

    *policy* is one of ``"suffix"``, ``"error"``, or ``"rename"``. Raises
    ``ValueError`` when *policy* is ``"error"`` and a conflict exists.
    """
    if not bmd.has_block_named(preferred):
        return preferred

    if policy == "error":
        raise ValueError(
            f"Block name '{preferred}' already exists (--nameCollision=error)."
        )

    if policy == "suffix":
        counter = 2
        while True:
            candidate = f"{preferred}_{counter}"
            if not bmd.has_block_named(candidate):
                return candidate
            counter += 1

    if policy == "rename":
        idx = bmd.next_block_index()
        return f"block{idx}"

    raise ValueError(f"Unknown name_collision policy: {policy!r}")


def _block_to_ordering_and_faces(
    block,
    bmd,
) -> tuple[list[int], list[tuple]]:
    """Map a ``Block`` object to a numeric ordering and face-tuple list.

    Returns ``(ordering, face_tuples)`` where ``ordering`` is a list of 8
    integer vertex IDs and ``face_tuples`` is a list of 6 4-tuples.
    """
    name_to_id: dict = {}
    for i, v in enumerate(bmd.vertices):
        if v.name not in name_to_id:
            name_to_id[v.name] = i

    ordering: list[int] = []
    for vname in block.vertices:
        vid = name_to_id.get(vname)
        if vid is None:
            vid = -1
        ordering.append(vid)

    face_tuples: list[tuple] = []
    for local_indices in HEX_FACE_INDICES:
        face = tuple(ordering[k] for k in local_indices)
        face_tuples.append(face)

    return ordering, face_tuples
