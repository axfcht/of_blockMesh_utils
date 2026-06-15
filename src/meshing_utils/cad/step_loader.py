"""Shared STEP-file loading utilities for meshing_utils tools.

Provides functions to locate and load STEP files without applying any
hex-topology filter.  Hex-specific logic remains in ``stpToBMD``.

OCP (PythonOCC) is imported lazily so that the module is always importable
(e.g. for unit-test collection) even when OCP is not installed.
"""

import re
from pathlib import Path

from meshing_utils.cad.step_names import NamedSolid, extract_solid_names

# ---------------------------------------------------------------------------
# Public helpers (no OCC dependency)
# ---------------------------------------------------------------------------

def read_step_unit(path: Path) -> str:
    """Parse the ``LENGTH_UNIT`` value from a STEP file header.

    Returns ``"unknown"`` when the header cannot be found.

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    str
        The length unit string (e.g. ``"MM"``), or ``"unknown"``.
    """
    unit_re = re.compile(r"LENGTH_UNIT\s*\(\s*'([^']+)'", re.IGNORECASE)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = unit_re.search(line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return "unknown"


def read_step_solid_names(path: Path) -> list[str]:
    """Extract solid names from a STEP file using regex (no OCP dependency).

    Searches for ``MANIFOLD_SOLID_BREP``, ``BREP_WITH_VOIDS``, and
    ``FACETED_BREP`` entries and returns the name strings in order of
    appearance.

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    list of str
        Names found in the STEP file, in order of appearance.  Empty
        list when the file cannot be read or contains no matching entries.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return re.findall(
        r"(?:MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS|FACETED_BREP)\s*\(\s*'([^']*)'",
        content,
        re.DOTALL,
    )


def find_single_step_file(geometry_dir: Path) -> Path:
    """Find exactly one STEP file in *geometry_dir* and return its path.

    Globs for ``*.stp`` and ``*.step`` files.

    Parameters
    ----------
    geometry_dir:
        Directory to search in (e.g. ``<case>/constant/geometry``).

    Returns
    -------
    Path
        The single STEP file found.

    Raises
    ------
    FileNotFoundError
        When no STEP file is found.
    ValueError
        When more than one STEP file is found.
    """
    geometry_dir = Path(geometry_dir)
    stp_files = list(geometry_dir.glob("*.stp")) + list(geometry_dir.glob("*.step"))
    if len(stp_files) == 0:
        raise FileNotFoundError(f"No STEP file found in {geometry_dir}")
    if len(stp_files) > 1:
        names = [f.name for f in stp_files]
        raise ValueError(
            f"Multiple STEP files found in {geometry_dir}: {names}. "
            "Specify --stpPath to disambiguate."
        )
    return stp_files[0]


# ---------------------------------------------------------------------------
# OCC-dependent helpers
# ---------------------------------------------------------------------------

def explore_solids(shape) -> list:
    """Return all ``TopoDS_Solid`` shapes found within *shape*.

    Parameters
    ----------
    shape:
        Any OCC ``TopoDS_Shape`` (compound, shell, solid, …).

    Returns
    -------
    list of TopoDS_Solid
    """
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(TopoDS.Solid_s(exp.Current()))
        exp.Next()
    return solids


def _label_name(label) -> str | None:
    """Extract a string name from a ``TDF_Label``, or return ``None``."""
    try:
        from OCP.TDataStd import TDataStd_Name

        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
            ext_str = name_attr.Get()
            return ext_str.ToExtString()
    except Exception:
        pass
    return None


def read_step_xcaf(path: Path) -> list[tuple]:
    """Read a STEP file using XCAF and return ``[(solid, label_or_None), ...]``.

    Unlike the version in ``stpToBMD``, this function returns
    **all** solids (no hex-topology filter).  Falls back to an empty list
    on any error (caller uses ``STEPControl_Reader`` as fallback).

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    list of (TopoDS_Solid, Optional[str])
    """
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TDF import TDF_LabelSequence
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool

        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document("XDE")
        app.NewDocument("XDE", doc)

        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            return []
        reader.Transfer(doc)

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        free_labels = TDF_LabelSequence()
        shape_tool.GetFreeShapes(free_labels)

        pairs: list[tuple] = []
        for i in range(1, free_labels.Size() + 1):
            label = free_labels.Value(i)
            top_shape = shape_tool.GetShape_s(label)
            solids = explore_solids(top_shape)
            if solids:
                sub_labels = TDF_LabelSequence()
                shape_tool.GetComponents_s(label, sub_labels, False)
                if sub_labels.Size() == len(solids):
                    for j, solid in enumerate(solids):
                        sub_lbl = sub_labels.Value(j + 1)
                        ref_label = sub_lbl
                        if shape_tool.IsReference_s(sub_lbl):
                            ref_label_out = sub_lbl.__class__()
                            shape_tool.GetReferredShape_s(sub_lbl, ref_label_out)
                            ref_label = ref_label_out
                        name = _label_name(ref_label) or _label_name(sub_lbl)
                        pairs.append((solid, name))
                else:
                    top_name = _label_name(label)
                    for solid in solids:
                        pairs.append((solid, top_name))

        return pairs

    except Exception:
        return []


def load_step_solids(path: Path) -> list[tuple]:
    """Load a STEP file and return all solids with their labels.

    Returns ``[(TopoDS_Solid, Optional[str]), ...]`` for every solid found,
    in XCAF order.  No hex-topology filter is applied — callers that need
    only hexahedral solids must filter themselves.

    Requires OCP (PythonOCC-core) to be installed.  Raises ``ImportError``
    with a helpful message when OCP is absent.

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    list of (TopoDS_Solid, Optional[str])
        Each element is a solid shape and its optional label string.

    Raises
    ------
    ImportError
        When OCP is not installed.
    RuntimeError
        When the file cannot be read or contains no solids at all.
    """
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise ImportError(
            "OCP (cadquery-ocp) is required for load_step_solids. "
            "Install it via: pip install cadquery-ocp"
        ) from exc

    path = Path(path)

    # Try XCAF path first (preserves labels)
    solid_label_pairs = read_step_xcaf(path)

    # Fallback: plain STEPControl_Reader (no labels)
    if not solid_label_pairs:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        solids = explore_solids(shape)
        if not solids:
            raise RuntimeError(f"No solids found in STEP file: {path}")
        solid_label_pairs = [(s, None) for s in solids]

    # Apply raw STEP name fallback (regex-based, no OCP needed)
    raw_names = read_step_solid_names(path)
    if len(raw_names) == len(solid_label_pairs):
        solid_label_pairs = [
            (
                solid,
                raw_name
                if not existing or not existing.strip()
                else existing,
            )
            for (solid, existing), raw_name in zip(solid_label_pairs, raw_names, strict=False)
        ]

    if not solid_label_pairs:
        raise RuntimeError(f"No solids found in STEP file: {path}")

    return solid_label_pairs


def load_solids_with_names(path: Path) -> list[NamedSolid]:
    """Load a STEP file and return solids with reliably resolved names.

    Uses :func:`extract_solid_names` (three-path strategy) to assign a
    unique, OpenFOAM-compatible name to every solid found in the file.
    The existing :func:`load_step_solids` API is left unchanged for
    backward compatibility.

    Requires OCP (PythonOCC-core) to be installed.  Raises ``ImportError``
    with a helpful message when OCP is absent.

    Parameters
    ----------
    path:
        Path to the STEP file.

    Returns
    -------
    list of NamedSolid
        One :class:`~meshing_utils.cad.step_names.NamedSolid` per solid,
        in the order they were extracted from the file.

    Raises
    ------
    ImportError
        When OCP is not installed.
    RuntimeError
        When the file cannot be read or contains no solids at all.
    """
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise ImportError(
            "OCP (cadquery-ocp) is required for load_solids_with_names. "
            "Install it via: pip install cadquery-ocp"
        ) from exc

    path = Path(path)

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    solids = explore_solids(shape)

    if not solids:
        # Try XCAF path as fallback to get solids
        solid_label_pairs = read_step_xcaf(path)
        if solid_label_pairs:
            solids = [s for s, _ in solid_label_pairs]
        else:
            raise RuntimeError(f"No solids found in STEP file: {path}")

    return extract_solid_names(reader, solids, path)
