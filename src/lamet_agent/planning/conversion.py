"""Conversion helpers for interactive planning."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from lamet_agent.manifest import parse_volume

from .core import (
    CorrelatorH5Mapping,
    H5DatasetSummary,
    H5Inspection,
    _artifacts_dir,
    _as_list,
    _dataclass_json,
    _resolve_manifest_path,
)


def _dataset_attrs(obj: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in obj.attrs.items():
        if len(attrs) >= 8:
            break
        attrs[str(key)] = str(value)
    return attrs


def inspect_correlator_h5_files(manifest_path: Path, payload: dict[str, Any]) -> list[H5Inspection]:
    """Inspect HDF5/NumPy files referenced by inputs.correlators only."""
    try:
        import h5py
    except ImportError:
        h5py = None

    inspections: list[H5Inspection] = []
    correlators = payload.get("inputs", {}).get("correlators", [])
    if not isinstance(correlators, list):
        return inspections
    for item in correlators:
        if not isinstance(item, dict):
            continue
        resolved = _resolve_manifest_path(manifest_path, payload, item.get("data_path"))
        correlator_id = str(item.get("correlator_id", ""))
        if resolved is None or not resolved.exists():
            inspections.append(H5Inspection(correlator_id=correlator_id, path=str(item.get("data_path", "")), exists=False))
            continue
        datasets: list[H5DatasetSummary] = []
        file_attrs: dict[str, str] = {}
        suffix = resolved.suffix.lower()
        try:
            if suffix in {".h5", ".hdf5"}:
                if h5py is None:
                    raise ValueError("h5py is not installed; install the analysis extra to inspect correlator HDF5 files.")
                with h5py.File(resolved, "r") as h5f:
                    file_attrs = _dataset_attrs(h5f)
                    if "bz_direction" in h5f.attrs:
                        file_attrs["bz_direction"] = str(h5f.attrs["bz_direction"])
                    def visit(name: str, obj: Any) -> None:
                        if isinstance(obj, h5py.Dataset):
                            datasets.append(
                                H5DatasetSummary(
                                    path=name,
                                    shape=[int(dim) for dim in obj.shape],
                                    dtype=str(obj.dtype),
                                    attrs=_dataset_attrs(obj),
                                )
                            )

                    h5f.visititems(visit)
            elif suffix == ".npy":
                import numpy as np

                arr = np.load(resolved, mmap_mode="r")
                datasets.append(H5DatasetSummary(path="array", shape=[int(dim) for dim in arr.shape], dtype=str(arr.dtype)))
            elif suffix == ".npz":
                import numpy as np

                with np.load(resolved) as npz:
                    for key in sorted(npz.files):
                        arr = npz[key]
                        datasets.append(H5DatasetSummary(path=str(key), shape=[int(dim) for dim in arr.shape], dtype=str(arr.dtype)))
            else:
                raise ValueError("Unsupported correlator data format; convert to .npy, .npz, or preferably standard .h5.")
        except Exception as exc:
            inspections.append(H5Inspection(correlator_id=correlator_id, path=str(resolved), exists=True, error=str(exc)))
            continue
        inspections.append(
            H5Inspection(
                correlator_id=correlator_id,
                path=str(resolved),
                exists=True,
                attrs=file_attrs,
                datasets=datasets,
            )
        )
    return inspections


def _standard_dataset_paths(correlator: dict[str, Any]) -> list[str]:
    correlator_type = correlator.get("correlator_type")
    source_operator = str(correlator.get("source_operator", ""))
    sink_operator = str(correlator.get("sink_operator", ""))
    momenta = _as_list(correlator.get("momentum"))
    if correlator_type == "2pt":
        return [f"{source_operator}/{sink_operator}/{momentum}" for momentum in momenta]
    if correlator_type == "3pt":
        current_operator = str(correlator.get("current_operator", ""))
        bT_values = _as_list(correlator.get("bT"))
        tseps = _as_list(correlator.get("tsep"))
        if not bT_values or not tseps:
            return []
        return [
            f"{source_operator}/{sink_operator}/{current_operator}/{momentum}/tsep{tsep}/bT{bT}/bz{z}"
            for momentum in momenta
            for tsep in tseps
            for bT in bT_values
            for z in _as_list(correlator.get("bz"))
        ]
    return []


def _dataset_names(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5f:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    out[name] = [int(dim) for dim in obj.shape]

            h5f.visititems(visit)
    elif suffix == ".npy":
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        out["array"] = [int(dim) for dim in arr.shape]
    elif suffix == ".npz":
        import numpy as np

        with np.load(path) as npz:
            for key in sorted(npz.files):
                out[str(key)] = [int(dim) for dim in npz[key].shape]
    return out


def _choose_source_datasets(correlator: dict[str, Any], source: Path) -> tuple[list[dict[str, Any]], bool, str | None]:
    names = _dataset_names(source)
    targets = _standard_dataset_paths(correlator)
    if not targets:
        return [], True, "Cannot build standard target paths from correlator metadata."
    if all(target in names for target in targets):
        for target in targets:
            shape = names[target]
            if len(shape) != 2:
                return [], True, f"Standard dataset {target!r} has shape {shape}; expected two dimensions."
            if correlator.get("correlator_type") == "2pt":
                try:
                    temporal_extent = parse_volume(str(correlator.get("volume", "")))[1]
                except ValueError as exc:
                    return [], True, str(exc)
                if shape[0] != temporal_extent:
                    return [], True, (
                        f"Standard 2pt dataset {target!r} has Lt={shape[0]}; "
                        f"expected {temporal_extent} from manifest volume."
                    )
            else:
                match = re.search(r"/tsep(\d+)/", f"/{target}/")
                if match is not None and shape[0] != int(match.group(1)) + 1:
                    return [], True, (
                        f"Standard 3pt dataset {target!r} has tau length {shape[0]}; "
                        f"expected {int(match.group(1)) + 1}."
                    )
        return [], False, None
    available = sorted(names)
    if correlator.get("correlator_type") == "2pt":
        if len(available) != 1:
            return [], True, f"Expected one 2pt dataset or the standard path; found {len(available)} datasets."
        return [{"source": available[0], "target": targets[0], "transpose": False}], True, (
            "Non-standard HDF5 2pt input requires explicit confirmation of time/cfg axes and transpose. "
            f"Available dataset: {_source_summary(names)}. Expected standard target: {targets[0]}."
        )
    if correlator.get("correlator_type") == "3pt":
        bz_values = _as_list(correlator.get("bz"))
        if len(available) != len(targets):
            return [], True, f"Expected {len(targets)} 3pt datasets for bz values {bz_values}; found {len(available)}."
        return [], True, (
            "Non-standard HDF5 3pt input requires explicit source-to-bz mapping and tau/cfg axis confirmation. "
            f"Available datasets: {_source_summary(names)}. Expected standard targets: {targets}."
        )
    return [], True, f"Unsupported correlator type {correlator.get('correlator_type')!r}."


def _source_summary(names: dict[str, list[int]]) -> str:
    return ", ".join(f"{key}: shape={shape}" for key, shape in sorted(names.items()))


def plan_correlator_h5_conversions(manifest_path: Path, payload: dict[str, Any]) -> list[CorrelatorH5Mapping]:
    """Return required non-ambiguous and ambiguous correlator data conversions."""
    data_dir = _artifacts_dir(manifest_path, payload) / "plan_data"
    conversions: list[CorrelatorH5Mapping] = []
    correlators = payload.get("inputs", {}).get("correlators", [])
    if not isinstance(correlators, list):
        return conversions
    for item in correlators:
        if not isinstance(item, dict):
            continue
        plan_sources = item.get("plan_sources")
        if isinstance(plan_sources, dict):
            two_point_path = _resolve_manifest_path(manifest_path, payload, plan_sources.get("two_point"))
            current = _resolve_manifest_path(manifest_path, payload, plan_sources.get("current"))
            targets = _standard_dataset_paths(item)
            if two_point_path is None or not two_point_path.exists() or current is None or not current.exists() or len(targets) != 1:
                continue
            two_point_names = _dataset_names(two_point_path)
            current_names = _dataset_names(current)
            source_name = next(iter(two_point_names)) if len(two_point_names) == 1 else None
            current_name = next(iter(current_names)) if len(current_names) == 1 else None
            if source_name is None or current_name is None:
                conversions.append(
                    CorrelatorH5Mapping(
                        correlator_id=str(item.get("correlator_id")),
                        source_file=str(two_point_path),
                        output_file=str(_resolve_manifest_path(manifest_path, payload, item.get("data_path")) or data_dir / f"{item.get('correlator_id')}.h5"),
                        datasets=[],
                        attrs={"standard_correlator_hdf5_version": 2, "bz_direction": item.get("bz_direction", "z")},
                        ambiguous=True,
                        reason="Planned 2pt-current composition requires exactly one dataset in the 2pt file and one dataset in the current file.",
                        operation="compose_2pt_current",
                    )
                )
                continue
            output = _resolve_manifest_path(manifest_path, payload, item.get("data_path")) or data_dir / f"{item.get('correlator_id')}.h5"
            script = data_dir / f"compose_{item.get('correlator_id')}.py"
            conversions.append(
                CorrelatorH5Mapping(
                    correlator_id=str(item.get("correlator_id")),
                    source_file=str(two_point_path),
                    output_file=str(output),
                    script_file=str(script),
                    datasets=[
                        {
                            "source": source_name,
                            "current_file": str(current),
                            "current_source": current_name,
                            "target": targets[0],
                        }
                    ],
                    attrs={"standard_correlator_hdf5_version": 2, "bz_direction": item.get("bz_direction", "z")},
                    operation="compose_2pt_current",
                )
            )
            continue
        source = _resolve_manifest_path(manifest_path, payload, item.get("data_path"))
        if source is None or not source.exists():
            continue
        suffix = source.suffix.lower()
        correlator_id = str(item.get("correlator_id", source.stem))
        output = data_dir / f"{correlator_id}.h5"
        script = data_dir / f"convert_{correlator_id}.py"
        if suffix not in {".h5", ".hdf5", ".npy", ".npz"}:
            conversions.append(
                CorrelatorH5Mapping(
                    correlator_id=correlator_id,
                    source_file=str(source),
                    output_file=str(output),
                    script_file=str(script),
                    datasets=[],
                    ambiguous=True,
                    reason="Unsupported correlator data format. Convert the data to .npy, .npz, or preferably standard .h5.",
                )
            )
            continue
        try:
            names = _dataset_names(source)
            targets = _standard_dataset_paths(item)
            if suffix in {".h5", ".hdf5"}:
                datasets, ambiguous, reason = _choose_source_datasets(item, source)
            elif suffix == ".npy":
                array_shape = names.get("array")
                plan_generated = bool(item.get("plan_generated"))
                if plan_generated and item.get("correlator_type") == "3pt" and array_shape and len(array_shape) == 3 and len(targets) == array_shape[0]:
                    datasets = [
                        {"source": "array", "target": target, "index": {0: index}, "transpose": False}
                        for index, target in enumerate(targets)
                    ]
                    ambiguous, reason = False, None
                elif len(targets) == 1 and array_shape:
                    if item.get("correlator_type") == "2pt":
                        lt_match = re.search(r"T(\d+)", str(item.get("volume", "")))
                        lt = int(lt_match.group(1)) if lt_match else None
                        transpose = bool(lt and len(array_shape) == 2 and array_shape[1] == lt and array_shape[0] != lt)
                        datasets = [{"source": "array", "target": targets[0], "transpose": transpose}]
                        ambiguous = not (plan_generated and lt and len(array_shape) == 2 and lt in array_shape)
                        reason = None if not ambiguous else (
                            "NumPy 2pt input requires explicit confirmation of cfg and time axes, momentum selection, and whether transpose is needed. "
                            f"Available array: {_source_summary(names)}. Expected standard target: {targets[0]}."
                        )
                    else:
                        datasets, ambiguous, reason = [{"source": "array", "target": targets[0], "transpose": False}], True, (
                            "NumPy 3pt input requires explicit confirmation of cfg, tau, z/bz ordering, momentum selection, and whether transpose is needed. "
                            f"Available array: {_source_summary(names)}. Expected standard targets: {targets}."
                        )
                else:
                    datasets, ambiguous, reason = [], True, (
                        "NumPy input cannot be mapped without an explicit source/target mapping. "
                        f"Available array: {_source_summary(names)}. Expected standard targets: {targets}."
                    )
            else:
                available = set(names)
                if targets and all(target in available for target in targets):
                    datasets, ambiguous, reason = [{"source": target, "target": target, "transpose": False} for target in targets], False, None
                elif item.get("plan_generated") and item.get("correlator_type") == "3pt" and len(names) == 1:
                    source_name = next(iter(names))
                    shape = names[source_name]
                    if len(shape) == 3 and len(targets) == shape[0]:
                        datasets = [
                            {"source": source_name, "target": target, "index": {0: index}, "transpose": False}
                            for index, target in enumerate(targets)
                        ]
                        ambiguous, reason = False, None
                    else:
                        datasets, ambiguous, reason = [], True, (
                            "NPZ 3pt input requires a single array with shape (bz, tau, cfg). "
                            f"Available arrays: {_source_summary(names)}. Expected standard targets: {targets}."
                        )
                elif len(targets) == 1 and len(names) == 1:
                    source_name = next(iter(names))
                    datasets, ambiguous, reason = [{"source": source_name, "target": targets[0], "transpose": False}], True, (
                        "NPZ input has one candidate array, but cfg/time or cfg/tau axes and momentum selection must be confirmed explicitly. "
                        f"Available arrays: {_source_summary(names)}. Expected standard target: {targets[0]}."
                    )
                else:
                    datasets, ambiguous, reason = [], True, (
                        "NPZ input requires explicit key-to-target and axis mapping. "
                        f"Available arrays: {_source_summary(names)}. Expected standard targets: {targets}."
                    )
        except Exception as exc:
            datasets, ambiguous, reason = [], True, str(exc)
        if not datasets and not ambiguous:
            continue
        conversions.append(
            CorrelatorH5Mapping(
                correlator_id=correlator_id,
                source_file=str(source),
                output_file=str(output),
                script_file=str(script),
                datasets=datasets,
                attrs=(
                    {"standard_correlator_hdf5_version": 2, "bz_direction": item["bz_direction"]}
                    if item.get("correlator_type") == "3pt"
                    else {"standard_correlator_hdf5_version": 2}
                ),
                ambiguous=ambiguous,
                reason=reason,
            )
        )
    return conversions


def _copy_h5_attrs(source_obj: Any, target_obj: Any) -> None:
    for key, value in source_obj.attrs.items():
        target_obj.attrs[key] = value


def convert_correlator_h5(mapping: CorrelatorH5Mapping) -> None:
    """Write one converted standard correlator HDF5 file."""
    if mapping.ambiguous:
        raise ValueError(f"Cannot convert ambiguous mapping for {mapping.correlator_id}: {mapping.reason}")
    import h5py
    import numpy as np

    output = Path(mapping.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mapping.script_file:
        script = Path(mapping.script_file)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "from lamet_agent.planning import CorrelatorH5Mapping, convert_correlator_h5\n\n"
            "mapping = CorrelatorH5Mapping(**"
            + repr(_dataclass_json(mapping))
            + ")\n"
            "convert_correlator_h5(mapping)\n",
            encoding="utf-8",
        )
    if mapping.operation == "compose_2pt_current":
        with h5py.File(output, "w") as dst:
            for key, value in mapping.attrs.items():
                dst.attrs[key] = value
            for item in mapping.datasets:
                c2_name = str(item.get("source") or "array")
                current_file = str(item["current_file"])
                current_name = str(item.get("current_source") or "array")
                if Path(mapping.source_file).suffix.lower() in {".h5", ".hdf5"}:
                    with h5py.File(mapping.source_file, "r") as h5f:
                        c2 = np.asarray(h5f[c2_name])
                elif Path(mapping.source_file).suffix.lower() == ".npz":
                    with np.load(mapping.source_file) as npz:
                        c2 = np.asarray(npz[c2_name])
                else:
                    c2 = np.asarray(np.load(mapping.source_file))
                if Path(current_file).suffix.lower() in {".h5", ".hdf5"}:
                    with h5py.File(current_file, "r") as h5f:
                        current = np.asarray(h5f[current_name])
                elif Path(current_file).suffix.lower() == ".npz":
                    with np.load(current_file) as npz:
                        current = np.asarray(npz[current_name])
                else:
                    current = np.asarray(np.load(current_file))
                data = c2 * current
                target_name = str(item["target"])
                match = re.search(r"/tsep(\d+)/", f"/{target_name}/")
                if match and data.ndim == 2 and data.shape[0] == 1 and int(match.group(1)) + 1 > 1:
                    data = np.repeat(data, int(match.group(1)) + 1, axis=0)
                dataset = dst.create_dataset(target_name, data=data)
                dataset.attrs["lamet_agent_original_2pt_file"] = mapping.source_file
                dataset.attrs["lamet_agent_original_2pt_dataset"] = str(item.get("source") or "array")
                dataset.attrs["lamet_agent_original_current_file"] = str(item["current_file"])
                dataset.attrs["lamet_agent_original_current_dataset"] = str(item.get("current_source") or "array")
                dataset.attrs["lamet_agent_plan_operation"] = mapping.operation
        return
    suffix = Path(mapping.source_file).suffix.lower()
    npy_data = np.load(mapping.source_file, mmap_mode="r") if suffix == ".npy" else None
    npz_data = np.load(mapping.source_file) if suffix == ".npz" else None
    h5_source = h5py.File(mapping.source_file, "r") if suffix in {".h5", ".hdf5"} else None
    try:
        with h5py.File(output, "w") as dst:
            for key, value in mapping.attrs.items():
                dst.attrs[key] = value
            for item in mapping.datasets:
                source_name = str(item.get("source") or "array")
                target_name = str(item["target"])
                if h5_source is not None:
                    data = np.asarray(h5_source[source_name])
                elif npz_data is not None:
                    data = np.asarray(npz_data[source_name])
                else:
                    data = np.asarray(npy_data)
                original_shape = list(data.shape)
                fixed_axes = {int(axis) for axis in (item.get("index") or {})}
                for axis, index in sorted((item.get("index") or {}).items(), key=lambda pair: int(pair[0]), reverse=True):
                    data = np.take(data, int(index), axis=int(axis))
                if item.get("axis_order") is not None:
                    axes = [int(axis) for axis in item["axis_order"]]
                    if sorted(axes) == list(range(1, data.ndim + 1)):
                        axes = [axis - 1 for axis in axes]
                    elif axes and max(axes) >= data.ndim:
                        remaining_axes = [axis for axis in range(len(original_shape)) if axis not in fixed_axes]
                        axes = [remaining_axes.index(axis) for axis in axes if axis in remaining_axes]
                    data = np.transpose(data, axes)
                if item.get("transpose"):
                    data = data.T
                dataset = dst.create_dataset(target_name, data=data)
                if h5_source is not None:
                    _copy_h5_attrs(h5_source[source_name], dataset)
                dataset.attrs["lamet_agent_original_file"] = mapping.source_file
                dataset.attrs["lamet_agent_original_dataset"] = source_name
                dataset.attrs["lamet_agent_original_shape"] = json.dumps(original_shape)
                dataset.attrs["lamet_agent_transposed"] = bool(item.get("transpose"))
                if item.get("axis_order") is not None:
                    dataset.attrs["lamet_agent_axis_order"] = json.dumps(item["axis_order"])
                if item.get("index"):
                    dataset.attrs["lamet_agent_index"] = json.dumps(item["index"])
    finally:
        if h5_source is not None:
            h5_source.close()
        if npz_data is not None:
            npz_data.close()
