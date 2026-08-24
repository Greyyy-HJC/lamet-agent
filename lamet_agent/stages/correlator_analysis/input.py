"""Strict correlator descriptor validation and HDF5 loading.

A dataset path may contain coordinate placeholders such as ``{tsep}`` and
``{z}``.  ``dataset_dims`` states the leaf-array axis order explicitly; the
loader never guesses axes from shapes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData, EnsembleInfo


def load_descriptor(path: Path) -> dict[str, Any]:
    """Load one JSON descriptor and its HDF5 datasets into raw ``EnsembleData``."""
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"correlator descriptor must be an existing .json file: {path}")
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict) or set(descriptor) != {"ensemble", "configuration_count", "correlators"} or not isinstance(descriptor.get("ensemble"), dict) or not isinstance(descriptor.get("correlators"), list):
        raise ValueError("descriptor requires ensemble, configuration_count, and correlators")
    try:
        ensemble = EnsembleInfo(**descriptor["ensemble"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ensemble must contain exactly the EnsembleInfo fields") from exc
    configuration_count = descriptor["configuration_count"]
    if not isinstance(configuration_count, int) or isinstance(configuration_count, bool) or configuration_count < 2:
        raise ValueError("configuration_count must be an integer of at least two")
    configuration_ids = [str(index) for index in range(configuration_count)]
    output: dict[str, Any] = {"descriptor": descriptor, "configuration_ids": list(configuration_ids), "correlators": {}}
    for record in descriptor["correlators"]:
        if not isinstance(record, dict):
            raise ValueError("each correlator record must be an object")
        required = {"id", "format", "path", "dataset", "dataset_dims", "dims", "coords", "selectors", "correlator_type", "hadron", "source_momentum", "sink_momentum", "current", "source_sink_separation"}
        if set(record) != required:
            raise ValueError(f"correlator record must contain exactly: {sorted(required)}")
        if not isinstance(record["id"], str) or not record["id"] or not isinstance(record["path"], str) or not isinstance(record["dataset"], str):
            raise ValueError("correlator id, path, and dataset must be strings")
        if record["id"] in output["correlators"]:
            raise ValueError(f"correlator id is repeated: {record['id']}")
        if record["format"] != "hdf5" or not isinstance(record["dims"], list) or not record["dims"] or record["dims"][0] != "configuration":
            raise ValueError("only HDF5 descriptors with configuration as the first dimension are supported")
        if any(not isinstance(dim, str) or not dim or dim == "resample" for dim in record["dims"]) or len(set(record["dims"])) != len(record["dims"]):
            raise ValueError("descriptor dimensions must be unique nonempty names")
        if not isinstance(record["coords"], dict) or not isinstance(record["dataset_dims"], list) or not isinstance(record["selectors"], dict):
            raise ValueError("coords, dataset_dims, and selectors must be explicit objects/lists")
        dataset_dims = record["dataset_dims"]
        if any(not isinstance(dim, str) or dim not in record["dims"] for dim in dataset_dims) or len(set(dataset_dims)) != len(dataset_dims):
            raise ValueError("dataset_dims must be unique names from dims")
        if record["correlator_type"] not in {"two_point", "three_point", "qda"}:
            raise ValueError("correlator_type must be two_point, three_point, or qda")
        momenta = [record["source_momentum"], record["sink_momentum"]]
        if any(not isinstance(momentum, list) or len(momentum) != 3 for momentum in momenta) or any(not isinstance(value, int) or isinstance(value, bool) for momentum in momenta for value in momentum):
            raise ValueError("source and sink momenta must be integer triples")
        if record["correlator_type"] == "two_point" and record["current"] is not None:
            raise ValueError("two-point correlators must have current=null")
        if record["correlator_type"] != "two_point" and not isinstance(record["current"], dict):
            raise ValueError("three-point and qDA correlators require a current object")
        if isinstance(record["current"], dict) and set(record["current"]) != {"kernel_operator", "parton", "renormalization_scheme"}:
            raise ValueError("current must contain exactly kernel_operator, parton, and renormalization_scheme")
        if record["source_sink_separation"] is not None and (not isinstance(record["source_sink_separation"], int) or isinstance(record["source_sink_separation"], bool) or record["source_sink_separation"] < 0):
            raise ValueError("source_sink_separation must be a nonnegative integer or null")
        if record["source_sink_separation"] is None and "tsep" not in record["dims"] and record["correlator_type"] == "three_point":
            raise ValueError("a null source_sink_separation requires an explicit tsep dimension")
        if not isinstance(record["hadron"], dict):
            raise ValueError("hadron must be an object")
        hdf5_path = Path(record["path"])
        if not hdf5_path.is_absolute():
            hdf5_path = path.parent / hdf5_path
        if not hdf5_path.is_file():
            raise ValueError(f"correlator HDF5 input does not exist: {hdf5_path.resolve()}")
        import h5py

        placeholders = re.findall(r"{([A-Za-z][A-Za-z0-9_]*)}", record["dataset"])
        if len(placeholders) != len(set(placeholders)) or any(dim not in record["dims"] or dim not in record["coords"] or dim in dataset_dims for dim in placeholders):
            raise ValueError("dataset placeholders must be unique coordinate dimensions outside dataset_dims")
        remaining_dims = [dim for dim in record["dims"] if dim not in placeholders]
        if set(remaining_dims) != set(dataset_dims):
            raise ValueError("dataset_dims plus dataset placeholders must cover dims exactly")
        shape = [configuration_count if dim == "configuration" else len(record["coords"].get(dim, [])) for dim in record["dims"]]
        if any(size == 0 for size in shape):
            raise ValueError("every non-configuration dimension needs nonempty coordinates")
        values = None
        coordinate_products = itertools.product(*(record["coords"][dim] for dim in placeholders)) if placeholders else [()]
        with h5py.File(hdf5_path, "r") as handle:
            for coordinate_values in coordinate_products:
                selected = dict(zip(placeholders, coordinate_values))
                dataset_name = record["dataset"].format(**selected)
                if dataset_name not in handle:
                    raise ValueError(f"HDF5 dataset does not exist: {dataset_name}")
                leaf = np.asarray(handle[dataset_name])
                if leaf.ndim != len(dataset_dims):
                    raise ValueError(f"dataset rank does not match dataset_dims: {dataset_name}")
                sizes = dict(zip(dataset_dims, leaf.shape))
                if sizes.get("configuration") != configuration_count:
                    raise ValueError(f"configuration count does not match descriptor: {dataset_name}")
                for dim, size in sizes.items():
                    if dim != "configuration" and size > len(record["coords"][dim]):
                        raise ValueError(f"dataset is longer than declared coordinate {dim}: {dataset_name}")
                if values is None:
                    values = np.zeros(shape, dtype=leaf.dtype)
                elif values.dtype != leaf.dtype:
                    raise ValueError("templated HDF5 datasets must share one dtype")
                target_dims = [dim for dim in record["dims"] if dim not in placeholders]
                leaf = np.transpose(leaf, [dataset_dims.index(dim) for dim in target_dims])
                selection = []
                for dim in record["dims"]:
                    if dim in selected:
                        selection.append(record["coords"][dim].index(selected[dim]))
                    elif dim == "configuration":
                        selection.append(slice(None))
                    else:
                        selection.append(slice(0, sizes[dim]))
                values[tuple(selection)] = leaf
        assert values is not None
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("HDF5 correlator datasets must be real or complex numeric arrays")
        dims = list(record["dims"])
        coords = record["coords"]
        if values.ndim != len(dims) or values.shape[0] != len(configuration_ids):
            raise ValueError(f"dataset shape does not match descriptor for {record['id']}")
        for dim, size in zip(dims[1:], values.shape[1:]):
            if dim not in coords or not isinstance(coords[dim], list) or len(coords[dim]) != size:
                raise ValueError(f"coordinate length does not match dimension {dim}")
        current = record.get("current") if isinstance(record.get("current"), dict) else {}
        attrs = {
            "correlator_id": record["id"],
            "ensemble_id": ensemble.id,
            "L_s": int(ensemble.L_s),
            "m_pi_gev": float(ensemble.m_pi_gev),
            "correlator_type": record["correlator_type"],
            "hadron": record.get("hadron", {}).get("name") if isinstance(record.get("hadron"), dict) else record.get("hadron"),
            "source_momentum": json.dumps(record["source_momentum"]),
            "sink_momentum": json.dumps(record["sink_momentum"]),
            "lattice_spacing_fm": float(ensemble.a_s),
            "momentum_gev": float(np.linalg.norm(record["sink_momentum"]) * ensemble.k_s),
            "units": json.dumps({"values": "dimensionless", **{dim: "lattice" for dim in dims[1:]}}),
            "coord_unit": "lattice",
        }
        for key in ("gfix", "volume", "source_operator", "sink_operator", "current_operator", "polarization", "bz_direction", "bT", "momentum"):
            if key in record["selectors"]:
                attrs[key] = record["selectors"][key]
        if record["source_sink_separation"] is not None:
            attrs["source_sink_separation"] = int(record["source_sink_separation"])
        for key in ("kernel_operator", "parton", "renormalization_scheme"):
            value = record.get(key, current.get(key))
            if value is not None:
                attrs[key] = value
        raw = EnsembleData(ensemble, "raw", [values[index] for index in range(values.shape[0])], dims[1:], {dim: coords[dim] for dim in dims[1:]}, attrs=attrs, name=record["id"])
        output["correlators"][record["id"]] = raw
    return output


def resample_correlators(raw: dict[str, Any], *, mode: str, group: str, bin_size: int, n_boot: int | None, seed: int) -> dict[str, EnsembleData]:
    """Apply one explicit plan to every configuration-aligned correlator."""
    data = raw["correlators"]
    first = next(iter(data.values()))
    n_configurations = first.n_sample
    if any(item.n_sample != n_configurations for item in data.values()):
        raise ValueError("configuration counts differ inside one resample group")
    if bin_size < 1 or n_configurations // bin_size < 2:
        raise ValueError("bin_size leaves fewer than two configurations")
    binned_count = n_configurations // bin_size
    plan_seed: int | None = None
    if mode == "bootstrap":
        plan_seed = int(seed)
    elif mode != "jackknife":
        raise ValueError("resampling must be bootstrap or jackknife")
    plan_header = json.dumps({"group": group, "configurations": raw.get("configuration_ids", []), "bin_size": bin_size, "mode": mode, "n_boot": n_boot, "seed": plan_seed}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    resample_id = hashlib.sha256(plan_header).hexdigest()
    result: dict[str, EnsembleData] = {}
    for correlator_id, item in data.items():
        prepared = item.bin(bin_size) if bin_size > 1 else item
        if mode == "bootstrap":
            sampled = prepared.bootstrap(int(n_boot), seed=plan_seed)
        else:
            sampled = prepared.jackknife()
        attrs = sampled.attrs
        attrs.update({"resample_id": resample_id, "resample_group": group})
        result[correlator_id] = EnsembleData(sampled.ensemble, sampled.resample, [sample for sample in sampled.values], sampled.dims, sampled.coords, attrs=attrs, name=sampled.name)
    return result
