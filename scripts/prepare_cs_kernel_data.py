"""Prepare a synthetic correlator_analysis run from cached pion_cg_tmdwf data.

This script reads the post-ground-state-fit bare quasi matrix elements from
the pion_cg_tmdwf cache and writes them into the directory layout expected by
lamet-agent's resume mechanism.  The resulting run directory can be used with:

    lamet-agent run examples/pion_cg_cs_kernel_manifest.json \
        --resume-from <output_dir> --start-stage renormalization

Expected inputs (from pion_cg_tmdwf):
    cache/bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy  (for px in p_ls, b in b_ls)
    output/dump/bare_quasi_zdep_p0_1st.gv              (p=0 reference, optional)

Example usage:
    python scripts/prepare_cs_kernel_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gvar as gv
import numpy as np

TMDWF_ROOT = Path("/home/jinchen/git/anl/pion_cg_tmdwf")
AGENT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = AGENT_ROOT / "examples" / "outputs" / "pion_cg_cs_kernel" / "run_prepared"

# Raw correlator data is written here and referenced by the manifest via ../data/pion_cg_cs_kernel/
DATA_DIR = AGENT_ROOT / "data" / "pion_cg_cs_kernel"

P_LS = [8, 9, 10]
P0_PX = 0
B_LS = [0, 2, 4, 6, 8, 10]
ZMAX = 21

SETUP_ID = "hisq_a06"
GAMMA = "g14"
FLAVOR = "u-d"
SMEARING = "SP"
RESAMPLING = "jackknife"


def load_npy_bare_quasi(px: int, b: int) -> dict[str, np.ndarray]:
    path = TMDWF_ROOT / "cache" / f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"
    if not path.exists():
        path = TMDWF_ROOT / "output" / "dump" / f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"
    data = np.load(path, allow_pickle=True).item()
    re = np.array(data["re"], dtype=float)
    im = np.array(data["im"], dtype=float)
    return {"re": re, "im": im}


def load_p0_gv(b: int) -> dict[str, np.ndarray] | None:
    """Load p=0 reference data from gvar dump."""
    path = TMDWF_ROOT / "cache" / "bare_quasi_zdep_p0_1st.gv"
    if not path.exists():
        path = TMDWF_ROOT / "output" / "dump" / "bare_quasi_zdep_p0_1st.gv"
    if not path.exists():
        return None
    data = gv.load(str(path))
    re_gv = data[f"b{b}_re"]
    im_gv = data[f"b{b}_im"]
    re_mean = np.array(gv.mean(re_gv), dtype=float)
    re_err = np.array(gv.sdev(re_gv), dtype=float)
    im_mean = np.array(gv.mean(im_gv), dtype=float)
    im_err = np.array(gv.sdev(im_gv), dtype=float)
    n_samp = 100
    rng = np.random.default_rng(42)
    re_samples = re_mean[None, :] + re_err[None, :] * rng.standard_normal((n_samp, len(re_mean)))
    im_samples = im_mean[None, :] + im_err[None, :] * rng.standard_normal((n_samp, len(im_mean)))
    return {"re": re_samples.T, "im": im_samples.T}


def build_family(
    px: int, py: int, pz: int, b: int, z_axis: np.ndarray,
    real_samples: np.ndarray, imag_samples: np.ndarray,
    stage_dir: Path,
) -> dict:
    """Build one matrix-element family dict and write its NPZ artifact."""
    n_z, n_samp = real_samples.shape
    real_jk = real_samples.T
    imag_jk = imag_samples.T

    real_mean = np.mean(real_jk, axis=0)
    imag_mean = np.mean(imag_jk, axis=0)
    n = real_jk.shape[0]
    real_error = np.std(real_jk, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))
    imag_error = np.std(imag_jk, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))

    observable = "qtmdwf" if b != 0 else "qda"
    metadata = {
        "setup_id": SETUP_ID,
        "momentum": [px, py, pz],
        "smearing": SMEARING,
        "b": b,
        "px": px,
        "py": py,
        "pz": pz,
        "gamma": GAMMA,
        "flavor": FLAVOR,
        "observable": observable,
        "hadron": "pion",
        "gauge": "cg",
        "analysis_channel": "qda",
        "fit_mode": "joint_ratio_fh",
        "resampling_method": RESAMPLING,
    }
    slug = f"{observable}_{SETUP_ID}_joint_ratio_fh_b{b}_p{px}{py}{pz}_{GAMMA}_{FLAVOR.replace('-','_')}_{SMEARING}"
    npz_path = stage_dir / f"{slug}_samples.npz"
    np.savez(npz_path, z_axis=z_axis, real_samples=real_jk, imag_samples=imag_jk)

    return {
        "metadata": metadata,
        "z_axis": z_axis.tolist(),
        "sample_count": n,
        "real": {"mean": real_mean.tolist(), "error": real_error.tolist()},
        "imag": {"mean": imag_mean.tolist(), "error": imag_error.tolist()},
        "sample_artifact": str(npz_path),
    }


def main():
    stage_dir = OUTPUT_DIR / "stages" / "correlator_analysis"
    stage_dir.mkdir(parents=True, exist_ok=True)

    z_axis = np.arange(ZMAX, dtype=float)
    families = []

    for px in P_LS:
        for b in B_LS:
            print(f"Loading p={px}, b={b} ...")
            data = load_npy_bare_quasi(px, b)
            family = build_family(px, 0, 0, b, z_axis, data["re"], data["im"], stage_dir)
            families.append(family)

    for b in B_LS:
        print(f"Loading p=0, b={b} (reference) ...")
        p0_data = load_p0_gv(b)
        if p0_data is not None:
            family = build_family(0, 0, 0, b, z_axis, p0_data["re"], p0_data["im"], stage_dir)
            families.append(family)
        else:
            print(f"  WARNING: p=0 b={b} data not found, skipping")

    two_point_payloads = []
    for px in [0] + P_LS:
        two_point_payloads.append({
            "momentum": [px, 0, 0],
            "setup_id": SETUP_ID,
            "smearing": SMEARING,
        })

    payload = {
        "two_point": two_point_payloads,
        "group_count": len(two_point_payloads),
        "matrix_element_families": families,
    }

    report = {
        "manifest_path": str(AGENT_ROOT / "examples" / "pion_cg_cs_kernel_manifest.json"),
        "stage_results": [
            {
                "stage_name": "correlator_analysis",
                "summary": "Prepared from pion_cg_tmdwf cached bare quasi data.",
                "payload": payload,
                "artifacts": [],
            }
        ],
    }

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {len(families)} families to {stage_dir}")
    print(f"Report: {report_path}")
    print(f"\nData directory (for manifest correlator paths): {DATA_DIR}")
    print(f"  -> populate with raw correlator .txt files if running from scratch")
    print(f"\nTo run the CS kernel pipeline (from prepared bare-quasi cache):")
    print(f"  lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\")
    print(f"    --resume-from {OUTPUT_DIR} --start-stage renormalization")


if __name__ == "__main__":
    main()
