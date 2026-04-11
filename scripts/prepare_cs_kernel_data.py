"""Prepare a correlator_analysis resume point from local bare quasi txt data.

Reads post-ground-state-fit bare quasi matrix elements from
``examples/data/pion_cg_cs_kernel/`` and writes the directory layout expected
by lamet-agent's ``--resume-from`` mechanism.  The resulting run directory is
placed at::

    examples/outputs/pion_cg_cs_kernel/run_prepared/

and can be used with::

    lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\
        --resume-from examples/outputs/pion_cg_cs_kernel/run_prepared \\
        --start-stage renormalization

Data source: ``examples/data/pion_cg_cs_kernel/*.txt``
See that directory's README for the file layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent

# Txt data directory (project-self-contained, gitignored except README and .gv)
DATA_DIR = AGENT_ROOT / "examples" / "data" / "pion_cg_cs_kernel"

OUTPUT_DIR = AGENT_ROOT / "examples" / "outputs" / "pion_cg_cs_kernel" / "run_prepared"

P_LS = [8, 9, 10]
B_LS = [0, 2, 4, 6, 8, 10]
ZMAX = 21

SETUP_ID = "hisq_a06"
GAMMA = "g14"
FLAVOR = "u-d"
SMEARING = "SP"
RESAMPLING = "jackknife"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bare_quasi(px: int, b: int) -> dict[str, np.ndarray]:
    """Load finite-momentum bare quasi jackknife samples from txt files."""
    re_path = DATA_DIR / f"bare_quasi_p{px}_b{b}_re.txt"
    im_path = DATA_DIR / f"bare_quasi_p{px}_b{b}_im.txt"
    if not re_path.exists() or not im_path.exists():
        raise FileNotFoundError(
            f"Missing data files for p={px}, b={b}.\n"
            f"Expected: {re_path}\n         {im_path}\n"
            "Restore the txt files from a trusted backup."
        )
    return {
        "re": np.loadtxt(re_path, dtype=float),
        "im": np.loadtxt(im_path, dtype=float),
    }


def load_p0_reference(b: int) -> dict[str, np.ndarray] | None:
    """Load p=0 reference (mean/sdev) and draw synthetic jackknife samples."""
    re_path = DATA_DIR / f"bare_quasi_p0_b{b}_re_meansdev.txt"
    im_path = DATA_DIR / f"bare_quasi_p0_b{b}_im_meansdev.txt"
    if not re_path.exists() or not im_path.exists():
        return None
    re_ms = np.loadtxt(re_path)
    im_ms = np.loadtxt(im_path)
    re_mean, re_err = re_ms[:, 0], re_ms[:, 1]
    im_mean, im_err = im_ms[:, 0], im_ms[:, 1]

    n_samp = 100
    rng = np.random.default_rng(42)
    re_samples = re_mean[None, :] + re_err[None, :] * rng.standard_normal((n_samp, len(re_mean)))
    im_samples = im_mean[None, :] + im_err[None, :] * rng.standard_normal((n_samp, len(im_mean)))
    return {"re": re_samples.T, "im": im_samples.T}


# ---------------------------------------------------------------------------
# Family construction
# ---------------------------------------------------------------------------

def build_family(
    px: int, py: int, pz: int, b: int, z_axis: np.ndarray,
    real_samples: np.ndarray, imag_samples: np.ndarray,
    stage_dir: Path,
) -> dict:
    """Build one matrix-element family dict and write its NPZ artifact."""
    real_jk = real_samples.T
    imag_jk = imag_samples.T

    real_mean  = np.mean(real_jk, axis=0)
    imag_mean  = np.mean(imag_jk, axis=0)
    n = real_jk.shape[0]
    real_error = np.std(real_jk, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))
    imag_error = np.std(imag_jk, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))

    observable = "qtmdwf" if b != 0 else "qda"
    metadata = {
        "setup_id": SETUP_ID,
        "momentum": [px, py, pz],
        "smearing": SMEARING,
        "b": b,
        "px": px, "py": py, "pz": pz,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stage_dir = OUTPUT_DIR / "stages" / "correlator_analysis"
    stage_dir.mkdir(parents=True, exist_ok=True)

    z_axis = np.arange(ZMAX, dtype=float)
    families = []

    for px in P_LS:
        for b in B_LS:
            print(f"Loading p={px}, b={b} ...")
            data = load_bare_quasi(px, b)
            families.append(build_family(px, 0, 0, b, z_axis, data["re"], data["im"], stage_dir))

    for b in B_LS:
        print(f"Loading p=0, b={b} (reference) ...")
        p0_data = load_p0_reference(b)
        if p0_data is not None:
            families.append(build_family(0, 0, 0, b, z_axis, p0_data["re"], p0_data["im"], stage_dir))
        else:
            print(f"  WARNING: p=0 b={b} data not found, skipping")

    two_point_payloads = [
        {"momentum": [px, 0, 0], "setup_id": SETUP_ID, "smearing": SMEARING}
        for px in [0] + P_LS
    ]

    payload = {
        "two_point": two_point_payloads,
        "group_count": len(two_point_payloads),
        "matrix_element_families": families,
    }

    report = {
        "manifest_path": str(AGENT_ROOT / "examples" / "pion_cg_cs_kernel_manifest.json"),
        "stage_results": [{
            "stage_name": "correlator_analysis",
            "summary": "Prepared from local bare quasi data (examples/data/pion_cg_cs_kernel/).",
            "payload": payload,
            "artifacts": [],
        }],
    }

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {len(families)} families to {stage_dir}")
    print(f"Report: {report_path}")
    print(f"\nTo run the CS kernel pipeline:")
    print(f"  lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\")
    print(f"    --resume-from {OUTPUT_DIR} --start-stage renormalization")


if __name__ == "__main__":
    main()
