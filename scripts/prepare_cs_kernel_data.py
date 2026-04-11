"""Prepare a correlator_analysis resume point from cached bare quasi matrix elements.

The script converts post-ground-state-fit bare quasi matrix elements into the
directory layout expected by lamet-agent's ``--resume-from`` mechanism.  The
resulting run directory is placed at::

    examples/outputs/pion_cg_cs_kernel/run_prepared/

and can be used with::

    lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\
        --resume-from examples/outputs/pion_cg_cs_kernel/run_prepared \\
        --start-stage renormalization

Cache lookup order (first match wins):
  1. data/pion_cg_cs_kernel_cache/   (local copy — project-independent)
  2. /home/jinchen/git/anl/pion_cg_tmdwf/cache/  (legacy fallback)

Run this script once after cloning the repo (or whenever the upstream cache
changes) to populate the local cache::

    python scripts/prepare_cs_kernel_data.py --save-cache

After that the project is self-contained and the external pion_cg_tmdwf
directory is no longer needed.

Example usage:
    python scripts/prepare_cs_kernel_data.py            # use local cache
    python scripts/prepare_cs_kernel_data.py --save-cache  # copy from upstream first
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import gvar as gv
import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent

# Local cache — gitignored, project-self-contained
LOCAL_CACHE_DIR = AGENT_ROOT / "data" / "pion_cg_cs_kernel_cache"

# Legacy upstream path (only needed when running --save-cache)
UPSTREAM_ROOT = Path("/home/jinchen/git/anl/pion_cg_tmdwf")

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
# Cache management
# ---------------------------------------------------------------------------

def _npy_filename(px: int, b: int) -> str:
    return f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"


def _gv_filename() -> str:
    return "bare_quasi_zdep_p0_1st.gv"


def save_cache_from_upstream() -> None:
    """Copy needed files from the upstream pion_cg_tmdwf project into LOCAL_CACHE_DIR."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for px in P_LS:
        for b in B_LS:
            fname = _npy_filename(px, b)
            src = UPSTREAM_ROOT / "cache" / fname
            if not src.exists():
                src = UPSTREAM_ROOT / "output" / "dump" / fname
            if not src.exists():
                print(f"  WARNING: {fname} not found in upstream, skipping")
                continue
            dst = LOCAL_CACHE_DIR / fname
            shutil.copy2(src, dst)
            copied += 1
            print(f"  Copied {fname}")

    # p=0 gv reference
    gv_src = UPSTREAM_ROOT / "cache" / _gv_filename()
    if not gv_src.exists():
        gv_src = UPSTREAM_ROOT / "output" / "dump" / _gv_filename()
    if gv_src.exists():
        shutil.copy2(gv_src, LOCAL_CACHE_DIR / _gv_filename())
        copied += 1
        print(f"  Copied {_gv_filename()}")
    else:
        print(f"  WARNING: {_gv_filename()} not found in upstream")

    print(f"\nSaved {copied} files to {LOCAL_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_npy(px: int, b: int) -> Path:
    fname = _npy_filename(px, b)
    local = LOCAL_CACHE_DIR / fname
    if local.exists():
        return local
    # Legacy fallback
    for base in [UPSTREAM_ROOT / "cache", UPSTREAM_ROOT / "output" / "dump"]:
        p = base / fname
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{fname} not found. Run with --save-cache to copy from the upstream project first."
    )


def _find_gv() -> Path | None:
    fname = _gv_filename()
    local = LOCAL_CACHE_DIR / fname
    if local.exists():
        return local
    for base in [UPSTREAM_ROOT / "cache", UPSTREAM_ROOT / "output" / "dump"]:
        p = base / fname
        if p.exists():
            return p
    return None


def load_npy_bare_quasi(px: int, b: int) -> dict[str, np.ndarray]:
    path = _find_npy(px, b)
    data = np.load(path, allow_pickle=True).item()
    return {"re": np.array(data["re"], dtype=float), "im": np.array(data["im"], dtype=float)}


def load_p0_gv(b: int) -> dict[str, np.ndarray] | None:
    """Load p=0 reference from gvar dump and generate Gaussian samples."""
    path = _find_gv()
    if path is None:
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


# ---------------------------------------------------------------------------
# Family construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Copy source files from the upstream pion_cg_tmdwf project into data/pion_cg_cs_kernel_cache/ before preparing.",
    )
    args = parser.parse_args()

    if args.save_cache:
        print(f"Saving cache from {UPSTREAM_ROOT} -> {LOCAL_CACHE_DIR} ...")
        save_cache_from_upstream()
        print()

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
        two_point_payloads.append({"momentum": [px, 0, 0], "setup_id": SETUP_ID, "smearing": SMEARING})

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
                "summary": "Prepared from local bare quasi cache (data/pion_cg_cs_kernel_cache/).",
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
    print(f"\nTo run the CS kernel pipeline:")
    print(f"  lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\")
    print(f"    --resume-from {OUTPUT_DIR} --start-stage renormalization")


if __name__ == "__main__":
    main()
