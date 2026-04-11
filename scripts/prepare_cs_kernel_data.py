"""Prepare a correlator_analysis resume point from cached bare quasi matrix elements.

The script converts post-ground-state-fit bare quasi matrix elements into the
directory layout expected by lamet-agent's ``--resume-from`` mechanism.  The
resulting run directory is placed at::

    examples/outputs/pion_cg_cs_kernel/run_prepared/

and can be used with::

    lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\
        --resume-from examples/outputs/pion_cg_cs_kernel/run_prepared \\
        --start-stage renormalization

Data lookup order (first match wins):
  1. examples/data/pion_cg_cs_kernel/*.txt      (txt export — primary, project-self-contained)
  2. examples/data/pion_cg_cs_kernel_cache/*.npy/.gv  (binary cache — fallback)
  3. /home/jinchen/git/anl/pion_cg_tmdwf/cache/ (upstream — legacy, external dependency)

Typical first-time setup on a new machine::

    # 1. Copy binaries from upstream once
    python scripts/prepare_cs_kernel_data.py --save-cache

    # 2. Export them as tracked-layout txt files
    python scripts/prepare_cs_kernel_data.py --export-txt

    # 3. Build the resume point and run
    python scripts/prepare_cs_kernel_data.py
    lamet-agent run examples/pion_cg_cs_kernel_manifest.json \\
        --resume-from examples/outputs/pion_cg_cs_kernel/run_prepared \\
        --start-stage renormalization

After step 2 the project is fully self-contained; steps 1 and 2 never need to
be repeated unless the upstream data changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gvar as gv
import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent

# Primary txt data (project-self-contained, gitignored)
TXT_DIR = AGENT_ROOT / "examples" / "data" / "pion_cg_cs_kernel"

# Binary cache (gitignored, populated by --save-cache)
CACHE_DIR = AGENT_ROOT / "examples" / "data" / "pion_cg_cs_kernel_cache"

# Legacy upstream (only needed for --save-cache)
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
# Cache/txt management
# ---------------------------------------------------------------------------

def save_cache_from_upstream() -> None:
    """Copy needed files from the upstream pion_cg_tmdwf project into CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for px in P_LS:
        for b in B_LS:
            fname = f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"
            src = UPSTREAM_ROOT / "cache" / fname
            if not src.exists():
                src = UPSTREAM_ROOT / "output" / "dump" / fname
            if not src.exists():
                print(f"  WARNING: {fname} not found in upstream, skipping")
                continue
            shutil.copy2(src, CACHE_DIR / fname)
            copied += 1
            print(f"  Copied {fname}")

    gv_fname = "bare_quasi_zdep_p0_1st.gv"
    gv_src = UPSTREAM_ROOT / "cache" / gv_fname
    if not gv_src.exists():
        gv_src = UPSTREAM_ROOT / "output" / "dump" / gv_fname
    if gv_src.exists():
        shutil.copy2(gv_src, CACHE_DIR / gv_fname)
        copied += 1
        print(f"  Copied {gv_fname}")
    else:
        print(f"  WARNING: {gv_fname} not found in upstream")

    print(f"\nSaved {copied} files to {CACHE_DIR}")


def export_txt_from_cache() -> None:
    """Convert binary cache files to txt layout under TXT_DIR."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for px in P_LS:
        for b in B_LS:
            npy = CACHE_DIR / f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"
            if not npy.exists():
                print(f"  WARNING: {npy.name} not in cache, skipping")
                continue
            data = np.load(npy, allow_pickle=True).item()
            re = np.array(data["re"], dtype=float)
            im = np.array(data["im"], dtype=float)
            np.savetxt(TXT_DIR / f"bare_quasi_p{px}_b{b}_re.txt", re, fmt="%.15e")
            np.savetxt(TXT_DIR / f"bare_quasi_p{px}_b{b}_im.txt", im, fmt="%.15e")
            written += 2
            print(f"  Wrote bare_quasi_p{px}_b{b}_re/im.txt")

    gv_path = CACHE_DIR / "bare_quasi_zdep_p0_1st.gv"
    if gv_path.exists():
        p0 = gv.load(str(gv_path))
        for b in B_LS:
            re_gv = p0[f"b{b}_re"]
            im_gv = p0[f"b{b}_im"]
            np.savetxt(
                TXT_DIR / f"bare_quasi_p0_b{b}_re_meansdev.txt",
                np.column_stack([gv.mean(re_gv), gv.sdev(re_gv)]),
                fmt="%.15e",
            )
            np.savetxt(
                TXT_DIR / f"bare_quasi_p0_b{b}_im_meansdev.txt",
                np.column_stack([gv.mean(im_gv), gv.sdev(im_gv)]),
                fmt="%.15e",
            )
            written += 2
            print(f"  Wrote bare_quasi_p0_b{b}_re/im_meansdev.txt")
    else:
        print(f"  WARNING: {gv_path.name} not in cache, p=0 reference skipped")

    print(f"\nWrote {written} txt files to {TXT_DIR}")


# ---------------------------------------------------------------------------
# Data loading (txt first, then binary cache, then upstream)
# ---------------------------------------------------------------------------

def _load_npy_raw(px: int, b: int) -> dict[str, np.ndarray]:
    """Load a bare quasi npy file from cache or upstream."""
    fname = f"bare_quasi_zdep_p{px}_b{b}_2st_joint.jk.npy"
    for candidate in [
        CACHE_DIR / fname,
        UPSTREAM_ROOT / "cache" / fname,
        UPSTREAM_ROOT / "output" / "dump" / fname,
    ]:
        if candidate.exists():
            data = np.load(candidate, allow_pickle=True).item()
            return {"re": np.array(data["re"], dtype=float), "im": np.array(data["im"], dtype=float)}
    raise FileNotFoundError(
        f"{fname} not found. Run --save-cache then --export-txt first."
    )


def load_npy_bare_quasi(px: int, b: int) -> dict[str, np.ndarray]:
    """Load bare quasi samples; prefer txt, fall back to binary cache / upstream."""
    re_txt = TXT_DIR / f"bare_quasi_p{px}_b{b}_re.txt"
    im_txt = TXT_DIR / f"bare_quasi_p{px}_b{b}_im.txt"
    if re_txt.exists() and im_txt.exists():
        return {
            "re": np.loadtxt(re_txt, dtype=float),
            "im": np.loadtxt(im_txt, dtype=float),
        }
    return _load_npy_raw(px, b)


def load_p0_gv(b: int) -> dict[str, np.ndarray] | None:
    """Load p=0 reference; prefer txt (mean/sdev), fall back to gv binary."""
    re_txt = TXT_DIR / f"bare_quasi_p0_b{b}_re_meansdev.txt"
    im_txt = TXT_DIR / f"bare_quasi_p0_b{b}_im_meansdev.txt"
    if re_txt.exists() and im_txt.exists():
        re_ms = np.loadtxt(re_txt)
        im_ms = np.loadtxt(im_txt)
        re_mean, re_err = re_ms[:, 0], re_ms[:, 1]
        im_mean, im_err = im_ms[:, 0], im_ms[:, 1]
    else:
        gv_path = CACHE_DIR / "bare_quasi_zdep_p0_1st.gv"
        if not gv_path.exists():
            for candidate in [UPSTREAM_ROOT / "cache" / gv_path.name,
                               UPSTREAM_ROOT / "output" / "dump" / gv_path.name]:
                if candidate.exists():
                    gv_path = candidate
                    break
        if not gv_path.exists():
            return None
        p0 = gv.load(str(gv_path))
        re_mean = np.array(gv.mean(p0[f"b{b}_re"]), dtype=float)
        re_err  = np.array(gv.sdev(p0[f"b{b}_re"]), dtype=float)
        im_mean = np.array(gv.mean(p0[f"b{b}_im"]), dtype=float)
        im_err  = np.array(gv.sdev(p0[f"b{b}_im"]), dtype=float)

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--save-cache", action="store_true",
        help="Copy source files from the upstream pion_cg_tmdwf project into "
             "examples/data/pion_cg_cs_kernel_cache/ (one-time operation).",
    )
    parser.add_argument(
        "--export-txt", action="store_true",
        help="Convert binary cache files to txt layout under "
             "examples/data/pion_cg_cs_kernel/ (run once after --save-cache).",
    )
    args = parser.parse_args()

    if args.save_cache:
        print(f"Saving cache from {UPSTREAM_ROOT} -> {CACHE_DIR} ...")
        save_cache_from_upstream()
        print()

    if args.export_txt:
        print(f"Exporting txt files to {TXT_DIR} ...")
        export_txt_from_cache()
        print()

    if args.save_cache or args.export_txt:
        return  # don't also build the resume point

    # --- build resume point ---
    stage_dir = OUTPUT_DIR / "stages" / "correlator_analysis"
    stage_dir.mkdir(parents=True, exist_ok=True)

    z_axis = np.arange(ZMAX, dtype=float)
    families = []

    for px in P_LS:
        for b in B_LS:
            print(f"Loading p={px}, b={b} ...")
            data = load_npy_bare_quasi(px, b)
            families.append(build_family(px, 0, 0, b, z_axis, data["re"], data["im"], stage_dir))

    for b in B_LS:
        print(f"Loading p=0, b={b} (reference) ...")
        p0_data = load_p0_gv(b)
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
