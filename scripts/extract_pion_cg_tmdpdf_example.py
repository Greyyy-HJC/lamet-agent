"""Extract a small pion CG qTMDPDF example slice into lamet-agent TXT inputs.

Purpose:
    Convert a representative `PX=PY=3, PZ=0` pion CG qTMDPDF dataset from the
    external `pion_cg_tmdpdf` repository into the TXT layouts consumed by the
    lamet-agent built-in correlator loaders.

Inputs:
    - The external repository root containing `data/c2pt_comb/` and
      `data/c3pt_h5/`.
    - The output directory under this repository where extracted TXT files and
      a README should be written.

Outputs:
    - One two-point TXT file with layout:
      `t, real_cfg_0..N, imag_cfg_0..N`
    - Multiple three-point TXT files with layout:
      `tsep, tau, real_cfg_0..N, imag_cfg_0..N`

Example usage:
    .venv/bin/python scripts/extract_pion_cg_tmdpdf_example.py \
        --source-root /home/jinchen/git/anl/pion_cg_tmdpdf \
        --output-dir examples/data/pion_cg_qtmdpdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--px", type=int, default=3)
    parser.add_argument("--py", type=int, default=3)
    parser.add_argument("--pz", type=int, default=0)
    parser.add_argument("--gamma", type=str, default="g8")
    parser.add_argument("--b-values", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--z-values", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tsep-values", type=int, nargs="+", default=[6, 8, 10, 12])
    return parser.parse_args()


def read_two_point(source_root: Path, *, px: int, py: int, pz: int) -> np.ndarray:
    stem = source_root / "data" / "c2pt_comb" / f"c2pt.CG52bxyp20_CG52bxyp20.SS.meson_g15.PX{px}_PY{py}_PZ{pz}"
    real = np.loadtxt(Path(f"{stem}.real"), skiprows=1, delimiter=",")
    imag = np.loadtxt(Path(f"{stem}.imag"), skiprows=1, delimiter=",")
    t_axis = real[:, :1]
    real_samples = real[:, 1:]
    imag_samples = imag[:, 1:]
    return np.concatenate([t_axis, real_samples, imag_samples], axis=1)


def read_three_point(
    source_root: Path,
    *,
    px: int,
    py: int,
    pz: int,
    gamma: str,
    b_value: int,
    z_value: int,
    tsep_values: list[int],
) -> np.ndarray:
    filename = (
        source_root
        / "data"
        / "c3pt_h5"
        / f"qpdf.SS.meson.ama.CG52bxyp20_CG52bxyp20.PX{px}_PY{py}_PZ{pz}.Z0-24.XY0-24.{gamma}.qx0_qy0_qz0.h5"
    )
    rows: list[np.ndarray] = []
    with h5py.File(filename, "r") as handle:
        for tsep in tsep_values:
            dataset = np.asarray(handle[f"dt{tsep}"][f"Z{b_value}"][f"XY{z_value}"][:])
            for tau, values in enumerate(dataset):
                rows.append(
                    np.concatenate(
                        [
                            np.asarray([tsep, tau], dtype=float),
                            np.real(values).astype(float),
                            np.imag(values).astype(float),
                        ]
                    )
                )
    return np.asarray(rows, dtype=float)


def write_readme(
    output_dir: Path,
    *,
    px: int,
    py: int,
    pz: int,
    gamma: str,
    b_values: list[int],
    z_values: list[int],
    tsep_values: list[int],
) -> None:
    lines = [
        "# Pion CG qTMDPDF Example Data",
        "",
        "- Source repository: `/home/jinchen/git/anl/pion_cg_tmdpdf`",
        f"- Retained kinematics: `PX={px}`, `PY={py}`, `PZ={pz}`, `gamma={gamma}`",
        f"- Retained `b` values: `{b_values}`",
        f"- Retained `z` values: `{z_values}`",
        f"- Retained `tsep` values: `{tsep_values}`",
        "- Two-point TXT layout: `t`, all real configurations, then all imaginary configurations.",
        "- Three-point TXT layout: `tsep`, `tau`, all real configurations, then all imaginary configurations.",
        "- These files are a repository-tracked representative slice for future pion CG qTMDPDF examples, not the full analysis dataset.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    two_point = read_two_point(args.source_root.resolve(), px=args.px, py=args.py, pz=args.pz)
    np.savetxt(output_dir / f"two_point_ss_p{args.px}{args.py}{args.pz}.txt", two_point)

    for b_value in args.b_values:
        for z_value in args.z_values:
            three_point = read_three_point(
                args.source_root.resolve(),
                px=args.px,
                py=args.py,
                pz=args.pz,
                gamma=args.gamma,
                b_value=b_value,
                z_value=z_value,
                tsep_values=list(args.tsep_values),
            )
            np.savetxt(
                output_dir / f"three_point_ss_{args.gamma}_p{args.px}{args.py}{args.pz}_b{b_value}_z{z_value:02d}.txt",
                three_point,
            )

    write_readme(
        output_dir,
        px=args.px,
        py=args.py,
        pz=args.pz,
        gamma=args.gamma,
        b_values=list(args.b_values),
        z_values=list(args.z_values),
        tsep_values=list(args.tsep_values),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
