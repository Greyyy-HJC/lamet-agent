"""Generate lightweight fake HDF5 data for the end-to-end example."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

LT = 32
N_CFG = 64
SEED = 20260507
TSEP_LIST = (4, 6, 8, 10)
ERROR_SCALE = 3.0
PT3_ERROR_SCALE = 2.0

SOURCE_SINK_PT2 = "SS"
SOURCE_SINK_QDA = "SP"
GAMMA_PT2 = "5"
GAMMA_PT3 = "T"
GAMMA_QDA = "T5"
MOMENTUM = "PX0PY0PZ0"
B_DIR = "b_X"
ETA = "eta0"
BT = "bT0"
BZ = "bz0"

DATA_DIR = Path(__file__).resolve().parent / "data"


def _pt2_model(
    t: np.ndarray, *, e0: float, de1: float, z0: float, z1: float
) -> np.ndarray:
    e1 = e0 + de1
    return (
        z0**2 / (2 * e0) * (np.exp(-e0 * t) + np.exp(-e0 * (LT - t)))
        + z1**2 / (2 * e1) * (np.exp(-e1 * t) + np.exp(-e1 * (LT - t)))
    )


def _make_pt2(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(LT, dtype=float)
    data = np.empty((LT, N_CFG), dtype=np.complex128)
    base = _pt2_model(t, e0=0.45, de1=0.55, z0=1.10, z1=0.45)

    for cfg in range(N_CFG):
        cfg_scale = 1.0 + rng.normal(0.0, 0.035 * ERROR_SCALE)
        smooth_noise = rng.normal(0.0, 0.002 * ERROR_SCALE, size=LT)
        smooth_noise = np.convolve(smooth_noise, [0.25, 0.5, 0.25], mode="same")
        real = base * cfg_scale * (1.0 + smooth_noise)
        imag = base * cfg_scale * rng.normal(0.0, 0.0008 * ERROR_SCALE, size=LT)
        data[:, cfg] = real + 1j * imag

    return data


def _make_qda(pt2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(LT, dtype=float)
    e0 = 0.45
    e1 = 0.62
    z0 = 1.10
    z1 = 0.45
    o0 = z0 * 2.60e9
    o1 = z1 * -1.70e10

    base = (
        z0
        * o0
        / (2 * e0)
        * (np.exp(-e0 * t) + np.exp(-e0 * (LT - t)))
        + z1
        * o1
        / (2 * e1)
        * (np.exp(-e1 * t) + np.exp(-e1 * (LT - t)))
    )

    data = np.empty_like(pt2)
    for cfg in range(N_CFG):
        cfg_scale = 1.0 + rng.normal(0.0, 0.050 * ERROR_SCALE)
        smooth_noise = rng.normal(0.0, 0.010 * ERROR_SCALE, size=LT)
        smooth_noise = np.convolve(smooth_noise, [0.25, 0.5, 0.25], mode="same")
        real = base * cfg_scale * (1.0 + smooth_noise)
        imag = base * rng.normal(0.0, 0.0010 * ERROR_SCALE, size=LT)
        data[:, cfg] = real + 1j * imag

    return data


def _make_pt3(pt2: np.ndarray, tsep: int, rng: np.random.Generator) -> np.ndarray:
    tau = np.arange(tsep + 2, dtype=float)
    centered = tau - tsep / 2
    excited = np.exp(-0.45 * np.minimum(tau, tsep + 1 - tau))
    ratio = (
        0.28
        + 0.035 * excited
        + 0.010 * centered / max(tsep, 1)
        + 1j * (0.018 * np.sin(np.pi * tau / (tsep + 1)))
    )

    source = pt2[tsep, :][None, :]
    noise = rng.normal(
        0.0,
        0.010 * ERROR_SCALE * PT3_ERROR_SCALE,
        size=(tsep + 2, N_CFG),
    ) + 1j * rng.normal(
        0.0,
        0.003 * ERROR_SCALE * PT3_ERROR_SCALE,
        size=(tsep + 2, N_CFG),
    )
    return source * ratio[:, None] * (1.0 + noise)


def _write_dataset(path: Path, dataset_path: str, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(
            dataset_path,
            data=data,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def main() -> None:
    rng = np.random.default_rng(SEED)
    pt2 = _make_pt2(rng)
    qda = _make_qda(pt2, rng)

    _write_dataset(
        DATA_DIR / "fake_pt2.h5",
        f"{SOURCE_SINK_PT2}/{GAMMA_PT2}/{MOMENTUM}",
        pt2,
    )
    _write_dataset(
        DATA_DIR / "fake_qda.h5",
        f"{SOURCE_SINK_QDA}/{GAMMA_QDA}/{MOMENTUM}/{B_DIR}/{ETA}/{BT}/{BZ}",
        qda,
    )

    for tsep in TSEP_LIST:
        pt3 = _make_pt3(pt2, tsep, rng)
        _write_dataset(
            DATA_DIR / f"fake_pt3_tsep{tsep}.h5",
            f"{SOURCE_SINK_PT2}/{GAMMA_PT3}/{MOMENTUM}/{B_DIR}/{ETA}/{BT}/{BZ}",
            pt3,
        )

    print(f"Wrote fake data to {DATA_DIR}")
    print(f"pt2: {pt2.shape}, qda: {qda.shape}")
    print("pt3:", {tsep: (tsep + 2, N_CFG) for tsep in TSEP_LIST})


if __name__ == "__main__":
    main()
