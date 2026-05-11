"""Generate lightweight fake qTMD HDF5 data for end-to-end examples."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

LT = 32
N_CFG = 64
SEED = 20260507
TSEP_LIST = (4, 6, 8, 10)

Z_ARR = np.arange(12)
B_ARR = np.arange(0, 12, 2)
QTMDWF_PZ_LIST = (0, 5, 6, 7)
QTMDPDF_PZ = 6

ERROR_SCALE = 5.0
PT3_ERROR_SCALE = 3.0
A_FM = 0.0836

SOURCE_SINK_PT2 = "SS"
SOURCE_SINK_QTMDWF = "SP"
SOURCE_SINK_PT3 = "SS"
GAMMA_PT2 = "5"
GAMMA_QTMDWF = "T5"
GAMMA_PT3 = "T"
B_DIR = "b_X"
ETA = "eta0"

DATA_DIR = Path(__file__).resolve().parent / "data"


def _momentum_key(pz: int) -> str:
    return f"PX0PY0PZ{pz}"


def _qtmd_path(source_sink: str, gamma: str, pz: int, b: int, z: int) -> str:
    return (
        f"{source_sink}/{gamma}/{_momentum_key(pz)}/{B_DIR}/{ETA}/"
        f"bT{b}/bz{z}"
    )


def _pt2_model(
    t: np.ndarray, *, e0: float, de1: float, z0: float, z1: float
) -> np.ndarray:
    e1 = e0 + de1
    return (
        z0**2 / (2 * e0) * (np.exp(-e0 * t) + np.exp(-e0 * (LT - t)))
        + z1**2 / (2 * e1) * (np.exp(-e1 * t) + np.exp(-e1 * (LT - t)))
    )


def _make_pt2(pz: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(LT, dtype=float)
    pz_shift = 0.012 * pz**2
    data = np.empty((LT, N_CFG), dtype=np.complex128)
    base = _pt2_model(
        t,
        e0=0.45 + pz_shift,
        de1=0.55 + 0.015 * pz,
        z0=1.10 / np.sqrt(1.0 + 0.05 * pz),
        z1=0.45 / np.sqrt(1.0 + 0.03 * pz),
    )

    for cfg in range(N_CFG):
        cfg_scale = 1.0 + rng.normal(0.0, 0.035 * ERROR_SCALE)
        smooth_noise = rng.normal(0.0, 0.002 * ERROR_SCALE, size=LT)
        smooth_noise = np.convolve(smooth_noise, [0.25, 0.5, 0.25], mode="same")
        real = base * cfg_scale * (1.0 + smooth_noise)
        imag = base * cfg_scale * rng.normal(0.0, 0.0008 * ERROR_SCALE, size=LT)
        data[:, cfg] = real + 1j * imag

    return data


def _qtmdwf_zb_envelope(pz: int, b: int, z: int) -> float:
    z_width = 3.4 + 0.03 * pz
    b_width = 8.5
    z_decay = np.exp(-0.5 * (z / z_width) ** 2)
    b_decay = np.exp(-0.5 * (b / b_width) ** 2)
    cs_like = np.exp(-0.050 * b * np.log((pz + 2.0) / 2.0))
    return float(z_decay * b_decay * cs_like)


def _make_qtmdwf(
    pt2: np.ndarray,
    *,
    pz: int,
    b: int,
    z: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(LT, dtype=float)
    e0 = 0.45 + 0.012 * pz**2
    e1 = e0 + 0.62 + 0.012 * pz
    z0 = 1.10 / np.sqrt(1.0 + 0.05 * pz)
    z1 = 0.45 / np.sqrt(1.0 + 0.03 * pz)
    envelope = _qtmdwf_zb_envelope(pz, b, z)
    o0 = z0 * 2.20e3 * envelope
    o1 = z1 * 2.60e2 * envelope

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
        cfg_scale = 1.0 + rng.normal(0.0, 0.060 * ERROR_SCALE)
        smooth_noise = rng.normal(0.0, 0.012 * ERROR_SCALE, size=LT)
        smooth_noise = np.convolve(smooth_noise, [0.25, 0.5, 0.25], mode="same")
        real = base * np.maximum(cfg_scale, 0.2) * np.maximum(1.0 + smooth_noise, 0.2)
        imag_phase = 0.015 * np.sin(0.30 * z + 0.08 * pz + 0.04 * b)
        imag = real * (
            imag_phase
            + rng.normal(0.0, 0.0020 * ERROR_SCALE, size=LT)
        )
        data[:, cfg] = real + 1j * imag

    return data


def _qtmdpdf_ratio(tau: np.ndarray, tsep: int, b: int, z: int) -> np.ndarray:
    centered = tau - tsep / 2
    endpoint_dist = np.minimum(tau + 0.5, tsep + 1.5 - tau)
    excited = np.exp(-0.45 * endpoint_dist)
    z_decay = np.exp(-((z / 3.0) ** 2))
    b_decay = np.exp(-0.5 * (b / 9.0) ** 2)
    plateau = 0.30 * z_decay * b_decay
    tau_curvature = 1.0 + 0.012 * (centered / max(tsep, 1)) ** 2
    real = plateau * tau_curvature + 0.020 * z_decay * b_decay * excited
    imag = 0.006 * real * np.sin(np.pi * centered / (tsep + 1))
    return real + 1j * imag


def _make_qtmdpdf_3pt(
    pt2: np.ndarray,
    *,
    tsep: int,
    b: int,
    z: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tau = np.arange(tsep + 2, dtype=float)
    source = pt2[tsep, :][None, :]
    ratio = _qtmdpdf_ratio(tau, tsep, b, z)
    noise = rng.normal(
        0.0,
        0.014 * ERROR_SCALE * PT3_ERROR_SCALE,
        size=(tsep + 2, N_CFG),
    ) + 1j * rng.normal(
        0.0,
        0.006 * ERROR_SCALE * PT3_ERROR_SCALE,
        size=(tsep + 2, N_CFG),
    )
    return source * ratio[:, None] * (1.0 + noise)


def _make_soft_function(rng: np.random.Generator) -> dict[str, np.ndarray]:
    b_fm = B_ARR * A_FM
    central = 1.0 / (1.0 + 0.65 * b_fm**2) * np.exp(-0.09 * B_ARR)
    data = np.empty((N_CFG, len(B_ARR)), dtype=float)
    for cfg in range(N_CFG):
        cfg_scale = 1.0 + rng.normal(0.0, 0.045 * ERROR_SCALE)
        b_noise = rng.normal(0.0, 0.020 * ERROR_SCALE, size=len(B_ARR))
        b_noise = np.convolve(b_noise, [0.25, 0.5, 0.25], mode="same")
        data[cfg] = np.maximum(central * cfg_scale * (1.0 + b_noise), 1e-6)

    return {"b_fm": b_fm, "softf_samp": data}


def _write_h5_datasets(path: Path, datasets: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        for dataset_path, data in datasets.items():
            h5f.create_dataset(
                dataset_path,
                data=data,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )


def _count_datasets(path: Path) -> int:
    count = 0
    with h5py.File(path, "r") as h5f:
        def visit(_name: str, obj: h5py.Dataset | h5py.Group) -> None:
            nonlocal count
            if isinstance(obj, h5py.Dataset):
                count += 1

        h5f.visititems(visit)
    return count


def _dataset_shape(path: Path, dataset_path: str) -> tuple[int, ...]:
    with h5py.File(path, "r") as h5f:
        return tuple(h5f[dataset_path].shape)


def _assert_shape(actual: tuple[int, ...], expected: tuple[int, ...], label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected shape {expected}, got {actual}")


def _run_shape_checks() -> None:
    qtmdwf_path = DATA_DIR / "fake_qtmdwf.h5"
    pz = 7
    fixed_b = 4
    fixed_z = 3
    raw_qtmdwf_shape = _dataset_shape(
        qtmdwf_path,
        _qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, fixed_b, fixed_z),
    )
    _assert_shape(raw_qtmdwf_shape, (LT, N_CFG), "qTMDWF raw")

    with h5py.File(qtmdwf_path, "r") as h5f:
        qtmdwf_zdep = np.stack(
            [
                h5f[_qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, fixed_b, z)][:]
                for z in Z_ARR
            ],
            axis=0,
        )
        qtmdwf_bdep = np.stack(
            [
                h5f[_qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, b, fixed_z)][:]
                for b in B_ARR
            ],
            axis=0,
        )
    _assert_shape(qtmdwf_zdep.shape, (len(Z_ARR), LT, N_CFG), "qTMDWF zdep")
    _assert_shape(qtmdwf_bdep.shape, (len(B_ARR), LT, N_CFG), "qTMDWF bdep")

    pt3_shapes: dict[int, tuple[int, ...]] = {}
    for tsep in TSEP_LIST:
        pt3_path = DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5"
        pt3_dataset = _qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, QTMDPDF_PZ, fixed_b, fixed_z)
        pt3_shapes[tsep] = _dataset_shape(pt3_path, pt3_dataset)
        _assert_shape(pt3_shapes[tsep], (tsep + 2, N_CFG), f"qTMDPDF 3pt ts{tsep}")

        with h5py.File(pt3_path, "r") as h5f:
            pt3_zdep = np.stack(
                [
                    h5f[_qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, QTMDPDF_PZ, fixed_b, z)][:]
                    for z in Z_ARR
                ],
                axis=0,
            )
            pt3_bdep = np.stack(
                [
                    h5f[_qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, QTMDPDF_PZ, b, fixed_z)][:]
                    for b in B_ARR
                ],
                axis=0,
            )
        _assert_shape(pt3_zdep.shape, (len(Z_ARR), tsep + 2, N_CFG), f"qTMDPDF zdep ts{tsep}")
        _assert_shape(pt3_bdep.shape, (len(B_ARR), tsep + 2, N_CFG), f"qTMDPDF bdep ts{tsep}")

    softf = np.load(DATA_DIR / "fake_soft_function.npy", allow_pickle=True).item()
    _assert_shape(tuple(softf["softf_samp"].shape), (N_CFG, len(B_ARR)), "soft function")
    _assert_shape(tuple(softf["b_fm"].shape), (len(B_ARR),), "soft function b_fm")

    qtmdwf_count = _count_datasets(qtmdwf_path)
    expected_qtmdwf_count = len(QTMDWF_PZ_LIST) * len(B_ARR) * len(Z_ARR)
    if qtmdwf_count != expected_qtmdwf_count:
        raise AssertionError(
            f"qTMDWF dataset count: expected {expected_qtmdwf_count}, got {qtmdwf_count}"
        )

    pt3_counts = {}
    expected_pt3_count = len(B_ARR) * len(Z_ARR)
    for tsep in TSEP_LIST:
        pt3_path = DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5"
        count = _count_datasets(pt3_path)
        if count != expected_pt3_count:
            raise AssertionError(
                f"qTMDPDF ts{tsep} dataset count: expected {expected_pt3_count}, got {count}"
            )
        pt3_counts[tsep] = count

    print("Shape checks:")
    print(f"  qTMDWF raw: {raw_qtmdwf_shape}; reader shape: {(N_CFG, LT)}")
    print(f"  qTMDWF zdep stack: {qtmdwf_zdep.shape}")
    print(f"  qTMDWF bdep stack: {qtmdwf_bdep.shape}")
    print(f"  qTMDPDF 3pt raw by tsep: {pt3_shapes}")
    print(f"  qTMDPDF dataset counts by tsep: {pt3_counts}")
    print(f"  soft function: {softf['softf_samp'].shape}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    pt2_by_pz = {pz: _make_pt2(pz, rng) for pz in QTMDWF_PZ_LIST}
    _write_h5_datasets(
        DATA_DIR / "fake_2pt.h5",
        {
            f"{SOURCE_SINK_PT2}/{GAMMA_PT2}/{_momentum_key(pz)}": pt2
            for pz, pt2 in pt2_by_pz.items()
        },
    )

    qtmdwf_datasets = {}
    for pz in QTMDWF_PZ_LIST:
        for b in B_ARR:
            for z in Z_ARR:
                qtmdwf_datasets[
                    _qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, b, z)
                ] = _make_qtmdwf(pt2_by_pz[pz], pz=pz, b=int(b), z=int(z), rng=rng)
    _write_h5_datasets(DATA_DIR / "fake_qtmdwf.h5", qtmdwf_datasets)

    for tsep in TSEP_LIST:
        pt3_datasets = {}
        for b in B_ARR:
            for z in Z_ARR:
                pt3_datasets[
                    _qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, QTMDPDF_PZ, b, z)
                ] = _make_qtmdpdf_3pt(
                    pt2_by_pz[QTMDPDF_PZ],
                    tsep=tsep,
                    b=int(b),
                    z=int(z),
                    rng=rng,
                )
        _write_h5_datasets(DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5", pt3_datasets)

    np.save(DATA_DIR / "fake_soft_function.npy", _make_soft_function(rng))
    _run_shape_checks()

    print(f"Wrote fake data to {DATA_DIR}")
    print(f"2pt momenta: {list(pt2_by_pz)}, shape per momentum: {(LT, N_CFG)}")
    print(
        "qTMDWF datasets:",
        len(qtmdwf_datasets),
        f"= {len(QTMDWF_PZ_LIST)} momenta x {len(B_ARR)} b x {len(Z_ARR)} z",
    )
    print(
        "qTMDPDF 3pt datasets per tsep:",
        len(B_ARR) * len(Z_ARR),
        f"= {len(B_ARR)} b x {len(Z_ARR)} z",
    )


if __name__ == "__main__":
    main()
