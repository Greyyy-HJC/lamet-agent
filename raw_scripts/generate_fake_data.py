"""Generate dual-path fake data for ground-state fit and bare matrix tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

LT = 32
N_CFG = 64
SEED = 20260507
TSEP_LIST = (4, 6, 8, 10)

# Part A: compact H5 set for ground-state fit checks.
H5_PZ_LIST = (0,)
H5_PT3_PZ = 0
H5_B_ARR = np.array([0], dtype=int)
H5_Z_ARR = np.arange(16, dtype=int)

# Part B: bare-matrix TXT bundle for CS-kernel style workflows.
BARE_QTMDWF_PZ_LIST = (0, 5, 6, 7)
BARE_QTMDPDF_PZ_LIST = (0, 7)
BARE_B_ARR = np.arange(0, 16, 2, dtype=int)
BARE_Z_ARR = np.arange(16, dtype=int)
SOFT_B_FM = 0.06 * np.arange(2, 19, dtype=float)
SOFT_TXT_NAME = "softf_hisq_mpi670.txt"

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
BARE_TXT_DIR = DATA_DIR / "fake_bare_txt"


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
    z_width = (3.4 + 0.03 * pz) / (1.0 + 0.025 * b)
    b_width = 8.5
    z_decay = np.exp(-0.5 * (z / z_width) ** 2)
    b_decay = np.exp(-0.5 * (b / b_width) ** 2)
    cs_like = np.exp(-0.050 * b * np.log((pz + 2.0) / 2.0))
    bt_z_decay = np.exp(-0.0035 * b * z * (1.0 + 0.30 * np.log((pz + 2.0) / 2.0)))
    return float(z_decay * b_decay * cs_like * bt_z_decay)


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


def _standardized_noise(rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=N_CFG)
    noise -= np.mean(noise)
    sdev = np.std(noise, ddof=1)
    if sdev == 0:
        return np.zeros(N_CFG, dtype=float)
    return noise / sdev


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _complex_samples(
    re_mean: float,
    im_mean: float,
    re_sigma: float,
    im_sigma: float,
    rng: np.random.Generator,
    *,
    rho: float,
) -> np.ndarray:
    shared = _standardized_noise(rng)
    independent = _standardized_noise(rng)
    imag_noise = rho * shared + np.sqrt(max(1.0 - rho**2, 0.0)) * independent
    imag_noise -= np.mean(imag_noise)
    imag_sdev = np.std(imag_noise, ddof=1)
    if imag_sdev > 0:
        imag_noise /= imag_sdev
    real = re_mean + re_sigma * shared
    imag = im_mean + im_sigma * imag_noise
    return real + 1j * imag


def _make_soft_function_table() -> np.ndarray:
    b_fm = SOFT_B_FM
    tail_floor = 0.036 * _sigmoid((b_fm - 0.94) / 0.055)
    mean = 1.34 * np.exp(-3.40 * b_fm**1.25) + tail_floor
    err = mean * (0.090 + 0.160 * b_fm) + 0.0060 * np.exp(-b_fm / 0.35)
    return np.column_stack((b_fm, mean, err))


def _make_bare_qtmdwf(pz: int, b: int, z: int, rng: np.random.Generator) -> np.ndarray:
    b_decay = np.exp(-0.115 * b - 0.0015 * b**2)
    z_width = max(4.6 - 0.18 * (pz - 5), 3.2) / (1.0 + 0.018 * b)
    leading = 0.106 * b_decay * np.exp(-0.5 * (z / z_width) ** 2)
    high_pz = max(pz - 4, 0) / 3.0
    tail = (
        0.0024
        * high_pz
        * (1.0 + 0.20 * (pz - 5))
        * np.exp(-0.018 * b)
        * _sigmoid((z - (9.2 - 0.10 * b)) / 0.85)
        * np.exp(-0.030 * max(z - 12, 0))
    )
    re_mean = leading - tail

    im_amp = 0.0022 * high_pz * np.exp(-0.070 * b)
    im_mean = im_amp * np.sin(0.54 * z + 0.63 * (pz - 5) - 0.085 * b)
    im_mean += 5.0e-5 * np.sin(0.70 * b + 0.24 * z) * np.exp(-0.08 * b)

    re_sigma = (
        0.00011
        + 0.046 * max(abs(re_mean), 0.0008)
        + 0.00024 * high_pz * _sigmoid((z - 8.0) / 1.4)
    )
    re_sigma *= 1.0 + 0.08 * np.sin(0.30 * b + 0.20 * z) ** 2
    im_sigma = (
        1.0e-5
        + 0.00015 * high_pz
        + 0.00036 * high_pz * _sigmoid((z - 2.5) / 2.0)
        + 0.000030 * b / 14.0
    )
    return _complex_samples(re_mean, im_mean, re_sigma, im_sigma, rng, rho=0.35)


def _make_bare_qtmdpdf(pz: int, b: int, z: int, rng: np.random.Generator) -> np.ndarray:
    b_decay = np.exp(-0.115 * b - 0.0040 * b**2)
    if pz == 0:
        core = 1.94 * b_decay * np.exp(-0.5 * (z / 3.25) ** 2)
        tail = 0.095 * np.exp(-0.055 * b) * _sigmoid((z - 11.6) / 1.1)
        dip = 0.045 * np.exp(-0.040 * b) * np.exp(-0.5 * ((z - 9.0) / 1.55) ** 2)
        re_mean = core + tail - dip
        im_mean = (
            0.070
            * np.exp(-0.045 * b)
            * np.exp(-0.5 * ((z - 8.5) / 3.1) ** 2)
            * np.sin(0.61 * z - 0.05 * b)
        )
    else:
        z_damp = np.exp(-0.5 * (z / 4.1) ** 2)
        re_mean = (
            1.93
            * np.exp(-0.105 * b - 0.0022 * b**2)
            * z_damp
            * np.cos(0.56 * z + 0.020 * b)
        )
        re_mean += 0.045 * np.exp(-0.065 * b) * _sigmoid((z - 10.5) / 1.2)
        im_mean = (
            1.08
            * np.exp(-0.090 * b - 0.0020 * b**2)
            * np.exp(-0.5 * (z / 4.4) ** 2)
            * np.sin(0.56 * z + 0.025 * b)
        )
        im_mean -= 0.090 * np.exp(-0.045 * b) * np.exp(-0.5 * ((z - 12.7) / 2.0) ** 2)

    tail_boost = max(z - 10, 0)
    cap = 0.025 + 0.080 * np.exp(-float(z) / 8.0)
    b_suppression = 0.80 + 0.20 * np.exp(-0.030 * b)
    re_sigma = cap * b_suppression * (1.0 + 0.05 * np.sin(0.35 * b + 0.40 * z))
    im_sigma = (
        1.0e-17
        if z == 0
        else (0.018 + 0.058 * np.exp(-float(z) / 9.0) + 0.0020 * tail_boost)
        * (0.85 + 0.15 * np.exp(-0.050 * b))
    )
    return _complex_samples(re_mean, im_mean, re_sigma, im_sigma, rng, rho=0.25)


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


def _write_complex_txt(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    two_col = np.column_stack((values.real, values.imag))
    np.savetxt(path, two_col, fmt="%.10e")


def _write_real_txt(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%.10e")


def _clean_txt_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for txt_path in path.glob("*.txt"):
        txt_path.unlink()


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


def _expected_h5_qtmdwf_paths() -> list[str]:
    paths = []
    for pz in H5_PZ_LIST:
        for b in H5_B_ARR:
            for z in H5_Z_ARR:
                paths.append(
                    _qtmd_path(
                        SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, int(pz), int(b), int(z)
                    )
                )
    return paths


def _expected_h5_pt3_paths(tsep: int) -> list[str]:
    paths = []
    for b in H5_B_ARR:
        for z in H5_Z_ARR:
            paths.append(_qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, H5_PT3_PZ, int(b), int(z)))
    return paths


def _h5_dataset_paths(path: Path) -> set[str]:
    out: set[str] = set()
    with h5py.File(path, "r") as h5f:
        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                out.add(name)

        h5f.visititems(visit)
    return out


def _run_shape_checks() -> None:
    qtmdwf_path = DATA_DIR / "fake_qtmdwf.h5"
    pz = int(H5_PZ_LIST[0])
    fixed_b = int(H5_B_ARR[0])
    fixed_z = int(H5_Z_ARR[0])
    raw_qtmdwf_shape = _dataset_shape(
        qtmdwf_path,
        _qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, fixed_b, fixed_z),
    )
    _assert_shape(raw_qtmdwf_shape, (LT, N_CFG), "qTMDWF raw")

    with h5py.File(qtmdwf_path, "r") as h5f:
        qtmdwf_zdep = np.stack(
            [
                h5f[_qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, fixed_b, z)][:]
                for z in H5_Z_ARR
            ],
            axis=0,
        )
        qtmdwf_bdep = np.stack(
            [
                h5f[_qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, pz, b, fixed_z)][:]
                for b in H5_B_ARR
            ],
            axis=0,
        )
    _assert_shape(qtmdwf_zdep.shape, (len(H5_Z_ARR), LT, N_CFG), "qTMDWF zdep")
    _assert_shape(qtmdwf_bdep.shape, (len(H5_B_ARR), LT, N_CFG), "qTMDWF bdep")

    pt3_shapes: dict[int, tuple[int, ...]] = {}
    for tsep in TSEP_LIST:
        pt3_path = DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5"
        pt3_dataset = _qtmd_path(
            SOURCE_SINK_PT3, GAMMA_PT3, H5_PT3_PZ, fixed_b, fixed_z
        )
        pt3_shapes[tsep] = _dataset_shape(pt3_path, pt3_dataset)
        _assert_shape(pt3_shapes[tsep], (tsep + 2, N_CFG), f"qTMDPDF 3pt ts{tsep}")

        with h5py.File(pt3_path, "r") as h5f:
            pt3_zdep = np.stack(
                [
                    h5f[_qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, H5_PT3_PZ, fixed_b, z)][:]
                    for z in H5_Z_ARR
                ],
                axis=0,
            )
            pt3_bdep = np.stack(
                [
                    h5f[_qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, H5_PT3_PZ, b, fixed_z)][:]
                    for b in H5_B_ARR
                ],
                axis=0,
            )
        _assert_shape(
            pt3_zdep.shape, (len(H5_Z_ARR), tsep + 2, N_CFG), f"qTMDPDF zdep ts{tsep}"
        )
        _assert_shape(
            pt3_bdep.shape, (len(H5_B_ARR), tsep + 2, N_CFG), f"qTMDPDF bdep ts{tsep}"
        )

    qtmdwf_count = _count_datasets(qtmdwf_path)
    expected_qtmdwf_count = len(H5_PZ_LIST) * len(H5_B_ARR) * len(H5_Z_ARR)
    if qtmdwf_count != expected_qtmdwf_count:
        raise AssertionError(
            f"qTMDWF dataset count: expected {expected_qtmdwf_count}, got {qtmdwf_count}"
        )
    if _h5_dataset_paths(qtmdwf_path) != set(_expected_h5_qtmdwf_paths()):
        raise AssertionError("qTMDWF H5 datasets do not match expected p/b/z grid")

    pt3_counts = {}
    expected_pt3_count = len(H5_B_ARR) * len(H5_Z_ARR)
    for tsep in TSEP_LIST:
        pt3_path = DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5"
        count = _count_datasets(pt3_path)
        if count != expected_pt3_count:
            raise AssertionError(
                f"qTMDPDF ts{tsep} dataset count: expected {expected_pt3_count}, got {count}"
            )
        pt3_counts[tsep] = count
        if _h5_dataset_paths(pt3_path) != set(_expected_h5_pt3_paths(tsep)):
            raise AssertionError(f"qTMDPDF ts{tsep} H5 datasets do not match expected grid")

    two_pt_path = DATA_DIR / "fake_2pt.h5"
    two_pt_expected = {
        f"{SOURCE_SINK_PT2}/{GAMMA_PT2}/{_momentum_key(pz)}" for pz in H5_PZ_LIST
    }
    if _h5_dataset_paths(two_pt_path) != two_pt_expected:
        raise AssertionError("2pt H5 datasets do not match expected momentum set")

    print("Shape checks:")
    print(f"  qTMDWF raw: {raw_qtmdwf_shape}; reader shape: {(N_CFG, LT)}")
    print(f"  qTMDWF zdep stack: {qtmdwf_zdep.shape}")
    print(f"  qTMDWF bdep stack: {qtmdwf_bdep.shape}")
    print(f"  qTMDPDF 3pt raw by tsep: {pt3_shapes}")
    print(f"  qTMDPDF dataset counts by tsep: {pt3_counts}")


def _run_txt_checks() -> None:
    qtmdwf_files = list((BARE_TXT_DIR / "qtmdwf").glob("pz*_b*_z*.txt"))
    qtmdpdf_files = list((BARE_TXT_DIR / "qtmdpdf").glob("pz*_b*_z*.txt"))
    soft_files = list((BARE_TXT_DIR / "soft_function").glob("*.txt"))
    expected_qtmdwf = len(BARE_QTMDWF_PZ_LIST) * len(BARE_B_ARR) * len(BARE_Z_ARR)
    expected_qtmdpdf = len(BARE_QTMDPDF_PZ_LIST) * len(BARE_B_ARR) * len(BARE_Z_ARR)
    if len(qtmdwf_files) != expected_qtmdwf:
        raise AssertionError(
            f"qtmdwf TXT count: expected {expected_qtmdwf}, got {len(qtmdwf_files)}"
        )
    if len(qtmdpdf_files) != expected_qtmdpdf:
        raise AssertionError(
            f"qtmdpdf TXT count: expected {expected_qtmdpdf}, got {len(qtmdpdf_files)}"
        )
    if soft_files != [BARE_TXT_DIR / "soft_function" / SOFT_TXT_NAME]:
        raise AssertionError(f"soft TXT files: expected only {SOFT_TXT_NAME}, got {soft_files}")

    qtmdwf_sample = np.loadtxt(BARE_TXT_DIR / "qtmdwf" / "pz0_b0_z0.txt")
    qtmdpdf_sample = np.loadtxt(BARE_TXT_DIR / "qtmdpdf" / "pz0_b0_z0.txt")
    soft_sample = np.loadtxt(BARE_TXT_DIR / "soft_function" / SOFT_TXT_NAME)
    _assert_shape(tuple(np.atleast_2d(qtmdwf_sample).shape), (N_CFG, 2), "qtmdwf TXT sample")
    _assert_shape(tuple(np.atleast_2d(qtmdpdf_sample).shape), (N_CFG, 2), "qtmdpdf TXT sample")
    _assert_shape(tuple(np.atleast_2d(soft_sample).shape), (len(SOFT_B_FM), 3), "soft TXT")
    if not np.all(np.isfinite(soft_sample)):
        raise AssertionError("soft_function TXT table must be finite")
    if not np.all(soft_sample[:, 1:] > 0):
        raise AssertionError("soft_function mean and err columns must stay positive")
    if not np.all(np.diff(soft_sample[:, 0]) > 0):
        raise AssertionError("soft_function b_fm column must be strictly increasing")


def _write_manifest() -> None:
    manifest = (
        "fake_bare_txt layout\n"
        "  qtmdwf/pz{pz}_b{b}_z{z}.txt -> two columns: real imag (N_CFG rows)\n"
        "  qtmdpdf/pz{pz}_b{b}_z{z}.txt -> two columns: real imag (N_CFG rows)\n"
        f"  soft_function/{SOFT_TXT_NAME} -> three columns: b_fm mean err\n"
        "source=standalone analytic model; no external data files are read at runtime\n"
        f"seed={SEED}\n"
        f"qtmdwf_pz={BARE_QTMDWF_PZ_LIST}\n"
        f"qtmdpdf_pz={BARE_QTMDPDF_PZ_LIST}\n"
        f"b_range={list(BARE_B_ARR)}\n"
        f"z_range={list(BARE_Z_ARR)}\n"
        f"soft_b_fm_range={list(SOFT_B_FM)}\n"
    )
    (BARE_TXT_DIR / "manifest.txt").parent.mkdir(parents=True, exist_ok=True)
    (BARE_TXT_DIR / "manifest.txt").write_text(manifest, encoding="utf-8")


def generate_h5_ground_state_subset(rng: np.random.Generator) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    pt2_by_pz = {pz: _make_pt2(pz, rng) for pz in H5_PZ_LIST}
    _write_h5_datasets(
        DATA_DIR / "fake_2pt.h5",
        {
            f"{SOURCE_SINK_PT2}/{GAMMA_PT2}/{_momentum_key(pz)}": pt2
            for pz, pt2 in pt2_by_pz.items()
        },
    )

    qtmdwf_datasets = {}
    for pz in H5_PZ_LIST:
        for b in H5_B_ARR:
            for z in H5_Z_ARR:
                qtmdwf_datasets[
                    _qtmd_path(SOURCE_SINK_QTMDWF, GAMMA_QTMDWF, int(pz), int(b), int(z))
                ] = _make_qtmdwf(pt2_by_pz[pz], pz=pz, b=int(b), z=int(z), rng=rng)
    _write_h5_datasets(DATA_DIR / "fake_qtmdwf.h5", qtmdwf_datasets)

    for tsep in TSEP_LIST:
        pt3_datasets = {}
        for b in H5_B_ARR:
            for z in H5_Z_ARR:
                pt3_datasets[
                    _qtmd_path(SOURCE_SINK_PT3, GAMMA_PT3, H5_PT3_PZ, int(b), int(z))
                ] = _make_qtmdpdf_3pt(
                    pt2_by_pz[H5_PT3_PZ], tsep=tsep, b=int(b), z=int(z), rng=rng
                )
        _write_h5_datasets(DATA_DIR / f"fake_qtmdpdf_3pt_ts{tsep}.h5", pt3_datasets)

    return pt2_by_pz, qtmdwf_datasets


def generate_bare_txt_bundle(rng: np.random.Generator) -> tuple[int, int, int]:
    _clean_txt_dir(BARE_TXT_DIR / "qtmdwf")
    _clean_txt_dir(BARE_TXT_DIR / "qtmdpdf")
    _clean_txt_dir(BARE_TXT_DIR / "soft_function")

    for pz in BARE_QTMDWF_PZ_LIST:
        for b in BARE_B_ARR:
            for z in BARE_Z_ARR:
                wf = _make_bare_qtmdwf(pz=int(pz), b=int(b), z=int(z), rng=rng)
                _write_complex_txt(BARE_TXT_DIR / "qtmdwf" / f"pz{pz}_b{b}_z{z}.txt", wf)

    for pz in BARE_QTMDPDF_PZ_LIST:
        for b in BARE_B_ARR:
            for z in BARE_Z_ARR:
                pdf = _make_bare_qtmdpdf(pz=int(pz), b=int(b), z=int(z), rng=rng)
                _write_complex_txt(BARE_TXT_DIR / "qtmdpdf" / f"pz{pz}_b{b}_z{z}.txt", pdf)

    _write_real_txt(
        BARE_TXT_DIR / "soft_function" / SOFT_TXT_NAME,
        _make_soft_function_table(),
    )

    _write_manifest()
    return (
        len(BARE_QTMDWF_PZ_LIST) * len(BARE_B_ARR) * len(BARE_Z_ARR),
        len(BARE_QTMDPDF_PZ_LIST) * len(BARE_B_ARR) * len(BARE_Z_ARR),
        1,
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    pt2_by_pz, qtmdwf_datasets = generate_h5_ground_state_subset(rng)
    qtmdwf_txt_count, qtmdpdf_txt_count, soft_txt_count = generate_bare_txt_bundle(rng)
    _run_shape_checks()
    _run_txt_checks()

    print(f"Wrote fake data to {DATA_DIR}")
    print(f"2pt momenta: {list(pt2_by_pz)}, shape per momentum: {(LT, N_CFG)}")
    print(
        "qTMDWF datasets:",
        len(qtmdwf_datasets),
        f"= {len(H5_PZ_LIST)} momentum x {len(H5_B_ARR)} b x {len(H5_Z_ARR)} z",
    )
    print(
        "qTMDPDF 3pt datasets per tsep:",
        len(H5_B_ARR) * len(H5_Z_ARR),
        f"= {len(H5_B_ARR)} b x {len(H5_Z_ARR)} z",
    )
    print(f"Bare TXT qTMDWF files: {qtmdwf_txt_count}")
    print(f"Bare TXT qTMDPDF files: {qtmdpdf_txt_count}")
    print(f"Bare TXT soft_function files: {soft_txt_count}")


if __name__ == "__main__":
    main()
