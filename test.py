import numpy as np
from matplotlib import pyplot as plt

from lqcd_analysis.data import EnsembleInfo, EnsembleData
from lqcd_analysis.perturbative_matching.coulomb_tmd_kernel_rg_resum_nll import coulomb_tmdwf_kernel_rg_resum_nll


def load_quasi_p0_re(ensemble_info, b_list, z_list):
    import gvar

    quasi_p0_mean = np.zeros((len(b_list), len(z_list)), "<f8")
    quasi_p0_sdev = np.zeros((len(b_list), len(z_list)), "<f8")
    for b_idx, b in enumerate(b_list):
        re = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p0_b{b}_re_meansdev.txt")
        for z_idx, z in enumerate(z_list):
            quasi_p0_mean[b_idx, z_idx] = re[z, 0]
            quasi_p0_sdev[b_idx, z_idx] = re[z, 1]
    return EnsembleData(
        ensemble_info,
        "gvar",
        gvar.gvar(quasi_p0_mean, quasi_p0_sdev),
        dims=["b", "z"],
        coords={"b": b_list, "z": z_list},
    )


def load_quasi(ensemble_info, n_jk, px_list, b_list, z_list):
    quasi = np.zeros((n_jk, len(px_list), len(b_list), len(z_list)), "<c16")
    for px_idx, px in enumerate(px_list):
        for b_idx, b in enumerate(b_list):
            re = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p{px}_b{b}_re.txt")
            im = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p{px}_b{b}_im.txt")
            for z_idx, z in enumerate(z_list):
                for jk in range(n_jk):
                    quasi[jk, px_idx, b_idx, z_idx] = re[z, jk] + 1j * im[z, jk]
    return EnsembleData(
        ensemble_info,
        "jackknife",
        list(quasi),
        dims=["px", "b", "z"],
        coords={"px": px_list, "b": b_list, "z": z_list},
    )


def symmetrize(value, coord, coord_out, context):
    return np.concatenate([np.zeros_like(value[:1]), value[:0:-1].conj(), value[:1].real, value[1:]])


def plt_errorbar(data, x_dim, label_dim, label_slice, xlim, ylim=None):
    x = data.coords[x_dim]
    for label in data.coords[label_dim][label_slice]:
        mean = data.at(label_dim, label).mean
        sdev = data.at(label_dim, label).sdev
        plt.errorbar(x, mean, sdev, fmt="x", label=f"{label_dim}={label}")
    plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])
    plt.legend()
    plt.show()
    plt.clf()


def plt_fill_between(data, x_dim, label_dim, label_slice, xlim, ylim=None):
    x = data.coords[x_dim]
    for label in data.coords[label_dim][label_slice]:
        mean = data.at(label_dim, label).mean
        sdev = data.at(label_dim, label).sdev
        plt.fill_between(x, mean - sdev, mean + sdev, alpha=0.3, label=f"{label_dim}={label}")
    plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()
    plt.clf()


ensemble_info = EnsembleInfo("", "", 0.06, 0.06, 48, 64, 300)
px_list = [8, 9, 10]
b_list = [0, 2, 4, 6, 8, 10]
z_list = list(range(21))
z_list_full = list(range(-21, 21))
px_pick = 8

# Load
quasi_p0_re = load_quasi_p0_re(ensemble_info, b_list, z_list)
quasi_bare = load_quasi(ensemble_info, 533, px_list, b_list, z_list)

# Normalize
quasi_bare_re_b0_z0 = quasi_bare.at("b", 0).at("z", 0).avg_data()
quasi_bare = quasi_bare.div(quasi_bare_re_b0_z0)

# Renormalize
quasi_p0_z0_re = quasi_p0_re.at("z", 0)
quasi_renorm = quasi_bare.div(quasi_p0_z0_re)

# Fourier transform
x_list = np.linspace(-1, 1, 201).tolist()
quasi_ft_px_list = []
for px in px_list:
    kx = 2 * np.pi * px / 48
    quasi_renorm_px = quasi_renorm.at("px", px)
    quasi_renorm_px = quasi_renorm_px.estimate_dim("z", z_list_full, symmetrize).real
    quasi_renorm_px = quasi_renorm_px.update_dim("z", "lambda", [z * kx for z in z_list_full])
    quasi_ft_px = quasi_renorm_px.fourier_transform_dim("lambda", "x", x_list, kx)
    quasi_ft_px = quasi_ft_px.update_dim("x", "x", [x + 0.5 for x in x_list])
    quasi_ft_px_list.append(quasi_ft_px)
quasi_ft = EnsembleData.concat(quasi_ft_px_list, "px", px_list)

# CS kernel
x_list = np.linspace(0.1, 0.9, 81)
p1_p2_list = []
cs_kernel_p1_p2_list = []
for p1_idx, p1 in enumerate(px_list):
    for p2_idx, p2 in enumerate(px_list[p1_idx + 1 :]):
        quasi_ft_p1 = quasi_ft.near("x", x_list.tolist()).at("px", p1)
        quasi_ft_p2 = quasi_ft.near("x", x_list.tolist()).at("px", p2)
        h_p1 = coulomb_tmdwf_kernel_rg_resum_nll(x_list, p1 * ensemble_info.k_s)
        h_p2 = coulomb_tmdwf_kernel_rg_resum_nll(x_list, p2 * ensemble_info.k_s)
        quasi_ft_ratio = quasi_ft_p2.array.real / quasi_ft_p1.array.real
        h_ratio = h_p2 / h_p1
        p_ratio = p2 / p1
        result_p1_p2 = np.log(quasi_ft_ratio / h_ratio) / np.log(p_ratio)
        p1_p2_list.append(f"({p1}, {p2})")
        cs_kernel_p1_p2_list.append(EnsembleData._from_xarray(ensemble_info, "jackknife", result_p1_p2))
cs_kernel = EnsembleData.concat(cs_kernel_p1_p2_list, "p1_p2", p1_p2_list)

data = quasi_bare.at("px", px_pick).avg_data()
plt_errorbar(data, "z", "b", slice(1, None), (-0.5, 20.5))

data = quasi_ft.at("px", px_pick).avg_data()
plt_fill_between(data, "x", "b", slice(1, None), (-0.5, 1.5))

data = cs_kernel.at("p1_p2", "(8, 9)").avg_data()
plt_fill_between(data, "x", "b", slice(1, None), (0.3, 0.7), (-3, 3))
