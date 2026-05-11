import numpy as np
from matplotlib import pyplot as plt

from lqcd_analysis.data import EnsembleInfo, EnsembleData
from lqcd_analysis.perturbative_matching.coulomb_tmd_kernel import coulomb_tmdwf_kernel_rg_nll


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


def symmetrize_func(values, z_list, z_list_out, context):
    n = len(z_list)
    if z_list != list(range(n)):
        raise ValueError("z_list must be [0, 1, ..., n-1].")
    if z_list_out != list(range(-n, n)):
        raise ValueError("z_list_out must be [-n+1, ..., -1, 0, 1, ..., n-1].")
    return np.concatenate([np.zeros_like(values[:1]), values[:0:-1].conj(), values[:1].real, values[1:]])


def fourier_transform_func(values, z_list, x_list, context, inverse: bool = True):
    n = len(z_list)
    fourier_modes = np.arange(-n // 2, n // 2 + n % 2)
    if not np.allclose(z_list, fourier_modes):
        raise ValueError("z_list must be symmetric around 0 and have a length of n.")
    px = context["px"]
    kx = 2 * np.pi * px / 48
    lambda_list = [z * kx for z in z_list]
    if len(x_list) == n and np.allclose(x_list, fourier_modes * (2 * np.pi) / (n * kx)):
        values_out = np.fft.ifftshift(values, axes=0)
        if inverse:
            values_out = np.fft.ifft(values_out, axis=0)
            values_out *= n * kx / (2 * np.pi)
        else:
            values_out = np.fft.fft(values_out, axis=0)
            values_out *= kx
        values_out = np.fft.fftshift(values_out, axes=0)
    else:
        if inverse:
            kernel = np.exp(1j * np.outer(np.asarray(lambda_list), np.asarray(x_list)))
            kernel *= kx / (2 * np.pi)
        else:
            kernel = np.exp(-1j * np.outer(np.asarray(lambda_list), np.asarray(x_list)))
            kernel *= kx
        values_out = np.tensordot(values, kernel, axes=([0], [0]))
        values_out = np.moveaxis(values_out, -1, 0)
    return values_out


def cs_kernel_func(values, px_list, p1_p2_list, context):
    cs_kernel_p1_p2_list = []
    for p1_p2 in p1_p2_list:
        p1, p2 = p1_p2.strip("()").split(", ")
        p1, p2 = int(p1), int(p2)
        quasi_ft_p1 = values[px_list.index(p1)]
        quasi_ft_p2 = values[px_list.index(p2)]
        h_p1 = coulomb_tmdwf_kernel_rg_nll(x_list, p1 * ensemble_info.k_s)
        h_p2 = coulomb_tmdwf_kernel_rg_nll(x_list, p2 * ensemble_info.k_s)
        quasi_ft_ratio = quasi_ft_p2.real / quasi_ft_p1.real
        h_ratio = h_p2 / h_p1
        p_ratio = p2 / p1
        cs_kernel_p1_p2_list.append(np.log(quasi_ft_ratio / h_ratio) / np.log(p_ratio))
    return np.stack(cs_kernel_p1_p2_list, axis=0)


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
quasi_renorm = quasi_renorm.transform_dim("z", "z", z_list_full, symmetrize_func).real
quasi_ft = quasi_renorm.transform_dim("z", "x", x_list, fourier_transform_func, ["px"])
quasi_ft = quasi_ft.update_dim("x", "x", [x + 0.5 for x in x_list])


# CS kernel
x_list = np.linspace(0.1, 0.9, 81)
quasi_ft_x_list = quasi_ft.near("x", x_list.tolist())
x_list = np.asarray(quasi_ft_x_list.coords["x"])
p1_p2_list = [f"({p1}, {p2})" for p1 in px_list for p2 in px_list if p2 > p1]
cs_kernel = quasi_ft_x_list.transform_dim("px", "p1_p2", p1_p2_list, cs_kernel_func)

data = quasi_bare.at("px", px_pick).avg_data()
plt_errorbar(data, "z", "b", slice(1, None), (-0.5, 20.5))

data = quasi_ft.at("px", px_pick).avg_data()
plt_fill_between(data, "x", "b", slice(1, None), (-0.5, 1.5))

data = cs_kernel.at("p1_p2", "(8, 9)").avg_data()
plt_fill_between(data, "x", "b", slice(1, None), (0.3, 0.7), (-3, 3))
