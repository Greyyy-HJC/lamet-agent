import numpy as np
from matplotlib import pyplot as plt


from lqcd_analysis.data import EnsembleInfo, EnsembleData


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


def plt_errorbar(data, x_dim, label_dim, label_slice, xlim):
    x = data.coords[x_dim]
    for label in data.coords[label_dim][label_slice]:
        mean = data.at(label_dim, label).mean
        sdev = data.at(label_dim, label).sdev
        plt.errorbar(x, mean, sdev, fmt="x", label=f"{label_dim}={label}")
    plt.xlim(xlim[0], xlim[1])
    plt.legend()
    plt.show()
    plt.clf()


def plt_fill_between(data, x_dim, label_dim, label_slice, xlim):
    x = data.coords[x_dim]
    for label in data.coords[label_dim][label_slice]:
        mean = data.at(label_dim, label).mean
        sdev = data.at(label_dim, label).sdev
        plt.fill_between(x, mean - sdev, mean + sdev, alpha=0.3, label=f"{label_dim}={label}")
    plt.xlim(xlim[0], xlim[1])
    plt.legend()
    plt.show()
    plt.clf()


ensemble_info = EnsembleInfo("", "", 0.06, 0.06, 48, 64, 300)
px_list = [8, 9, 10]
b_list = [0, 2, 4, 6, 8, 10]
z_list = list(range(21))
px_pick = 8

quasi_p0_re = load_quasi_p0_re(ensemble_info, b_list, z_list)
quasi_bare = load_quasi(ensemble_info, 533, px_list, b_list, z_list)

quasi_bare_re_b0_z0 = quasi_bare.at("b", 0).at("z", 0).avg_data()
quasi_bare = quasi_bare.div(quasi_bare_re_b0_z0)
quasi_bare_px = quasi_bare.at("px", px_pick)

quasi_p0_z0 = quasi_p0_re.at("z", 0)
quasi_renorm = quasi_bare.div(quasi_p0_z0)
quasi_renorm_px = quasi_renorm.at("px", px_pick)
quasi_renorm_px.padding_dim("z", 100)
quasi_renorm_px.symmetric_dim("z")

quasi_ft_px = quasi_renorm_px.fourier_transform_dim("z", "xPz")
quasi_ft_px.update_value("xPz", lambda v: v * (2 * np.pi * px_pick / 48))
quasi_ft_px.update_dim("xPz", lambda x: x / (2 * np.pi * px_pick / 48) + 0.5, "x")

quasi_renorm_px.update_dim("z", lambda z: z * (2 * np.pi * px_pick / 48), "lambda")
quasi_ft_px_v2 = quasi_renorm_px.fourier_transform_dim("lambda", "x", (2 * np.pi * px_pick / 48))
quasi_ft_px_v2.update_dim("x", lambda x: x + 0.5)

plt_errorbar(quasi_bare_px.avg_data(), "z", "b", slice(1, None), (-0.5, 20.5))

plt_fill_between(quasi_ft_px.avg_data(), "x", "b", slice(1, None), (-0.5, 1.5))
plt_fill_between(quasi_ft_px_v2.avg_data(), "x", "b", slice(1, None), (-0.5, 1.5))
