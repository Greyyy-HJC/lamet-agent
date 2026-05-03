import numpy as np
from matplotlib import pyplot as plt
import gvar as gv


from lqcd_analysis.data import EnsembleInfo, EnsembleData

ensemble_info = EnsembleInfo("", "", 0.06, 0.06, 48, 64, 300)
px_list = [8, 9, 10]
b_list = [0, 2, 4, 6, 8, 10]
z_list = list(range(21))

quasi_p0_mean = np.zeros((len(b_list), len(z_list)), "<f8")
quasi_p0_sdev = np.zeros((len(b_list), len(z_list)), "<f8")

for b_idx, b in enumerate(b_list):
    re = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p0_b{b}_re_meansdev.txt")
    for z_idx, z in enumerate(z_list):
        quasi_p0_mean[b_idx, z_idx] = re[z, 0]
        quasi_p0_sdev[b_idx, z_idx] = re[z, 1]

quasi_p0 = EnsembleData(
    ensemble_info,
    "gvar",
    gv.gvar(quasi_p0_mean, quasi_p0_sdev),
    dims=["b", "z"],
    coords={"b": b_list, "z": z_list},
)

quasi = np.zeros((533, len(px_list), len(b_list), len(z_list)), "<c16")

for px_idx, px in enumerate(px_list):
    for b_idx, b in enumerate(b_list):
        re = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p{px}_b{b}_re.txt")
        im = np.loadtxt(f"./pion_cg_cs_kernel/bare_quasi_p{px}_b{b}_im.txt")
        for z_idx, z in enumerate(z_list):
            for jk in range(533):
                quasi[jk, px_idx, b_idx, z_idx] = re[z, jk] + 1j * im[z, jk]

quasi_bare_re = EnsembleData(
    ensemble_info,
    "jackknife",
    [quasi[jk].real for jk in range(533)],
    dims=["px", "b", "z"],
    coords={"px": px_list, "b": b_list, "z": z_list},
)

quasi_p0_z0 = quasi_p0.at("z", 0)
quasi_renorm_re = quasi_bare_re.div(quasi_p0_z0)

quasi_renorm_re_b0_z0 = quasi_renorm_re.at("b", 0).at("z", 0).avg_data()
quasi_renorm_re = quasi_renorm_re.div(quasi_renorm_re_b0_z0)
quasi_renorm_re_p8 = quasi_renorm_re.at("px", 8)

quasi_ft_re = quasi_renorm_re.spatial_fourier_transform("z", "xPz")
quasi_ft_re_p8 = quasi_ft_re.at("px", 8)
quasi_ft_re_p8.update_dim("xPz", lambda x: x / 8 + 0.5, "x")
quasi_ft_re_p8.update_value("x", lambda x: x * 8)
quasi_ft_re_p8.sort_dim("x")

data = quasi_renorm_re_p8.gvar
mean = gv.mean(data)
sdev = gv.sdev(data)
z = quasi_renorm_re_p8.coords["z"]
for b_idx, b in enumerate(b_list):
    plt.errorbar(z, mean[b_idx], sdev[b_idx], fmt="x", label=f"b={b}")
plt.xlim(-0.5, 20.5)
plt.legend()
plt.show()
plt.clf()

data = quasi_ft_re_p8.gvar
mean = gv.mean(data)
sdev = gv.sdev(data)
x = quasi_ft_re_p8.coords["x"]
for b_idx, b in enumerate(b_list):
    plt.fill_between(x, mean[b_idx] - sdev[b_idx], mean[b_idx] + sdev[b_idx], alpha=0.3, label=f"b={b}")
plt.xlim(-0.5, 1.5)
plt.legend()
plt.show()
plt.clf()
