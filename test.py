import numpy as np
import gvar as gv


from lqcd_analysis.data import EnsembleInfo, EnsembleData, GlobalData

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

quasi_p0 = GlobalData(
    values=1 / gv.gvar(quasi_p0_mean, quasi_p0_sdev),
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
    ensemble=ensemble_info,
    values=[quasi[jk].real for jk in range(533)],
    dims=["px", "b", "z"],
    coords={"px": px_list, "b": b_list, "z": z_list},
)

print(quasi_bare_re.gvar())
quasi_renorm_re = quasi_bare_re.multiplicative_renormalization(quasi_p0)
print(quasi_renorm_re.gvar())

quasi_ft_re = quasi_renorm_re.spatial_fourier_transform("z", "xPz")
quasi_ft_re_p8 = quasi_ft_re.at("px", 8)
quasi_ft_re_p8.update_dim("xPz", "x", lambda x: x / (2 * np.pi * 8 / 48))
