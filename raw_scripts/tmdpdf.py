# %%
import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cs_kernel import fit_csk_for_b
from lametlat.correlators.resampling import bs_ls_avg
from lametlat.fourier_transform.core import complete_z_negative, sum_ft_re_im
from lametlat.fourier_transform.extrapolation import extrapolate_nucleon_cg_qpdf_nla
from lametlat.perturbative_matching.cg_tmd import CG_tmdpdf_kernel_RGR
from lametlat.plotting.plot_settings import (
    COLOR_CYCLE,
    ERRORBAR_STYLE,
    FONT_SIZE,
    default_plot,
)
from lametlat.utils.constants import GEV_FM, lat_unit_convert
from read_fake_data import (
    get_fake_qtmdpdf_samp,
    get_fake_qtmdwf_samp,
)


a = 0.08
Ls = 64
pz = 7
pz_gev = lat_unit_convert(pz, a, Ls, "P")
zmax = 16
zeta = 4
b_ls = np.arange(2, 16, 2)
x_ls = np.linspace(-2, 2, 1000)
x_mask = np.abs(x_ls) > 0.1

def load_b_qtmdpdf_zdep_gv(b):
    # normalization and renormalization
    
    bare_qtmdwf_p0_b0_z0 = bs_ls_avg(get_fake_qtmdwf_samp(pz=0, b=0, z=0)[0]) # real part only
    bare_qtmdwf_p0_z0 = bs_ls_avg(get_fake_qtmdwf_samp(pz=0, b=b, z=0)[0])
    
    renorm_factor = bare_qtmdwf_p0_z0 / bare_qtmdwf_p0_b0_z0
    
    bare_re = [bs_ls_avg(get_fake_qtmdpdf_samp(pz=pz, b=b, z=z)[0]) for z in range(zmax)]
    bare_im = [bs_ls_avg(get_fake_qtmdpdf_samp(pz=pz, b=b, z=z)[1]) for z in range(zmax)]
    renorm_pz_re = bare_re / bare_re[0] / renorm_factor
    renorm_pz_im = bare_im / bare_re[0] / renorm_factor
    
    return np.squeeze(renorm_pz_re), np.squeeze(renorm_pz_im)

def build_qtmdpdf_xdep_for_b(b):
    renorm_pz_re, renorm_pz_im = load_b_qtmdpdf_zdep_gv(b)
    lam_ls = np.arange(zmax) * 2 * a * pz_gev / GEV_FM
    
    (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    ) = extrapolate_nucleon_cg_qpdf_nla(
        lam_ls,
        renorm_pz_re,
        renorm_pz_im,
        [8, 15],
        100,
        weight_ini=0.0,
        m0=0.1 / pz_gev,
        label=f"fake_qtmdpdf_P{pz}_b{b}",
    )
    
    lam_full, re_full, im_full = complete_z_negative(
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        im_flip_for_ft=True,
    )
    quasi_re_gv = np.asarray(
        [sum_ft_re_im(lam_full, re_full, im_full, x)[0] for x in x_ls],
        dtype=object,
    )
    quasi_im_gv = np.asarray(
        [sum_ft_re_im(lam_full, re_full, im_full, x)[1] for x in x_ls],
        dtype=object,
    )
    
    return quasi_re_gv, quasi_im_gv

def read_csk_soft_function_for_b(b):
    csk_result = gv.load("./data/cs_kernel.gv")
    soft_function = np.loadtxt("./data/fake_bare_txt/soft_function/softf_hisq_mpi670.txt", delimiter=" ", skiprows=0)
    
    csk_b_fm = csk_result["b_fm"]
    csk_b_gv = csk_result["csk_gv"]
    
    soft_b_fm = soft_function[:, 0]
    soft_b_gv = gv.gvar(soft_function[:, 1], soft_function[:, 2])
    
    #todo should be an interpolation
    csk_b_idx = np.argmin(np.abs(csk_b_fm - b * a))
    soft_b_idx = np.argmin(np.abs(soft_b_fm - b * a))
    
    return csk_b_gv[csk_b_idx], soft_b_gv[soft_b_idx]

def matching_prefactor_for_b(b):
    csk_b_gv, soft_b_gv = read_csk_soft_function_for_b(b)
    x_used = x_ls[x_mask]
    hard_kernel = np.asarray(
        [CG_tmdpdf_kernel_RGR(float(x), pz_gev=pz_gev, mu=2) for x in x_used],
        dtype=float,
    )
    matching_prefactor = (
        np.sqrt(soft_b_gv)
        / hard_kernel
        / np.exp(0.5 * np.log((2 * x_used * pz_gev) ** 2 / zeta) * csk_b_gv)
    )
    return matching_prefactor


b = 2
renorm_pz_re, renorm_pz_im = load_b_qtmdpdf_zdep_gv(b)
print(np.shape(renorm_pz_re))
print(np.shape(renorm_pz_im))

fig, ax = default_plot()
ax.errorbar(np.arange(zmax), gv.mean(renorm_pz_re), gv.sdev(renorm_pz_re), label="Re renormalized qTMDPDF")
ax.set_xlabel(r"$z$", **FONT_SIZE)
ax.set_ylabel(r"Re renormalized qTMDPDF", **FONT_SIZE)
ax.set_title(r"Renormalized quasi TMDPDF, $P^z=$" + f"{pz_gev:.2f}" + r" GeV")
ax.legend(loc="upper right", ncol=4, fontsize=12)
plt.tight_layout()
plt.show()

quasi_re_gv, quasi_im_gv = build_qtmdpdf_xdep_for_b(b)
print(np.shape(quasi_re_gv))
print(np.shape(quasi_im_gv))

fig, ax = default_plot()
ax.fill_between(x_ls, gv.mean(quasi_re_gv) - gv.sdev(quasi_re_gv), gv.mean(quasi_re_gv) + gv.sdev(quasi_re_gv), label="Re quasi TMDPDF")
ax.set_xlim(-1, 1)
ax.set_xlabel(r"$x$", **FONT_SIZE)
ax.set_ylabel(r"Re quasi TMDPDF", **FONT_SIZE)
ax.set_title(r"Renormalized quasi TMDPDF, $P^z=$" + f"{pz_gev:.2f}" + r" GeV")
ax.legend(loc="upper right", ncol=4, fontsize=12)
plt.tight_layout()
plt.show()


csk_b_gv, soft_b_gv = read_csk_soft_function_for_b(b)
print(csk_b_gv)
print(soft_b_gv)

matching_prefactor = matching_prefactor_for_b(b)
lc_gv = np.zeros(len(x_ls), dtype=object)
lc_gv[x_mask] = quasi_re_gv[x_mask] * matching_prefactor

fig, ax = default_plot()
ax.fill_between(x_ls, gv.mean(lc_gv) - gv.sdev(lc_gv), gv.mean(lc_gv) + gv.sdev(lc_gv), label="Re lightcone TMDPDF")
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 3.5)
ax.set_xlabel(r"$x$", **FONT_SIZE)
ax.set_ylabel(r"Re lightcone TMDPDF", **FONT_SIZE)
ax.set_title(r"Lightcone TMDPDF from fake qTMDPDF, $P^z=$" + f"{pz_gev:.2f}" + r" GeV")
ax.legend(loc="upper right", ncol=4, fontsize=12)
plt.tight_layout()
plt.show()



# %%
