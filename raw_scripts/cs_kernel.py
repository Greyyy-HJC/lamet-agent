# %%
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from lametlat.correlators.resampling import bs_ls_avg
from lametlat.fourier_transform.core import complete_z_negative, sum_ft_re_im
from lametlat.perturbative_matching.cg_tmd import CG_tmdwf_kernel_RGR
from lametlat.plotting.plot_settings import ERRORBAR_STYLE, default_plot
from lametlat.utils.constants import GEV_FM
from lametlat.utils.core import constant_fit
from read_fake_data import get_fake_qtmdwf_samp


a = 0.08
zmax = 16
pz_fit_ls = np.array([5, 6, 7])
pz_gev_dict = {5: 1.16, 6: 1.39, 7: 1.62, 8: 1.85}
b_ls = np.arange(2, 16, 2)

x_raw = np.linspace(-2, 2, 1000)
x_ls = x_raw + 0.5


def broad_csk_prior():
    priors = gv.BufferDict()
    priors["A"] = gv.gvar(1, 10)
    priors["csk"] = gv.gvar(-0.5, 1)
    return priors


def make_csk_fcn(x, pz_gev_ls, fix_pz=1.62):
    hard_kernel = np.asarray(
        [CG_tmdwf_kernel_RGR(x, pz_gev=pz_gev, mu=2) for pz_gev in pz_gev_ls],
        dtype=float,
    )
    log_ratio = np.log(np.asarray(pz_gev_ls, dtype=float) / fix_pz)

    def fcn(_pz_gev, prior):
        return prior["A"] * hard_kernel * np.exp(log_ratio * prior["csk"])

    return fcn


def fit_csk_point(x, pz_gev_ls, qtmdwf_gv):
    fit_res = lsf.nonlinear_fit(
        data=(pz_gev_ls, qtmdwf_gv),
        prior=broad_csk_prior(),
        fcn=make_csk_fcn(x, pz_gev_ls),
        maxit=10000,
        fitter="scipy_least_squares",
    )
    return fit_res.p["csk"]


def x_cut_indices(b):
    # 2 x bT Pz >> 1, Pz >> Lambda_QCD
    x_min = min(max(0.5 * GEV_FM / a / b / 1.16, 0.4 / 1.16), 0.45)
    idx_ini = int(np.argmax(x_ls > x_min))
    idx_fin = len(x_ls) - idx_ini
    return idx_ini, idx_fin


def load_b_qtmdwf_zdep_gv(b):
    """Load one b slice and convert all needed samples into correlated gvars."""
    bare_p0_b0_z0 = bs_ls_avg(get_fake_qtmdwf_samp(pz=0, b=0, z=0)[0]) # real part only
    bare_p0_z0 = bs_ls_avg(get_fake_qtmdwf_samp(pz=0, b=b, z=0)[0]) # normalization
    renorm_factor = bare_p0_z0 / bare_p0_b0_z0

    
    qtmdwf_re_dict = {}
    qtmdwf_im_dict = {}
    for pz in pz_fit_ls:
        bare_pz_re = [bs_ls_avg(get_fake_qtmdwf_samp(pz=pz, b=b, z=z)[0]) for z in range(zmax)]
        bare_pz_im = [bs_ls_avg(get_fake_qtmdwf_samp(pz=pz, b=b, z=z)[1]) for z in range(zmax)]
        
        renorm_pz_re = bare_pz_re / bare_pz_re[0] / renorm_factor # normalization and renormalization
        renorm_pz_im = bare_pz_im / bare_pz_im[0] / renorm_factor
    
        qtmdwf_re_dict[int(pz)] = renorm_pz_re
        qtmdwf_im_dict[int(pz)] = renorm_pz_im

    return qtmdwf_re_dict, qtmdwf_im_dict


def build_qtmdwf_xdep_for_b(b):
    qtmdwf_re_dict, qtmdwf_im_dict = load_b_qtmdwf_zdep_gv(b)
    xdep_dict = {}

    for pz in pz_fit_ls:
        lam_pos = np.arange(zmax) * a * pz_gev_dict[int(pz)] / GEV_FM
        re_pos = np.asarray(qtmdwf_re_dict[int(pz)], dtype=object).ravel()
        im_pos = np.zeros_like(re_pos)

        lam_full, re_full, im_full = complete_z_negative(lam_pos, re_pos, im_pos)

        xdep_dict[int(pz)] = np.asarray(
            [sum_ft_re_im(lam_full, re_full, im_full, x)[0] for x in x_raw]
        )

    return xdep_dict


def fit_csk_for_b(b):
    qtmdwf_xdep_gv = build_qtmdwf_xdep_for_b(b)
    pz_gev_ls = np.asarray([pz_gev_dict[int(pz)] for pz in pz_fit_ls], dtype=float)
    idx_ini, idx_fin = x_cut_indices(b)
    x_used = x_ls[idx_ini:idx_fin]

    csk_xdep_gv = []
    for x_idx, x in zip(range(idx_ini, idx_fin), x_used):
        qtmdwf_at_x = np.asarray(
            [qtmdwf_xdep_gv[int(pz)][x_idx] for pz in pz_fit_ls],
            dtype=object,
        )
        csk_xdep_gv.append(fit_csk_point(x, pz_gev_ls, qtmdwf_at_x))

    csk_xdep_gv = np.asarray(csk_xdep_gv, dtype=object)
    csk_b_gv = constant_fit(csk_xdep_gv, const_prior=gv.gvar(-1, 10))

    print(
        f"b={b:2d}: x range [{x_used[0]:.4f}, {x_used[-1]:.4f}], "
        f"n_x={len(x_used)}, csk={csk_b_gv}"
    )
    return {
        "b": b,
        "qtmdwf_xdep_gv": qtmdwf_xdep_gv,
        "x_used": x_used,
        "csk_xdep_gv": csk_xdep_gv,
        "csk_b_gv": csk_b_gv,
    }


def plot_qtmdwf_xdep(results, pz_to_plot=6):
    fig, ax = default_plot()
    for result in results:
        b = result["b"]
        qtmdwf_gv = result["qtmdwf_xdep_gv"][pz_to_plot]
        ax.fill_between(
            x_ls,
            gv.mean(qtmdwf_gv) - gv.sdev(qtmdwf_gv),
            gv.mean(qtmdwf_gv) + gv.sdev(qtmdwf_gv),
            alpha=0.3,
            label=f"b={b}",
        )

    ax.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel(r"Re qTMDWF$(x, b_T)$")
    ax.set_title(f"qTMDWF x-dependence, pz={pz_to_plot}")
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_csk_xdep(results):
    fig, ax = default_plot()
    for result in results:
        b = result["b"]
        x_used = result["x_used"]
        csk_xdep_gv = result["csk_xdep_gv"]

        ax.errorbar(
            x_used,
            gv.mean(csk_xdep_gv),
            gv.sdev(csk_xdep_gv),
            label=f"b={b}",
            **ERRORBAR_STYLE,
        )

    ax.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel(r"CS kernel from each x")
    ax.set_title(r"CS kernel x-dependence before constant fit")
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_csk_b_dependence(results):
    fig, ax = default_plot()
    b_fm = b_ls * a
    csk_gv = np.asarray([result["csk_b_gv"] for result in results], dtype=object)
    ax.errorbar(
        b_fm,
        gv.mean(csk_gv),
        gv.sdev(csk_gv),
        label="CS kernel",
        **ERRORBAR_STYLE,
    )
    ax.set_xlabel(r"$b_T$ [fm]")
    ax.set_ylabel("CS kernel")
    ax.set_title(r"CS kernel $b_T$ dependence")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    dump_dic = {"b_fm": b_fm, "csk_gv": csk_gv}
    gv.dump(dump_dic, "./data/cs_kernel.gv")


def main():
    results = [fit_csk_for_b(int(b)) for b in tqdm(b_ls, desc="fit CS kernel in bT")]
    plot_qtmdwf_xdep(results, pz_to_plot=7)
    plot_csk_xdep(results)
    plot_csk_b_dependence(results)
    return results


if __name__ == "__main__":
    main()


# %%
