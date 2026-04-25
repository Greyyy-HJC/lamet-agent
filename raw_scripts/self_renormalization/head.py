import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Figure and plotting parameters
fig_width = 6.75  # in inches, 2x as wide as APS column
gr = 1.618034333  # golden ratio
fig_size = (fig_width, fig_width / gr)
plt_axes = [0.1, 0.12, 0.85, 0.8]
errorb = {"markersize": 5, "mfc": "none", "linestyle": "none", "capsize": 3, "elinewidth": 1}
errorl = {"markersize": 5, "mfc": "none", "capsize": 3, "elinewidth": 1}
fs_p = {"fontsize": 13}
ls_p = {"labelsize": 13}

# Physical constants
gev_fm = 0.1973269631  # 1 = 0.197 GeV . fm
delta_sys = 0  # 0.002, used to add sys error for pdf data after interpolation

# Lattice parameters
a_milc_ls = [0.1213, 0.0882, 0.0574, 0.0425, 0.0318]
a_rbc_ls = [0.11, 0.0828, 0.0626]

a_gluon_ls = [0.0888, 0.1207, 0.151]

# QCD parameters
lqcd = 0.1  # Lambda_QCD
k = 3.320
d_pdf = -0.08183  # for pdf
d_da = 0.19  # 0.1 for a^2 order, 0.19 for a order, from Yushan
m0_da = gv.gvar(-0.094, 0.024)  # from Yushan
mu = 2  # GeV, for renormalization

cf = 4/3
nf = 3
b0 = 11 - 2/3 * nf
ca = 3

# z lists
z_ls = np.arange(0.06, 1.26, 0.06)  # read bare pdf then interpolated into z_ls
z_ls_extend = np.arange(0.06, 1.56, 0.06)  # extend f1 and zR

def interp_1d(x_in, y_in, x_out, method="linear"):
    """Interpolation function"""
    f = interpolate.interp1d(x_in, y_in, kind=method)
    y_out = f(x_out)
    return y_out

def ZMS_pdf(z):
    """MS-bar renormalization factor for PDF"""
    # z in fm
    ans = 1 + (2 * np.pi / (b0 * np.log(mu / lqcd)) * cf / (2 * np.pi)) * (
        3/2 * np.log(mu**2 * (z / gev_fm)**2 * np.exp(2 * np.euler_gamma) / 4) + 5/2
    )
    return ans

def ZMS_pdf_gluon(z):
    """MS-bar renormalization factor for gluon PDF from Fei"""
    
    ans = 1 + (2 * np.pi / (b0 * np.log(mu / lqcd)) * ca / (4 * np.pi)) * (
        5/3 * np.log(mu**2 * (z / gev_fm)**2 * np.exp(2 * np.euler_gamma) / 4) + 3
    )
    return ans