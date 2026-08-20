<!-- lamet-agent formula cache; kernel=GI_gz_quark_PDF_hybrid_NLO; arxiv=2604.00143; equations=Eqs. (C6)-(C8); digest=b0d0fb641f3c6199; paper_used=true -->
$$C_{\mathrm{hybrid}}^{g_z}(\xi,L,y,z_sP_z) = \left[\, \frac{1+\xi^2}{1-\xi}\left(L+\ln\frac{4\xi(1-\xi)}{1}\right) - 1 + 1 \,\right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2}\left[\, -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_sP_z\big)}{\pi(1-\xi)} \,\right]^{(-\infty,\infty)}_{+(1)}$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription is defined as in the paper:

$$\int_{-\infty}^{\infty} d\xi\, [g(\xi)]^{D}_{+(x_0)}\, f(\xi) = \int_D d\xi\, g(\xi)\big(f(\xi)-f(x_0)\big)$$

with $D=(-\infty,\infty)$ and $x_0=1$. The first bracket combines the ratio-scheme coefficient of Eq. (C7) (with the $\gamma^z$ shift $2(1-\xi)$ included) and the second is the hybrid correction of Eq. (C8) with strength $3/2$. The LO $\delta(1-\xi)$ term is implicit in the discretization's identity matrix.

#### Consistency check

The code reproduces Eqs. (C6)–(C8) of arXiv:2604.00143 term by term. The regular coefficient matches Eq. (C7) exactly: the splitting function $(1+\xi^2)/(1-\xi)$, the log $\ln(4\xi(1-\xi))$ combined with $L$, the constant $-1$, and the $+1$ from the $\gamma^z$ shift. The hybrid correction of Eq. (C8) is reproduced with the correct prefactor $3/2$, the $-1/|1-\xi|$ term, and the sine-integral term $\mathrm{Si}((1-\xi)|y|z_sP_z)/(\pi(1-\xi))$. The plus-prescription domain $(-\infty,\infty)$ and subtraction point $x_0=1$ match the paper's notation. The only discrepancy is that the code omits the $\delta C_M$ (leading-renormalon/mass) term and the NNLO piece of Eq. (C6), which the code explicitly states are not implemented (NLO only). No other discrepancies found.
