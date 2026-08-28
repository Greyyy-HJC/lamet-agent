<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gz_hybrid_nlo; arxiv=2604.00143; equations=Eqs. (C6)-(C8); digest=b16784b124346018; paper_used=true -->
$$C_{\rm hybrid}^{g_z}(\xi,y,\mu,P_z,z_s) = C_{\rm ratio}^{g_z}(\xi,L) + \Delta C_{\rm hybrid}(\xi,y,z_sP_z)$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$. The ratio-scheme coefficient is

$$C_{\rm ratio}^{g_z}(\xi,L) = \frac{1+\xi^2}{1-\xi}\Bigl[L+\ln(4\xi(1-\xi))-1\Bigr]_+^{(0,1)} + 1 + \frac{3}{2}\frac{1}{|1-\xi|} + \theta(\xi>1)\Bigl[\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+1\Bigr] + \theta(\xi<0)\Bigl[-\frac{1+\xi^2}{1-\xi}\ln\frac{|\xi|}{|\xi-1|}-1\Bigr] + 2(1-\xi)\theta(0<\xi<1)$$

where the plus-prescription is defined as in the paper:

$$\int_0^1 d\xi\,[g(\xi)]^{(0,1)}_{+(1)}\,\varphi(\xi) = \int_0^1 d\xi\,g(\xi)\bigl(\varphi(\xi)-\varphi(1)\bigr)$$

The hybrid correction is

$$\Delta C_{\rm hybrid}(\xi,y,z_sP_z) = \frac{3}{2}\Bigl[-\frac{1}{|1-\xi|} + \frac{2\,{\rm Si}\bigl((1-\xi)|y|z_sP_z\bigr)}{\pi(1-\xi)}\Bigr]$$

where ${\rm Si}(z)=\int_0^z dt\,\sin t/t$ is the sine integral. The full coefficient is

$$C_{\rm hybrid}^{g_z}(\xi,y,\mu,P_z,z_s) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi}\Bigl[C_{\rm ratio}^{g_z}(\xi,L) + \Delta C_{\rm hybrid}(\xi,y,z_sP_z)\Bigr]$$

with the $\delta(1-\xi)$ term implicit in the plus-prescription.

#### Consistency check

The code reproduces Eqs. (C6)–(C8) of arXiv:2604.00143 term by term. The regular coefficient matches: the splitting function $(1+\xi^2)/(1-\xi)$, the log $L+\ln(4\xi(1-\xi))$ inside $[0,1]$, the $\ln(\xi/(\xi-1))$ branches outside, the constant $+1$, and the $2(1-\xi)$ shift for $\gamma^z$ vs $\gamma^t$ are all present with correct signs and arguments. The plus-prescription is implemented as the paper's $[g]^{(0,1)}_{+(1)}$ with the subtraction at $\xi=1$ over $[0,1]$. The hybrid correction matches Eq. (C8): the prefactor $3/2$, the $-1/|1-\xi|$ term, and the sine-integral term $2\,{\rm Si}((1-\xi)|y|z_sP_z)/(\pi(1-\xi))$ are all correct. The code omits the $\delta C_M$ (leading-renormalon/mass) term and the NNLO piece of Eq. (C6), as documented in its docstring — this is a deliberate NLO-only truncation, not a discrepancy. No other disagreements found.

