<!-- lamet-agent formula cache; kernel=GI_gzg5_quark_PDF_ratio_NLO; arxiv=2604.00143; equations=Eq. (C7); digest=d8a22b203852203b; paper_used=true -->
$$C_{\mathrm{GI}}^{g_zg_5}(\xi,L)=\left[\frac{1+\xi^2}{1-\xi}\left(L+\ln\frac{4\xi(1-\xi)}{1}\right)-1\right]^{D}_{+(1)}+\left[\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+1\right]^{D}_{+(1)}+\frac{3}{2|1-\xi|}+2(1-\xi)\quad(0<\xi<1),$$

with $L=\ln(4y^2P_z^2/\mu^2)$, $\xi=x/y$, and the plus-prescription defined as in the paper:
$$\int_0^1 d\xi\,[g(\xi)]^{D}_{+(x_0)}\,\varphi(\xi)=\int_0^1 d\xi\,g(\xi)\big(\varphi(\xi)-\varphi(x_0)\big),$$
with $x_0=1$ and $D=[0,1]$ for the first bracket, $D=(-\infty,\infty)$ for the second. The scheme-specific correction is the $+2(1-\xi)$ term, which distinguishes the $\gamma^z$ from the $\gamma^t$ coefficient.

#### Consistency check

The code implements exactly the coefficient above: the splitting function $(1+\xi^2)/(1-\xi)$, the log $L+\ln(4\xi(1-\xi))$ (with the $\ln 4$ removed from $L$ as in the code), the $+1$ constant, the $3/(2|1-\xi|)$ tail, and the $+2(1-\xi)$ shift. The plus-prescription is restored by the column-sum in `build_matching_matrix`, matching the paper's $[\,\cdot\,]^{D}_{+(1)}$ with the correct domain split. No discrepancies found.

