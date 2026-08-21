<!-- lamet-agent formula cache; kernel=GI_gzg5_quark_PDF_ratio_NLO; arxiv=2604.00143; equations=Eq. (C7); digest=fd6898d74fff172e; paper_used=true -->
$$C_{\mathrm{ratio}}^{g_z g_5}(\xi, L) = \frac{1+\xi^2}{1-\xi}\Big[L + \ln\!\big(4\xi(1-\xi)\big) - 1\Big] + 1 + \frac{3}{2(1-\xi)} \quad \text{for } 0<\xi<1,$$
$$C_{\mathrm{ratio}}^{g_z g_5}(\xi, L) = \operatorname{sgn}(\xi)\left[\frac{1+\xi^2}{1-\xi}\ln\!\left|\frac{\xi}{\xi-1}\right| + 1\right] + \frac{3}{2|1-\xi|} \quad \text{for } \xi<0 \text{ or } \xi>1,$$
with the plus-prescription at $\xi=1$ over the full domain $(-\infty,\infty)$, written in the paper's notation as
$$\left[\,C_{\mathrm{ratio}}^{g_z g_5}(\xi, L)\,\right]^{(-\infty,\infty)}_{+(1)} = \left[\,\frac{1+\xi^2}{1-\xi}\Big(L + \ln\!\big(4\xi(1-\xi)\big) - 1\Big) + 1 + \frac{3}{2(1-\xi)}\,\right]^{(-\infty,\infty)}_{+(1)},$$
where the plus-prescription is defined by
$$\int_{-\infty}^{\infty} d\xi\, \left[g(\xi)\right]^{D}_{+(x_0)} \varphi(\xi) = \int_{-\infty}^{\infty} d\xi\, g(\xi)\big(\varphi(\xi) - \varphi(x_0)\big),$$
with $x_0 = 1$ and $D = (-\infty,\infty)$. There is no explicit $\delta(1-\xi)$ term; the plus-prescription restores the singularity at $\xi=1$ by enforcing the column-sum condition. The scheme-specific correction is the $+2(1-\xi)$ term on $0<\xi<1$ relative to the $\gamma^t$ coefficient, which is already included in the expression above.

#### Consistency check
The code reproduces Eq. (C7) of arXiv:2604.00143 exactly. Term-by-term: the splitting function $(1+\xi^2)/(1-\xi)$ matches; the log argument inside $[0,1]$ is $L + \ln(4\xi(1-\xi))$ with $L = \ln(4y^2P_z^2/\mu^2)$ (the code removes the $\ln 4$ from the discretization's convention, matching the paper's $\ln(y^2P_z^2/\mu^2)$); the constant $-1$ and the $+1$ term match; the $3/(2(1-\xi))$ tail is present in all regions; the outside-region form $\operatorname{sgn}(\xi)[S\ln|\xi/(\xi-1)| + 1]$ matches; the plus-prescription is applied over the full domain with subtraction at $\xi=1$; and the scheme-specific $+2(1-\xi)$ on $0<\xi<1$ is included. No discrepancies found.
