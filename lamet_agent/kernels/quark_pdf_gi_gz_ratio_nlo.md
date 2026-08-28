<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gz_ratio_nlo; arxiv=2604.00143; equations=Eq. (C7); digest=68124b57c9a523fe; paper_used=true -->
$$C_{g_z}^{\mathrm{ratio}}(\xi, L) = \left[ \frac{1+\xi^2}{1-\xi} \left( L + \ln\frac{4\xi(1-\xi)}{1} - 1 \right) + 1 \right]^{D}_{+(1)} + \frac{3}{2|1-\xi|} + 2(1-\xi) \quad (0<\xi<1),$$

$$C_{g_z}^{\mathrm{ratio}}(\xi, L) = \left[ \operatorname{sgn}(\xi) \left( \frac{1+\xi^2}{1-\xi} \ln\frac{|\xi|}{|\xi-1|} + 1 \right) \right]^{D}_{+(1)} + \frac{3}{2|1-\xi|} \quad (\xi<0 \text{ or } \xi>1),$$

where $L = \ln(4y^2P_z^2/\mu^2)$, $\xi = x/y$, and the plus-prescription is defined as in the paper:

$$\int_0^1 d\xi\, [g(\xi)]^{D}_{+(1)} \varphi(\xi) = \int_0^1 d\xi\, g(\xi) \big(\varphi(\xi) - \varphi(1)\big),$$

with the domain $D$ being $[0,1]$ for the first bracket and $(-\infty,\infty)$ for the second. The scheme-specific correction is the $2(1-\xi)$ term on $0<\xi<1$, which distinguishes the $\gamma^z$ from the $\gamma^t$ coefficient. There is no explicit $\delta(1-\xi)$ term; the plus-prescription restores the singularity at $\xi=1$.

#### Consistency check

The code implements exactly the coefficient above. The regular coefficient matches: for $0<\xi<1$, the code gives $S(L - \ln 4 + \ln(4\xi(1-\xi)) - 1) + 1$ with $S=(1+\xi^2)/(1-\xi)$, which equals the paper's $S(\ln(y^2P_z^2/\mu^2) + \ln(4\xi(1-\xi)) - 1) + 1$ after the code's $\ln 4$ removal. The $3/(2|1-\xi|)$ tail and the $2(1-\xi)$ correction are present. The plus-prescription is restored by the column-sum in `build_matching_matrix`, matching the paper's $[g]^{D}_{+(1)}$ with domain $[0,1]$ for the first branch and $(-\infty,\infty)$ for the second. No discrepancies found.

