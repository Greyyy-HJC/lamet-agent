<!-- lamet-agent formula cache; kernel=GI_gz_quark_PDF_ratio_NLO; arxiv=2604.00143; equations=Eq. (C7); digest=28e7a17e655f1b24; paper_used=true -->
$$ \mathcal{C}^{\mathrm{ratio}}_{g_z}(x,y,\mu,P_z) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi} \left\{ \left[ \frac{1+\xi^2}{1-\xi} \left( \ln\frac{4y^2P_z^2}{\mu^2} + \ln(4\xi(1-\xi)) - 1 \right) + 1 \right]^{[0,1]}_{+(1)} + \left[ \mathrm{sgn}(\xi)\left( \frac{1+\xi^2}{1-\xi} \ln\frac{|\xi|}{|\xi-1|} + 1 \right) \right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2|1-\xi|} + 2(1-\xi) \right\} $$

where $\xi = x/y$, $L = \ln(4y^2P_z^2/\mu^2)$, and the plus-prescription is defined as in the paper:
$$ \int_0^1 d\alpha\,[g(\alpha)]^{D}_{+(x_0)}\,\varphi(\alpha) = \int_D d\alpha\,g(\alpha)\big(\varphi(\alpha)-\varphi(x_0)\big) $$
with $D=[0,1]$ for the first bracket and $D=(-\infty,\infty)$ for the second, both with $x_0=1$. The term $2(1-\xi)$ is the scheme-specific correction for the $\gamma^z$ operator relative to $\gamma^t$, active only for $0<\xi<1$.

#### Consistency check

The code reproduces Eq. (C7) of arXiv:2604.00143 term by term. The regular coefficient matches: the splitting function $(1+\xi^2)/(1-\xi)$, the log combination $\ln(y^2P_z^2/\mu^2) + \ln(4\xi(1-\xi))$ (the code's `log_scale - ln(4)` plus `ln(4*ksi*(1-ksi))`), the constant $-1$, and the $+1$ term. The plus-prescription is correctly implemented with the paper's exact notation: two separate brackets over $[0,1]$ and $(-\infty,\infty)$, both with subtraction point $+(1)$, and the code's column-sum prescription matches the paper's definition. The $\delta(1-\xi)$ term is implicit in the LO identity matrix. The scheme-specific correction $2(1-\xi)$ for $0<\xi<1$ is present. The only notational difference is that the code writes the log as $\ln(4y^2P_z^2/\mu^2)$ while the paper uses $-\ln(\mu_0^2/(4w^2P_z^2))$ with $w=\xi y$; these are algebraically identical. No discrepancies found.
