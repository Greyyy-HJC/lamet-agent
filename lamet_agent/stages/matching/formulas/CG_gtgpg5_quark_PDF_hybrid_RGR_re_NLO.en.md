<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=b02e76344af257a1; paper_used=true -->
$$C_{\rm RGR}^{\,(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right)\right]_{+(1)}^{[-\infty,\infty]} + \frac{\alpha_s C_F}{2\pi}\,\frac{3}{2}\left[-\frac{1}{|1-\xi|}+\frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right]_{+(1)}^{[-\infty,\infty]},$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the ratio-scheme kernel (Eq. 2.18 of the paper, with the transversity splitting $2\xi/(1-\xi)$ and no $+\xi-1$ or $+{\rm sgn}(\xi)$ terms):

$$C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\begin{cases}
\left(\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1}-\frac{1}{|1-\xi|}\right)_{+(1)}^{[1,\infty]} & \xi>1 \\[4pt]
\left(\frac{2\xi}{1-\xi}\left[-\ln\frac{\mu^2}{4x^2P_z^2}+\ln\frac{1-\xi}{\xi}\right]-\frac{1}{|1-\xi|}\right)_{+(1)}^{[0,1]} & 0<\xi<1 \\[4pt]
\left(-\frac{2\xi}{1-\xi}\ln\frac{-\xi}{1-\xi}-\frac{1}{|1-\xi|}\right)_{+(1)}^{[-\infty,0]} & \xi<0
\end{cases}$$

plus the shared arctan/arctanh piece (identical in Eq. 2.16 and 2.18):

$$\frac{3\xi-1}{\xi-1}\cdot\frac{1}{\sqrt{|1-2\xi|}}\begin{cases}
\arctan\frac{\sqrt{1-2\xi}}{|\xi|} & \xi<\frac12 \\[4pt]
\arctanh\frac{\sqrt{2\xi-1}}{|\xi|} & \xi>\frac12
\end{cases}$$

(analytic at $\xi=1/2$, where it equals $(3\xi-1)/(\xi-1)/|\xi|$). The plus prescription is defined as in the paper: $[g(\xi)]_{+(x_0)}^{D}$ makes each $y$-column integrate to zero over the domain $D$, with the subtraction at $x_0=1$; the code implements this by summing the regular density over the full quasi grid and subtracting that total from the diagonal.

The RGR resummation is not a fixed-order coefficient: each row $x$ is built from the fixed-order matrix evaluated at that row's own scale $\mu_0(x)=2\kappa xP^z$ (with $\kappa$ the scale-variation knob, scanned over $0.8$–$1.2$), then DGLAP-evolved to $\mu$ by a path-ordered matrix exponential of the two-loop (NLL) non-singlet transversity splitting function $P_{\rm trans}^{(1)}(\nu)=4\nu/(1-\nu)\,[\dots]$ (the code's `_p_nlo_transversity`). Rows whose $\mu_0(x)$ falls below the perturbative cutoff $\mu_{\rm min}=0.6$ GeV are set to zero—this is the paper's $x_{\rm min}$.

#### Consistency check

The code reproduces App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236 for the transversity case, with the following term-by-term agreement: the regular coefficient matches Eq. (2.18) exactly (the transversity splitting $2\xi/(1-\xi)$, the $-1/|1-\xi|$ tail, and the shared arctan/arctanh branch); the logarithms have the correct arguments ($\ln(\mu^2/4x^2P_z^2)$ in the $0<\xi<1$ region, $\ln[\xi/(\xi-1)]$ and $\ln[-\xi/(1-\xi)]$ in the other regions); the plus prescription is applied with the paper's exact notation $[\,\cdot\,]_{+(1)}^{D}$ over the three domains $[1,\infty]$, $[0,1]$, $[-\infty,0]$, plus the overall $[-\infty,\infty]$ bracket for the hybrid correction; there is no $\delta(1-\xi)$ term (the code's LO delta is the identity stencil, not a separate delta in the NLO coefficient); and the scheme-specific correction is the hybrid term $\frac{\alpha_s C_F}{2\pi}\frac{3}{2}[-\frac{1}{|1-\xi|}+\frac{2{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}]_{+(1)}^{[-\infty,\infty]}$, which matches Eq. (2.21) of the paper (the paper states $\delta C_{\rm hyb}=0$ for transversity at NLO, so the hybrid kernel equals the ratio kernel—the code confirms this by setting `zspz` to `None` and returning the ratio kernel directly). The RGR evolution operator and the per-row scale $\mu_0=2\kappa xP^z$ follow the paper's Eq. (matchingRGI) and the method of App. 'A Method Solving RG Equation'. No discrepancies were found.
