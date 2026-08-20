<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=67655fb2fa6709cf; paper_used=true -->
$$C_{\rm RGR}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{(1)}_{\rm ratio}\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left(-\frac{1}{|1-\xi|} + \frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right)\right]^{[-\infty,\infty]}_{+(1)} + \delta(1-\xi),$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the ratio-scheme piece (Eq. 2.16 of the paper, with the plus prescription over $[0,1]$ for $0<\xi<1$):

$$C^{(1)}_{\rm ratio}\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\begin{cases}
\left(\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+1-\frac{3}{2(1-\xi)}\right)^{[1,\infty]}_{+(1)} & \xi>1,\\[4pt]
\left(\frac{1+\xi^2}{1-\xi}\left[-L+\ln\left(\frac{1-\xi}{\xi}\right)-1\right]+1+\frac{3}{2(1-\xi)}\right)^{[0,1]}_{+(1)} & 0<\xi<1,\\[4pt]
\left(-\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-1+\frac{3}{2(1-\xi)}\right)^{[-\infty,0]}_{+(1)} & \xi<0,
\end{cases}$$

where the plus prescription is defined as in the paper: for a function $g(\xi)$ with a singularity at $\xi=x_0$, $[g(\xi)]^{D}_{+(x_0)}$ satisfies $\int_D d\xi\,[g(\xi)]^{D}_{+(x_0)}\,h(\xi) = \int_D d\xi\,g(\xi)\,[h(\xi)-h(x_0)]$ for any smooth test function $h$, and the domain $D$ is indicated by the superscript. The hybrid correction replaces the $\overline{\rm MS}$ $0.5/|1-\xi|$ term with the sine-integral expression shown, evaluated at the per-$y$ Wilson-line scale $z_s|y|P_z$.

The resummation is implemented row-by-row: for each light-cone $x$, the fixed-order kernel is evaluated at the intrinsic scale $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, scanned over $0.8$–$1.2$ in the paper), and then evolved to the final $\mu$ via the path-ordered matrix exponential of the two-loop (NLL) non-singlet DGLAP kernel for the valence combination $q-\bar{q}$ (the code uses the unpolarized splitting function plus the $C$-parity term $16C_F(C_F-C_A/2)$). Rows with $\mu_0(x)<\mu_{\rm min}$ (the perturbative cutoff, corresponding to the paper's $x_{\rm min}$) are set to zero, reflecting the breakdown of perturbation theory at small $x$.

#### Consistency check

The code reproduces the paper's Eq. (2.19)–(2.20) for the hybrid NLO kernel exactly: the ratio-scheme piece matches Eq. (2.16) term-by-term (including the $\ln(\xi/(\xi-1))$ for $\xi>1$, the $L$ and $\ln((1-\xi)/\xi)$ for $0<\xi<1$, and the $\ln(-\xi/(1-\xi))$ for $\xi<0$), the plus prescription is implemented via the column-sum subtraction with the domain $[-\infty,\infty]$ and subtraction point $+(1)$, and the hybrid correction with the sine integral and the $3/2$ prefactor matches the paper's expression. The resummation follows App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236: the per-row scale $\mu_0=2\kappa xP_z$, the DGLAP evolution with the two-loop valence splitting function, and the cutoff at small $\mu_0$ are all as described in the paper. No discrepancies were found between the code and the paper for this coefficient.
