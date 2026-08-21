<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=6a209a68a9a1fee6; paper_used=true -->
$$C_{\rm RGR}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C_{\rm ratio}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left(-\frac{1}{|1-\xi|} + \frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right)\right]^{(-\infty,\infty)}_{+(1)}$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the ratio-scheme coefficient (Eq. 2.16 of the paper, with the plus prescription at $\xi=1$ over the domain $[0,1]$):

$$C_{\rm ratio}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\left[\left(\frac{1+\xi^2}{1-\xi}\left[-\ln\frac{\mu^2}{4x^2P_z^2} + \ln\frac{1-\xi}{\xi} - 1\right] + 1 + \frac{3}{2(1-\xi)}\right)^{[0,1]}_{+(1)}\right]$$

The hybrid correction replaces the $\overline{\rm MS}$ $0.5/|1-\xi|$ term with the sine-integral form, as in Eq. (2.19)–(2.20) of the paper. The plus prescription is defined by $\int_D d\xi\,[g(\xi)]^D_{+(x_0)} = 0$ for any test function, with the subtraction at $x_0=1$.

The RGR kernel is not fixed order: each row $x$ is built from the fixed-order matrix evaluated at $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, $c'$ in the paper), then evolved to $\mu$ via a path-ordered matrix exponential of the two-loop non-singlet DGLAP kernel (the valence combination, $P_{qq}^{(2)} + 16C_F(C_F-C_A/2)$ structure). Rows with $\mu_0(x)<\mu_{\min}$ are set to zero, implementing the paper's $x_{\min}$ cutoff.

#### Consistency check

The code reproduces the paper's Eq. (2.16) for the ratio-scheme coefficient exactly: the splitting function $(1+\xi^2)/(1-\xi)$, the log $\ln(\mu^2/(4x^2P_z^2))$, the $\ln((1-\xi)/\xi)$ term, the $+1$ constant, and the $3/(2(1-\xi))$ term all match. The hybrid correction in Eq. (2.19)–(2.20) is also reproduced: the $-\frac{1}{|1-\xi|}$ plus the sine-integral term with argument $(1-\xi)|y|z_sP_z$ match the code's `C_hybrid`. The plus prescription is implemented as the column-sum subtraction, matching the paper's $[\,\cdot\,]^{D}_{+(1)}$ convention. The RGR evolution follows App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236: the per-row scale $\mu_0=2\kappa xP_z$, the DGLAP evolution operator, and the $x_{\min}$ cutoff are all present. No discrepancies found between code and paper for this coefficient.
