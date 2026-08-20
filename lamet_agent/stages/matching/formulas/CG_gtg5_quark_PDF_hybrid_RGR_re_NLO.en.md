<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=a6b07247ed7af7b3; paper_used=true -->
$$C_{\rm RGR}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[ C^{\rm ratio(1)}\left(\xi, \frac{\mu}{|x| P_{z}}\right) + \frac{\alpha_s C_{F}}{2 \pi}\frac{3}{2} \left[-\frac{1}{|1-\xi|}+\frac{2 {\rm Si}[(1-\xi)|y| z_s P_z]}{\pi (1-\xi)} \right]_{+(1)}^{[-\infty,\infty]} \right]_{+(1)}^{[-\infty,\infty]},$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$. The ratio-scheme piece is

$$C^{\rm ratio(1)}\left(\xi, \frac{\mu}{|x| P_{z}}\right)= \frac{\alpha_{s} C_{F}}{2 \pi} \begin{cases}\left(\frac{1+\xi^{2}}{1-\xi} \ln \frac{\xi}{\xi-1}+1-\frac{3}{2(1-\xi)}\right)_{+(1)}^{[1, \infty]} & \xi>1 \\ \left(\frac{1+\xi^{2}}{1-\xi}\left[-\ln \frac{\mu^{2}}{4 x^{2} P_{z}^{2}}+\ln (\frac{1-\xi}{\xi})-1\right]+1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[0,1]} & 0<\xi<1 \\ \left(-\frac{1+\xi^{2}}{1-\xi} \ln \frac{-\xi}{1-\xi}-1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[-\infty, 0]} & \xi<0 \end{cases}$$

The plus-prescription is defined as in the paper: for a function $g(\xi)$ regular except at $\xi=1$,  
$$\int_{-\infty}^{\infty} d\xi\, [g(\xi)]_{+(1)}^{D} \, f(\xi) = \int_D d\xi\, g(\xi)\,[f(\xi)-f(1)],$$  
with the domain $D$ indicated by the superscript. The hybrid correction replaces the $\overline{\rm MS}$ $0.5/|1-\xi|$ term with the Wilson-line sine-integral term shown, where $z_s P_z$ is the dimensionless hybrid-scheme parameter.

The RGR kernel is not a fixed-order coefficient: each row $x$ is evaluated at the intrinsic scale $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, $c'$ in the paper), then evolved to the common scale $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet DGLAP kernel for the full helicity channel $(q+\bar q)/2$. Rows with $\mu_0(x)<\mu_{\min}$ (the paper's $x_{\min}$) are set to zero, reflecting the breakdown of perturbation theory at small $x$.

#### Consistency check

The code reproduces the paper's Eq. (2.19)–(2.20) for the hybrid NLO kernel: the ratio-scheme coefficient $C^{\rm ratio(1)}$ matches the paper's Eq. (2.16) term-by-term (the splitting-function log, the $\ln(\mu^2/(4x^2P_z^2))$ term, the $\ln(\xi/(1-\xi))$ and $\ln(-\xi/(1-\xi))$ branches, the $+1$/$-1$ constants, and the $-3/(2|1-\xi|)$ term), and the hybrid correction with the sine-integral and the $3/2$ prefactor matches the paper's Eq. (2.20). The plus-prescription domains and the $+(1)$ subscript are copied verbatim. The RGR construction follows the paper's App. A method (Eq. matchingRGI): the per-row scale $\mu_0=2\kappa xP_z$, the DGLAP evolution operator, and the $x_{\min}$ cutoff are all implemented as described. The only discrepancy is notational: the code's $\kappa$ is the paper's $c'$, and the code's $\mu_{\min}$ is the paper's $x_{\min}$ expressed in momentum units; no numerical or structural disagreement was found.
