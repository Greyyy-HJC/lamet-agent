<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=3b2240b4f7336c15; paper_used=true -->
$$C_{\rm RGR}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left(-\frac{1}{|1-\xi|}+\frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right)\right]_{+(1)}^{[-\infty,\infty]}$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the ratio-scheme kernel is

$$C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right)= \frac{\alpha_{s} C_{F}}{2 \pi} \begin{cases}\left(\frac{1+\xi^{2}}{1-\xi} \ln \frac{\xi}{\xi-1}+1-\frac{3}{2(1-\xi)}\right)_{+(1)}^{[1, \infty]} & \xi>1 \\ \left(\frac{1+\xi^{2}}{1-\xi}\left[-\ln \frac{\mu^{2}}{4 x^{2} P_{z}^{2}}+\ln (\frac{1-\xi}{\xi})-1\right]+1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[0,1]} & 0<\xi<1 \\ \left(-\frac{1+\xi^{2}}{1-\xi} \ln \frac{-\xi}{1-\xi}-1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[-\infty, 0]} & \xi<0 \end{cases}$$

The plus-prescription is defined as in the paper: for a function $g(\xi)$ with a singularity at $\xi=1$, $[g(\xi)]_{+(1)}^{D}$ is the distribution such that $\int_D d\xi\,[g(\xi)]_{+(1)}^{D}\,\phi(\xi) = \int_D d\xi\,g(\xi)\,[\phi(\xi)-\phi(1)]$ for any test function $\phi$, with the domain $D$ indicated by the superscript. The $\delta(1-\xi)$ term is implicit in the plus-prescription and is not written separately.

The RGR kernel is not a fixed-order coefficient. Each row $x$ is constructed by evaluating the fixed-order kernel at the row's own scale $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, $\kappa=1$ for the central value), then evolving to the final scale $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet DGLAP splitting function $P[w,\alpha_s(\mu)]$:

$$C^{-1}_{\rm RGR}\left(\frac{x}{y},\frac{\mu}{|x|P_z}\right) = \mathcal{P}\exp\left[\int_{\mu_0(x)}^{\mu} \frac{d\mu'}{\mu'}\,P\left[\frac{x}{y},\alpha_s(\mu')\right]\right] C^{-1}\left(\frac{x}{y},\frac{\mu_0(x)}{|x|P_z}\right)$$

Rows with $\mu_0(x)<\mu_{\rm min}$ (the perturbative cutoff, corresponding to the paper's $x_{\rm min}$) are set to zero. The splitting function used is the full helicity (q+q̄)/2 channel, which is the valence kernel plus an additional $n_f$ structure.

The scheme-specific correction is the Wilson-line term $\frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left(-\frac{1}{|1-\xi|}+\frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right)$, which replaces the $\overline{\rm MS}$ counterterm and encodes the hybrid-scheme renormalization with Wilson-line length $z_s$.

#### Consistency check

The code reproduces the paper's Eq. (matchingRGI) and the hybrid-scheme NLO kernel of App. A (Eqs. (2.19)–(2.20) of the cited paper) term by term: the ratio-scheme coefficient $C^{\rm ratio(1)}$ matches exactly, including the $\ln(\mu^2/(4x^2P_z^2))$ argument and the $\ln(\xi/(\xi-1))$, $\ln((1-\xi)/\xi)$, and $\ln(-\xi/(1-\xi))$ branch logs; the plus-prescription domains $[1,\infty)$, $[0,1]$, $[-\infty,0]$ and the overall $[-\infty,\infty]$ with subscript $+(1)$ are reproduced verbatim; the Wilson-line Si correction with its $|y|z_sP_z$ argument matches; and the RGR construction via per-row scale $\mu_0=2\kappa xP_z$ and DGLAP evolution with the two-loop non-singlet kernel follows the paper's App. 'A Method Solving RG Equation'. No discrepancies were found between the code and the paper for this coefficient.
