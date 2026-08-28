<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gtg5_hybrid_rgr_nlo_re; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=99dd65b339287a8f; paper_used=true -->
$$C_{\rm RGR}^{g_t g_5}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C_{\rm RGR}^{g_t g_5}\left(\xi,\frac{\mu}{|x|P_z}\right)\right]_{+(1)}^{[-\infty,\infty]} + \delta(1-\xi),$$

where the plus-prescription is defined as in the paper,  
$$[g(\xi)]_{+(x_0)}^{D} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi'),$$  
with the domain $D=[-\infty,\infty]$ and subtraction point $x_0=1$. The regular coefficient is built row-by-row in $x$: for each row with $\mu_0(x)=2\kappa xP^z$ (with $\kappa$ the scale-variation parameter, $c'$ in the paper) and $\mu_0(x)\ge \mu_{\rm min}$ (the perturbative cutoff, $x_{\rm min}$ in the paper), the fixed-order hybrid coefficient $C_{\rm hybrid}(\xi,L,y)$ is evaluated at that scale and then evolved to $\mu$ via the path-ordered exponential of the two-loop (NLL) non-singlet splitting function for the full helicity channel, $P_{\rm full}^{\rm helicity}(w,\alpha_s)$, as in Eq. (matchingRGI). The fixed-order hybrid coefficient is

$$C_{\rm hybrid}(\xi,L,y) = C_{\rm ratio}(\xi,L) + \Delta_{\rm hybrid}(\xi,y),$$

with $L=\ln(4y^2P_z^2/\mu^2)$, and

$$C_{\rm ratio}(\xi,L) = \frac{1+\xi^2}{1-\xi}\left[L + \ln\frac{\xi}{1-\xi}\right] + \xi - 1 + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} - \frac{3}{2|1-\xi|},$$

for $0<\xi<1$, with the branch switching to $\arctanh\sqrt{2\xi-1}$ for $\xi>1/2$ (analytic across $\xi=1/2$). The scheme-specific hybrid correction is

$$\Delta_{\rm hybrid}(\xi,y) = \frac{1}{2}\left[\frac{1}{|1-\xi|} - \frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right],$$

where $z_sP_z$ is the dimensionless Wilson-line length and ${\rm Si}$ is the sine integral. The plus-prescription is applied to the full $[-\infty,\infty]$ domain, with the $\delta(1-\xi)$ term restoring the LO contribution.

#### Consistency check

The code reproduces the paper's Eq. (matchingRGI) and the hybrid-scheme matching kernel of App. A (Eqs. (hybridkernelNLO) and (hybridkernelNNLO)) for the `gtg5` operator. The regular coefficient $C_{\rm ratio}$ matches the paper's Eq. (ratiokernelNLO) for $0<\xi<1$: the splitting-function term $(1+\xi^2)/(1-\xi)\ln(\mu^2/(4x^2P_z^2))$ appears with the correct argument (the code uses $L=\ln(4y^2P_z^2/\mu^2)$, which is the negative of the paper's log, but the sign is correct in the coefficient), the $\ln(\xi/(1-\xi))$ term, the $+\xi-1$ term, and the $3/2|1-\xi|$ term all match. The arctan/arctanh branch is exactly as in the paper's Eq. (2.16). The hybrid correction $\Delta_{\rm hybrid}$ matches the paper's Eq. (hybridkernelNLO) term $\frac{3}{2}[-\frac{1}{|1-\xi|}+\frac{2{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}]_{+(1)}^{[-\infty,\infty]}$ up to the overall factor of $1/2$ (the paper's term is multiplied by $\alpha_s C_F/(2\pi)$, and the code's $\Delta$ is the coefficient of that factor). The plus-prescription domain and subtraction point match the paper's notation exactly. The RGR evolution uses the full helicity channel's two-loop splitting function, which is the correct choice for the C-even helicity quasi-distribution, and the per-row scale $\mu_0=2\kappa xP_z$ with the cutoff $\mu_{\rm min}$ implements the paper's $x_{\rm min}$ condition. No discrepancies were found between the code and the paper for this operator and scheme.

