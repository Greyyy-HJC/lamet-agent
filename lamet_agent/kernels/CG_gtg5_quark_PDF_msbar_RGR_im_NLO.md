<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=029196f8f4e063bb; paper_used=true -->
The matching coefficient for the `gtg5` operator in the `msbar` scheme, as implemented by the kernel, is the NLO+RGR (next-to-leading-order plus renormalization-group-resummed) coefficient. It is not a fixed-order expression; rather, each row $x$ is constructed by evaluating the fixed-order NLO $\overline{\rm MS}$ kernel at the row’s own scale $\mu_0(x) = 2\kappa x P_z$ and then DGLAP-evolving that row from $\mu_0(x)$ to the final scale $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function. Rows whose $\mu_0(x)$ falls below the perturbative cutoff $\mu_{\min}$ are set to zero, implementing the paper’s $x_{\min}$.

Define $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The fixed-order NLO $\overline{\rm MS}$ kernel (Eq. 2.14 of the paper) that seeds each row is:

$$C^{(1)}_{\overline{\rm MS}}(\xi, L) = C_{\rm ratio}(\xi, L) + \frac{1}{2|\xi-1|},$$

with the ratio-scheme coefficient (Eq. 2.16):

$$C_{\rm ratio}(\xi, L) = \begin{cases} \frac{1+\xi^2}{1-\xi}\left[L + \ln\frac{\xi}{\xi-1} - 1\right] + \xi - 1 - \frac{3}{2(1-\xi)} + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}}, & 0<\xi<1, \\[4pt] \frac{1+\xi^2}{1-\xi}\left[\ln\frac{-\xi}{1-\xi} - 1\right] + \xi - 1 - \frac{3}{2(1-\xi)} + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}}, & \xi<0, \\[4pt] \frac{1+\xi^2}{1-\xi}\left[\ln\frac{\xi}{\xi-1} - 1\right] + \xi - 1 - \frac{3}{2(1-\xi)} + \frac{3\xi-1}{\xi-1}\frac{\arctanh\sqrt{2\xi-1}}{\sqrt{2\xi-1}}, & \xi>1, \end{cases}$$

where the arctan/arctanh branch is chosen by the sign of $1-2\xi$, and the $\xi=1/2$ limit is taken analytically. The plus prescription is applied at $\xi=1$ with the paper’s exact notation, splitting the coefficient into two brackets over different domains:

$$C^{(1)}_{\overline{\rm MS}}(\xi, L) = \left[ \frac{1+\xi^2}{1-\xi}\left(L + \ln\frac{1-\xi}{\xi} - 1\right) + \xi - 1 + \frac{3}{2(1-\xi)} + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} \right]^{[0,1]}_{+(1)} + \left[ \frac{1+\xi^2}{1-\xi}\left(\ln\frac{\xi}{\xi-1} - 1\right) + \xi - 1 - \frac{3}{2(1-\xi)} + \frac{3\xi-1}{\xi-1}\frac{\arctanh\sqrt{2\xi-1}}{\sqrt{2\xi-1}} \right]^{(-\infty,\infty)}_{+(1)},$$

with the paper’s definition $[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi')$, and the $\delta(1-\xi)$ term is included implicitly through the plus prescription. The scheme-specific finite correction is the $+\frac{1}{2|\xi-1|}$ term, which is restricted to $\xi\in[0,2]$ in the plus-subtraction integrand (the paper’s $-\frac{1}{2}\int_0^2 d\xi'$ counterterm).

The RGR construction then replaces the fixed-order kernel: for each row $x$, the fixed-order matrix is evaluated at $\mu_0(x) = 2\kappa x P_z$ (with $\kappa$ the scale-variation parameter), and the row is evolved to $\mu$ by the operator

$$U(\mu_0, \mu) = \mathcal{P}\exp\left[\int_{\ln\mu_0^2}^{\ln\mu^2} \frac{d\ln t^2}{2} \left(\frac{\alpha_s(\sqrt{t})}{4\pi} P_{\rm LO} + \left(\frac{\alpha_s(\sqrt{t})}{4\pi}\right)^2 P_{\rm NLO}\right)\right],$$

where $P_{\rm LO}$ is the one-loop non-singlet splitting function and $P_{\rm NLO}$ is the two-loop valence-channel (non-singlet, $q-\bar{q}$) splitting function, which includes the $16C_F(C_F - C_A/2)$ C-parity term. The evolution is discretized as a product of matrix exponentials over $\ln\mu^2$ steps. Rows with $\mu_0(x) < \mu_{\min}$ are zeroed.

#### Consistency check

The code reproduces the paper’s Eq. (2.14) and (2.16) for the fixed-order NLO $\overline{\rm MS}$ kernel: the regular coefficient, the logarithms (with argument $4y^2P_z^2/\mu^2$), the arctan/arctanh branch, and the $+\frac{1}{2|\xi-1|}$ scheme correction all match the paper’s expressions. The plus prescription is implemented with the paper’s exact bracket notation, including the domain split $[0,1]$ and $(-\infty,\infty)$ and the subtraction point $+(1)$. The RGR construction follows the paper’s App. A method (Eq. matchingRGI): each row is matched at its own scale $2xP_z$ and DGLAP-evolved, with the cutoff implementing $x_{\min}$. No discrepancies were found between the code and the paper for the terms checked.

