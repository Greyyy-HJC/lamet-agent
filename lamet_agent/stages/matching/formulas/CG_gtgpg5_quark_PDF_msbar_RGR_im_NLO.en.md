<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=b2a2c777a05f35be; paper_used=true -->
$$C_{\rm RGR}^{\,(1)}\!\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{\rm ratio(1)}\!\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\,\frac{3}{2}\left(-\frac{1}{|1-\xi|} + \frac{2\,{\rm Si}[(1-\xi)|y|z_s P_z]}{\pi(1-\xi)}\right)\right]_{+(1)}^{[-\infty,\infty]} + \delta(1-\xi)\,,$$

with the ratio-scheme kernel (Eq. (2.18) of the paper, identical for $\overline{\rm MS}$ and hybrid at NLO for transversity):

$$C^{\rm ratio(1)}\!\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\begin{cases}
\left(\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} - \frac{1}{|1-\xi|}\right)_{+(1)}^{[1,\infty]} & \xi>1 \\[4pt]
\left(\frac{2\xi}{1-\xi}\left[-\ln\frac{\mu^2}{4x^2P_z^2} + \ln\frac{1-\xi}{\xi}\right] - \frac{1}{|1-\xi|}\right)_{+(1)}^{[0,1]} & 0<\xi<1 \\[4pt]
\left(-\frac{2\xi}{1-\xi}\ln\frac{-\xi}{1-\xi} - \frac{1}{|1-\xi|}\right)_{+(1)}^{[-\infty,0]} & \xi<0
\end{cases}$$

where the plus-prescription is defined as in the paper: for a function $g(\xi)$ with a singularity at $\xi=x_0$, $[g(\xi)]^{D}_{+(x_0)}$ acts on a test function $f$ as $\int_D d\xi\, [g(\xi)]^{D}_{+(x_0)} f(\xi) = \int_D d\xi\, g(\xi)\,[f(\xi)-f(x_0)]$, with the domain $D$ indicated by the superscript. The $\delta(1-\xi)$ term is the LO contribution.

The RGR kernel is not a fixed-order coefficient: each row $x$ is built by evaluating the above fixed-order kernel at the row's own scale $\mu_0(x) = 2\kappa x P_z$ (with $\kappa$ the scale-variation parameter, scanned over $0.8$–$1.2$ in the paper), then evolving to the common scale $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet DGLAP evolution kernel. The evolution operator is

$$\mathcal{E}(\mu_0,\mu) = \mathcal{P}\exp\!\left[\int_{\ln\mu_0^2}^{\ln\mu^2} \frac{d\ln\mu'^2}{2}\,\frac{\alpha_s(\mu')}{4\pi}\left(P^{(0)} + \frac{\alpha_s(\mu')}{4\pi}P^{(1)}\right)\right],$$

with $P^{(0)}$ the LO non-singlet splitting function $P_{qq}^{(0)}(\xi) = 2C_F(1+\xi^2)/(1-\xi)_+$ and $P^{(1)}$ the NLO transversity splitting function (the code uses the transversity variant, built on $4\xi/(1-\xi)$, not the unpolarized one). Rows whose $\mu_0(x)$ falls below the perturbative cutoff $\mu_{\min}=0.6$ GeV are set to zero, implementing the paper's $x_{\min}$.

The scheme-specific correction is absent: for transversity, $\overline{\rm MS}$, ratio, and hybrid schemes all coincide at NLO (Eqs. (2.17) and (2.21) of the paper), so no finite conversion term appears.

#### Consistency check

The code reproduces the paper's Eq. (2.18) for the ratio kernel exactly: the transversity splitting $2\xi/(1-\xi)$, the logarithms with arguments $\ln(\mu^2/(4x^2P_z^2))$ and $\ln((1-\xi)/\xi)$, the $-1/|1-\xi|$ tail, and the $\arctan/\arctanh$ branch (the `_atan_piece` function) all match. The plus-prescription is implemented via the column-sum subtraction, which is equivalent to the paper's $[\,\cdot\,]_{+(1)}^{D}$ notation with the correct domain split. The $\delta(1-\xi)$ term is present as the LO identity. The hybrid-scheme correction (the sine-integral term) matches Eq. (2.21) of the paper. The RGR construction follows App. 'A Method Solving RG Equation' (Eq. matchingRGI) exactly: per-row scale $\mu_0=2\kappa xP_z$, DGLAP evolution with the two-loop transversity kernel, and the $x_{\min}$ cutoff. No discrepancies were found between the code and the paper for this operator.
