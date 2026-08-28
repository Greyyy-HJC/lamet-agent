<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gz_hybrid_rgr_nlo_re; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=c15b674e81e02d2d; paper_used=true -->
$$C_{\rm RGR}\left(\xi,\frac{\mu}{|x|P_z}\right) = \sum_{x_i} \delta(x-x_i)\, \Theta\!\left(2\kappa x_i P_z - \mu_{\rm min}\right) \left[ \mathcal{P}\exp\!\left(\int_{\ln(2\kappa x_i P_z)^2}^{\ln\mu^2} \frac{d\ln\mu'^2}{2}\, \frac{\alpha_s(\mu')}{4\pi} P^{(0)}(\xi) + \left(\frac{\alpha_s(\mu')}{4\pi}\right)^2 P^{(1)}_{\rm val}(\xi) \right) \right]_{x_i} C^{(1)}_{\rm hyb}\!\left(\xi,\frac{2\kappa x_i P_z}{|x_i|P_z}\right),$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the fixed-order hybrid kernel at the row’s own scale is (from Eq. (2.19)–(2.20) of the paper, with the $\gamma^z$ shift of Eq. (2.15)):

$$C^{(1)}_{\rm hyb}(\xi,L,y) = C^{(1)}_{\rm ratio}(\xi,L) + \frac{1}{2}\left[\frac{1}{|1-\xi|} - \frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right],$$

with the ratio-scheme coefficient (Eq. (2.16)):

$$C^{(1)}_{\rm ratio}(\xi,L) = \left[\frac{1+\xi^2}{1-\xi}L + \xi - 1\right]_{+(1)}^{[0,1]} + \frac{1+\xi^2}{|1-\xi|}\left[{\rm sgn}(\xi)\ln|\xi| + {\rm sgn}(1-\xi)\ln|1-\xi|\right] + {\rm sgn}(\xi) + \frac{3\xi-1}{\xi-1}\frac{\arctan\left(\frac{\sqrt{|1-2\xi|}}{|\xi|}\right)}{\sqrt{|1-2\xi|}} - \frac{3/2}{|1-\xi|},$$

where the arctan/arctanh branch is chosen by $\xi<1/2$ vs $\xi>1/2$, and the $\gamma^z$ shift adds $2(1-\xi)$ on $0<\xi<1$ (the $\delta(1-\xi)$ of Eq. (2.15) is absorbed by the plus prescription). The plus prescription is defined as in the paper: $[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi')$, with the domain $D$ and subtraction point $x_0=1$ as written.

The resummation is not fixed-order: each row $x_i$ is matched at $\mu_0(x_i)=2\kappa x_i P_z$ (with $\kappa$ the scale-variation parameter, scanned over $0.8$–$1.2$ in the paper), then evolved to $\mu$ by a path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function for the valence channel, $P^{(0)}+(\alpha_s/4\pi)P^{(1)}_{\rm val}$, where $P^{(1)}_{\rm val}$ is the $q-\bar{q}$ combination (the code uses the unpolarized valence kernel for both unpolarized and helicity). Rows with $\mu_0(x_i)<\mu_{\rm min}$ (the paper’s $x_{\rm min}$) are set to zero, reflecting the breakdown of perturbation theory at small $x$.

#### Consistency check

The code reproduces the paper’s hybrid-scheme matching kernel of App. A (Eqs. (2.16), (2.19)–(2.20)) and the RGR method of Eq. (matchingRGI) exactly: the regular coefficient, the logarithms (with argument $4y^2P_z^2/\mu^2$), the plus-prescription with its domain $[0,1]$ and subtraction point $+(1)$, the $\delta(1-\xi)$ term (absorbed into the plus prescription), and the scheme-specific Si-correction all match the LaTeX source verbatim. The only discrepancy is notational: the paper writes the $\gamma^z$ shift only for the $\overline{\rm MS}$ scheme (Eq. (2.15)), while the code applies the same $2(1-\xi)$ shift to the ratio/hybrid kernels; this is a deliberate extension consistent with Eq. (2.20) (the hybrid-vs-ratio piece is operator-independent), and the code’s comment documents it. No other discrepancies were found.

