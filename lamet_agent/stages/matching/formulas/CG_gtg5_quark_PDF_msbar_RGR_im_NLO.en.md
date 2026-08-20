<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=06b836da0d83b429; paper_used=true -->
$$C_{\rm RGR}^{\overline{\rm MS}}\left(\xi,\frac{\mu}{|x|P_z}\right) = \delta(1-\xi) + \frac{\alpha_s(\mu_0)}{2\pi} C_F \left[\,C^{\overline{\rm MS}}\left(\xi,L(\mu_0)\right)\right]^{(-\infty,\infty)}_{+(1)} + \mathcal{O}(\alpha_s^2),$$

where $\xi=x/y$, $L(\mu_0)=\ln(4y^2P_z^2/\mu_0^2)$, and the scale $\mu_0=2\kappa xP_z$ is chosen per row $x$ (with $\kappa$ an order-one variation parameter). The plus-prescription is defined as in the paper:

$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi'),$$

with the domain $D=(-\infty,\infty)$ and subtraction point $x_0=1$. The regular coefficient is

$$C^{\overline{\rm MS}}(\xi,L) = C_{\rm ratio}(\xi,L) + \frac{1}{2|\xi-1|},$$

with $C_{\rm ratio}$ given by Eq. (2.16) of the paper (the ratio-scheme kernel), including the splitting-function piece $(1+\xi^2)/(1-\xi)\,L$ for $0<\xi<1$, the signed logarithms, the $\arctan/\arctanh$ branch term, and the $-3/(2|\xi-1|)$ term. The scheme-specific correction is the $1/(2|\xi-1|)$ added to $C_{\rm ratio}$.

The resummation is implemented by evaluating the fixed-order kernel at $\mu_0(x)$ and then evolving to $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function for the valence channel (the $q-\bar{q}$ combination), using the code's `_p_nlo_valence`. Rows with $\mu_0(x)<\mu_{\rm min}$ (the perturbative cutoff, corresponding to the paper's $x_{\rm min}$) are set to zero.

#### Consistency check

The code reproduces the paper's Eq. (matchingRGI) in structure: the per-row scale $\mu_0=2\kappa xP_z$ matches the paper's $Q_{\rm eff}=2xP_zc'$, the DGLAP evolution operator matches the paper's solution method, and the valence-channel splitting function is the correct two-loop non-singlet kernel. The regular coefficient $C^{\overline{\rm MS}}$ matches the paper's Eq. (2.14) (the $\overline{\rm MS}$ kernel for the helicity case), including the $1/(2|\xi-1|)$ correction and the $\arctan/\arctanh$ branch. The plus-prescription domain $(-\infty,\infty)$ and subtraction point $+(1)$ match the paper's convention. The only discrepancy is that the paper's Eq. (2.14) writes the plus-prescription with the domain $[0,1]$ for the splitting-function piece, whereas the code applies a single $(-\infty,\infty)$ plus-bracket to the entire coefficient; this is a notational difference in how the plus-prescription is applied, not a numerical one, since the code's delta subtraction integrates over the full domain. No other discrepancies were found.
