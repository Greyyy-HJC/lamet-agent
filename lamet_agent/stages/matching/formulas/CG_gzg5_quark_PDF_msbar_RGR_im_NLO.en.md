<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=b14e182e08da1a7c; paper_used=true -->
The matching coefficient for the `gzg5` operator in the `msbar` scheme, as implemented by the kernel, is the NLO+RGR (next-to-leading-order plus renormalization-group-resummed) coefficient. It is built row-by-row in $x$: for each $x$, the fixed-order $\overline{\rm MS}$ kernel is evaluated at the row’s own scale $\mu_0(x) = 2\kappa x P^z$ (with $\kappa$ a scale-variation parameter, $c'$ in the paper), and then DGLAP-evolved to the final scale $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function. Rows where $\mu_0(x) < \mu_{\min}$ (the perturbative cutoff, corresponding to the paper’s $x_{\min}$) are set to zero.

The fixed-order $\overline{\rm MS}$ kernel for `gzg5` is identical to that for `gz` (Eq. 2.15 of the code’s reference), which is the `gt` kernel plus a scheme-specific finite correction. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the regular (off-diagonal) coefficient is

$$
C^{\overline{\rm MS}}_{gzg5}(\xi, L) = C^{\overline{\rm MS}}_{gz}(\xi, L) = C^{\overline{\rm MS}}_{gt}(\xi, L) + 2(1-\xi) \quad (0<\xi<1),
$$

where

$$
C^{\overline{\rm MS}}_{gt}(\xi, L) = C_{\rm ratio}(\xi, L) + \frac{1}{2|1-\xi|},
$$

and

$$
C_{\rm ratio}(\xi, L) = \frac{1+\xi^2}{1-\xi} L + \xi - 1 + \frac{1+\xi^2}{1-\xi}\left[\operatorname{sgn}(\xi)\ln|\xi| + \operatorname{sgn}(1-\xi)\ln|1-\xi|\right] + \operatorname{sgn}(\xi) + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} - \frac{3}{2|1-\xi|},
$$

with the arctan branch switching to $\operatorname{arctanh}\sqrt{2\xi-1}$ for $\xi>1/2$ (analytic at $\xi=1/2$). The plus-prescription at $\xi=1$ is implemented by the code’s discretization, which makes each $y$-column integrate to zero. In the paper’s notation, the full NLO kernel is split into plus-brackets over different domains:

$$
C^{(1)}(\xi, L) = \left[\,C^{\overline{\rm MS}}_{gzg5}(\xi, L)\,\right]^{[0,1]}_{+(1)} + \left[\,C^{\overline{\rm MS}}_{gzg5}(\xi, L)\,\right]^{(-\infty,\infty)}_{+(1)} + \delta(1-\xi)\left(1 + \frac{1}{2}(1+L)\right),
$$

where the first bracket covers $0<\xi<1$ and the second covers $\xi<0$ and $\xi>1$, with the plus-prescription defined as

$$
\int_a^b d\xi\, [g(\xi)]^{[a,b]}_{+(x_0)} f(\xi) = \int_a^b d\xi\, g(\xi)\left[f(\xi) - f(x_0)\right].
$$

The scheme-specific finite correction is the extra $\delta(1-\xi)$ term (coefficient $1$) and the $2(1-\xi)$ off-diagonal piece, both from Eq. 2.15 of the code’s reference.

#### Consistency check

The code’s fixed-order `gzg5` kernel reproduces Eq. 2.15 of arXiv:2602.11283 (the code’s reference), which is the `gz` kernel: it equals the `gt` kernel plus $2(1-\xi)$ off-diagonal and a $\delta(1-\xi)$ diagonal term. The `gt` kernel matches the paper’s Eq. (2.14) (the $\overline{\rm MS}$ kernel) with the ratio-scheme backbone of Eq. (2.16), including the arctan/arctanh branch and the $0.5/|1-\xi|$ conversion. The plus-prescription is implemented via the column-sum subtraction, equivalent to the paper’s $[\,\cdot\,]^{D}_{+(1)}$ with the domain split as above. The RGR procedure follows App. A of arXiv:2209.01236 (Eq. matchingRGI): each row is matched at $\mu_0 = 2\kappa xP^z$ and evolved to $\mu$ using the two-loop non-singlet splitting function for the valence channel (the code uses the valence variant, which differs from the full channel by the $16C_F(C_F-C_A/2)$ term). The cutoff $\mu_{\min}$ implements the paper’s $x_{\min}$. No discrepancies were found between the code and the paper’s notation or the implemented terms; the code follows the paper’s App. A exactly, with the only difference being that the paper’s $c'$ is called $\kappa$ in the code.
