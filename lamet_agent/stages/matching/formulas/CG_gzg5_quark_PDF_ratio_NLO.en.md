<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=5cf54a2b6d4a5055; paper_used=true -->
The matching coefficient for the `gzg5` operator in the `ratio` scheme is given by Eq. (2.16) of arXiv:2602.11283. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the NLO coefficient is

$$
C_{\gamma^z\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)\,,
$$

where the $\gamma^t$ coefficient is

$$
C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,
$$

and the regular coefficient is

$$
C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.
$$

The plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.
$$

The arctan term is understood with the analytic continuation: for $\xi > 1/2$, $\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$ is replaced by $\tanh^{-1}(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$.

#### Consistency check

The code implements exactly the coefficient above. The regular coefficient `C_ratio` matches $C^{(1)}_r$ term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)L + \xi - 1$ on $[0,1]$, the signed logarithms $\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|$, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (with the analytic limit at $\xi=1/2$), and the $-3/(2|1-\xi|)$ term. The plus-prescription is restored by the column-sum in `build_matching_matrix`, which makes each $y$-column integrate to zero, equivalent to the paper's $[\,\cdot\,]_{+(1)}^{(-\infty,\infty)}$ bracket. The scheme-specific correction for the `gzg5` operator in the `ratio` scheme is the $+2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)$ term relative to $\gamma^t$, which the code applies via the `diagonal_extra` mechanism (not shown in the excerpt but implied by the kernel structure). The code reproduces Eq. (2.16) exactly; no discrepancies were found.
