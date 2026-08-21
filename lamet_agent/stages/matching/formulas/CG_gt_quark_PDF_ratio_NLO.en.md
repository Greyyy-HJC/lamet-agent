<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=8869e6267faec100; paper_used=true -->
$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the regular coefficient is

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The code implements exactly this $C^{(1)}_r$ (the `C_ratio` function), with the $\tan^{-1}$ branch chosen as $\arctan$ for $\xi<1/2$ and $\arctanh$ for $\xi>1/2$, and the analytic limit at $\xi=1/2$. The scheme-specific correction is the $+{1\over 2|1-\xi|}$ term and the $\delta(1-\xi)$ term with its integral counterterm, which the code adds via the `diagonal_extra` mechanism in the matching matrix construction.

#### Consistency check

The code reproduces Eq. (2.16) of arXiv:2602.11283 exactly. Term-by-term comparison:

- **Regular coefficient**: The code's `C_ratio` matches the paper's $C^{(1)}_r$ precisely, including the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh piece, and the $-3/(2|1-\xi|)$ term.
- **Logarithms**: All arguments match: $\ln(4p_z^2/\mu^2)$, $\ln|\xi|$, $\ln|1-\xi|$ — no discrepancies.
- **Plus prescription**: The code uses the column-sum prescription (each $y$-column integrates to zero), which is equivalent to the paper's $[g]^D_{+(1)}$ with the subtraction at $\xi=1$. The domain split $[0,1]$ and $(-\infty,\infty)$ is preserved.
- **Delta term**: The paper's $\frac{1}{2}\delta(1-\xi)[1+\ln(4p_z^2/\mu^2)-\int_0^2 d\xi'\,1/|1-\xi'|]$ is implemented via the `diagonal_extra` correction, which adds exactly this finite piece.
- **Scheme correction**: The $+1/(2|1-\xi|)$ term is present in the code's assembly (it appears as the `+ 0.5/|1-ksi|` in the `C_ratio` call structure, though the code splits it into the `diagonal_extra` path).

No discrepancies found. The code faithfully implements Eq. (2.16) of the paper.
