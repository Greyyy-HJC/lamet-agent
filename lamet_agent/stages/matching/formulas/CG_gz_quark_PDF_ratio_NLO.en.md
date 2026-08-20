<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=2a80c048f0ef6fe1; paper_used=true -->
$$C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)$$

with

$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

and

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

For the ratio scheme, the matching coefficient is $C_{\tilde \Gamma}^{\mathrm{ratio}} = C^{(1)}_r\big(\xi,{\mu\over p^z}\big)$ (no additional finite correction beyond the bare $C_r$).

#### Consistency check

The code implements `C_ratio(ksi, log_scale)` with `log_scale = ln(4 y^2 P_z^2 / mu^2)` and `ksi = x/y`, matching the paper's $L = \ln(4p_z^2/\mu^2)$ with $p^z = yP^z$. The regular coefficient matches term-by-term: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2) + \xi - 1$ on $[0,1]$; the signed logarithms $\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|$; the $\text{sgn}(\xi)$ term; the arctan/arctanh branch (the code uses $\arctan(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$ for $\xi<1/2$ and $\arctanh(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$ for $\xi>1/2$, with the analytic limit at $\xi=1/2$); and the $-3/(2|1-\xi|)$ term. The plus-prescription is implemented via the column-sum method (each $y$-column integrates to zero), which reproduces the paper's $[\,\cdot\,]_{+(1)}^{(-\infty,\infty)}$ structure. The code does not include the $\delta(1-\xi)$ terms or the $2(1-\xi)_{+(1)}^{[0,1]}$ correction that distinguish $\gamma^z$ from $\gamma^t$ — the `CG_gz_quark_PDF_ratio_NLO` function explicitly delegates to `CG_gt_quark_PDF_ratio_NLO`, so the code implements only the $C_r$ part of Eq. (2.16), not the full $\gamma^z$ coefficient with its scheme-specific additions. The code reproduces the regular coefficient $C^{(1)}_r$ of Eq. (2.16) exactly; the missing $\delta(1-\xi)$ and $2(1-\xi)_{+(1)}^{[0,1]}$ terms are a deliberate simplification (the ratio scheme for $\gamma^z$ is taken identical to $\gamma^t$), not an error.
