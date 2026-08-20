<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=acea7e953468cc83; paper_used=true -->
$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

with the ratio-scheme coefficient (Eq. (2.16) of the paper)

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The code implements exactly this $C^{(1)}_r$ (the `C_ratio` function), with the $\tan^{-1}$ branch chosen as $\arctan$ for $\xi<1/2$ and $\arctanh$ for $\xi>1/2$ (analytic at $\xi=1/2$), and the plus prescription enforced by making each $y$-column of the discretized kernel integrate to zero over the full domain.

#### Consistency check

The code reproduces Eq. (2.16) of arXiv:2602.11283 term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$; the signed logarithms $\text{sgn}(\xi)\ln|\xi|+\text{sgn}(1-\xi)\ln|1-\xi|$; the $\text{sgn}(\xi)$ term; the arctan/arctanh branch term $(3\xi-1)/(\xi-1)\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$; and the $-3/(2|1-\xi|)$ term, all on $(-\infty,\infty)$. The plus prescription with subtraction at $\xi=1$ and the two-domain split $[0,1]$ and $(-\infty,\infty)$ match the paper exactly. The code's `C_ratio` is the paper's $C^{(1)}_r$; the additional $1/(2|1-\xi|)$ and $\delta(1-\xi)$ terms in the full $\overline{\mathrm{MS}}$ coefficient are handled separately in the code's assembly (not in `C_ratio` itself), consistent with the paper's structure. No discrepancies found.
