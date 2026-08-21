<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=945f252060964c3f; paper_used=true -->
$$C_{\gamma^t\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the regular coefficient is

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The arctan branch is chosen by $\xi$ relative to $1/2$: for $\xi<1/2$ use $\arctan(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$; for $\xi>1/2$ use $\arctanh(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$; at $\xi=1/2$ the analytic limit is $1/|\xi|$.

#### Consistency check

The code implements exactly the regular coefficient $C^{(1)}_r$ of Eq. (2.16) with the same splitting-function piece, the same signed logarithms, the same arctan/arctanh branch, and the same $-3/(2|1-\xi|)$ term. The plus-prescription is restored by the column-sum method, which matches the paper's $[g]^D_{+(1)}$ definition with the subtraction at $\xi=1$ and the split domains $[0,1]$ and $(-\infty,\infty)$. The code does not include the $1/(2|1-\xi|)$ or the $\delta(1-\xi)$ terms from the full $\overline{\mathrm{MS}}$ coefficient—these are the scheme-specific corrections that the ratio scheme omits, consistent with the paper's presentation of $C^{(1)}_r$ as the ratio-scheme kernel. No discrepancies found.
