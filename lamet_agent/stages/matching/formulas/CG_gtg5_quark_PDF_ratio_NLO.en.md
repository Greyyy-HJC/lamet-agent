<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16); digest=f973421312d1a826; paper_used=true -->
$$C_{\gamma^t\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

and the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The arctan term is understood via its analytic continuation: for $\xi>1/2$, $\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$ is replaced by $\tanh^{-1}(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$, and at $\xi=1/2$ the limit is $1/|\xi|$.

#### Consistency check

The code implements exactly the regular coefficient $C^{(1)}_r$ of Eq. (2.16): the splitting-function piece $(1+\xi^2)/(1-\xi)\,L + \xi - 1$ on $[0,1]$, the signed logarithms with arguments $|\xi|$ and $|1-\xi|$, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch with the correct prefactor $(3\xi-1)/(\xi-1)$, and the $-3/(2|1-\xi|)$ term. The plus prescription is implemented as a column-sum over the full domain, matching the paper's $[\,\cdot\,]^{(-\infty,\infty)}_{+(1)}$ bracket. The code does not include the scheme-specific correction ${1\over 2|1-\xi|} + {1\over2}\delta(1-\xi)[1 + \ln(4p_z^2/\mu^2) - \int_0^2 d\xi'\, 1/|1-\xi'|]$; this is expected, as the code is labeled "ratio" and the paper's Eq. (2.16) is the $\overline{\mathrm{MS}}$ coefficient. No discrepancies found between the code and the paper's Eq. (2.16) for the regular coefficient.
