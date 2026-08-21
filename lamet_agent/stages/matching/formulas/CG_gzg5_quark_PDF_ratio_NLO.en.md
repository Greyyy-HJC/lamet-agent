<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.16) with the gamma^z shift of Eq. (2.15); digest=fd954ef187d8a1ff; paper_used=true -->
$$C_{\gamma^z\gamma_5(1)}^{\mathrm{ratio}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z(1)}^{\mathrm{ratio}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} ,$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the regular coefficient is

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)} .$$

The plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\, ,$$

with $x_0=1$ (i.e. $+(1)$). The $\gamma^z$ shift of Eq. (2.15) is applied as a hard-coefficient term $2(1-\xi)$ on $0<\xi<1$, plus-prescribed by the shared discretization; the $\delta(1-\xi)$ of Eq. (2.15) is not included because the ratio scheme divides out the normalization.

#### Consistency check

The code reproduces Eq. (2.16) with the $\gamma^z$ shift of Eq. (2.15) exactly. Term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)\,L + \xi-1$ on $[0,1]$ matches; the signed logarithms $\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|$ match; the arctan/arctanh branch at $\xi=1/2$ matches (analytic limit); the $-3/(2|1-\xi|)$ term matches; the plus-prescription domains $[0,1]$ and $(-\infty,\infty)$ with subtraction point $+(1)$ match; the $\gamma^z$ shift $2(1-\xi)$ on $0<\xi<1$ matches. The only deliberate difference is the omission of the $\delta(1-\xi)$ from Eq. (2.15), which the code correctly identifies as an MSbar-only normalization term not present in the ratio scheme. No discrepancies found.
