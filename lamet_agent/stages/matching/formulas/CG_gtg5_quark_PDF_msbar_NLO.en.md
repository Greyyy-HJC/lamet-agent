<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.14); digest=20da6a36ec813bf3; paper_used=true -->
$$C_{\gamma^t\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')$$

The code implements this as: the regular coefficient $C_{\rm msbar}(\xi,L) = C_r(\xi,L) + 0.5/|1-\xi|$ for all $\xi\neq 1$, with the delta subtraction using $C_{\rm msbar\_plus}$ (which restricts the $0.5/|1-\xi|$ term to $\xi\in[0,2]$) and a diagonal extra term $0.5(1+L)$.

#### Consistency check

The code reproduces Eq. (2.14) of arXiv:2602.11283 term by term. The regular coefficient $C_r$ matches exactly: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (analytic at $\xi=1/2$), and the $-3/(2|1-\xi|)$ tail. The $+0.5/|1-\xi|$ correction and the delta term $0.5\delta(1-\xi)[1+L-\int_0^2 d\xi'\,1/|1-\xi'|]$ are both present with the correct signs and arguments. The plus-prescription domains $[0,1]$ and $(-\infty,\infty)$ match the paper's notation exactly, and the code's delta subtraction correctly integrates $C_{\rm msbar\_plus}$ over the full domain with the $[0,2]$ restriction on the $0.5/|1-\xi|$ piece. No discrepancies found.
