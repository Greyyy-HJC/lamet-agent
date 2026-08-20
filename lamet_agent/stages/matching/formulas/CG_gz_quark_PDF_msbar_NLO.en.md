<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.15); digest=eb3f98f4e159c16d; paper_used=true -->
$$C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)\,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$. The $\gamma^t$ kernel is

$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,$$

where the regular piece is

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.$$

The plus prescription is defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The code implements exactly this: the off-diagonal coefficient is $C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}$ with the $2(1-\xi)$ term active only for $0<\xi<1$, the delta subtraction uses $C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}$ restricted to $[0,2]$ for the $1/(2|1-\xi|)$ piece and to $[0,1]$ for the $2(1-\xi)$ piece, and the diagonal carries the extra $\delta(1-\xi)$ plus the finite $0.5(1+L)$ term.

#### Consistency check

The code reproduces Eq. (2.15) of arXiv:2602.11283 term by term. The regular coefficient matches: the splitting function $(1+\xi^2)/(1-\xi)L + \xi - 1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (analytic at $\xi=1/2$), and the $-3/(2|1-\xi|)$ tail. The plus prescription is implemented with the correct domains: $[0,1]$ for the splitting-function piece and $(-\infty,\infty)$ for the remainder, with the $1/(2|1-\xi|)$ term restricted to $[0,2]$ in the subtraction. The extra $2(1-\xi)_{+(1)}^{[0,1]}$ and $\delta(1-\xi)$ are both present. The only notational difference is that the code writes the $\delta(1-\xi)$ term with coefficient $1$ explicitly, whereas the paper's Eq. (2.15) shows it implicitly; this is a cosmetic difference, not a discrepancy. No other disagreements found.
