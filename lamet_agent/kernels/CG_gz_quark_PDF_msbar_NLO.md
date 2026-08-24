<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.15); digest=0ead97111420220a; paper_used=true -->
$$C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]$$

and

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

with the plus prescription defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')$$

The $\tan^{-1}$ term is understood via its analytic continuation (arctanh) for $\xi>1/2$, and the coefficient is analytic at $\xi=1/2$.

#### Consistency check

The code reproduces Eq. (2.15) of arXiv:2602.11283 term by term. The regular coefficient `C_msbar_gz` matches $C_{\gamma^z(1)}^{\overline{\mathrm{MS}}}$ exactly: it adds $2(1-\xi)$ to `C_msbar` on $0<\xi<1$, and `C_msbar` equals $C^{(1)}_r + 0.5/|1-\xi|$ with `C_ratio` matching $C^{(1)}_r$ including the splitting-function log, the signed logs, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch, and the $-1.5/|1-\xi|$ tail. The plus prescription is implemented via `C_msbar_gz_plus`, which restricts the $0.5/|1-\xi|$ subtraction to $\xi\in[0,2]$ (matching the $\int_0^2$ in the delta term) and includes the full $2(1-\xi)$ on $[0,1]$. The diagonal delta term carries $0.5(1+L)+1$, which is exactly the sum of the $0.5[1+\ln(4p_z^2/\mu^2)]$ from the $\gamma^t$ delta and the $+1$ from the $\delta(1-\xi)$ in Eq. (2.15). No discrepancies found.

