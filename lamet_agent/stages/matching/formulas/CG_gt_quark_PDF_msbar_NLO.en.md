<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.14); digest=72638ddfeeb26a76; paper_used=true -->
$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,,$$

with the plus functions defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The code implements this coefficient as $C_{\overline{\mathrm{MS}}}(\xi,L) = C_r(\xi,L) + 0.5/|1-\xi|$ for the off-diagonal entries, with the delta-term subtraction using $C_{\overline{\mathrm{MS}},+}$ (where the $0.5/|1-\xi|$ is restricted to $\xi\in[0,2]$) and a diagonal extra of $0.5(1+L)$.

#### Consistency check

The code reproduces Eq. (2.14) of arXiv:2602.11283 term by term. The regular coefficient $C_r$ matches exactly: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (analytic at $\xi=1/2$), and the $-3/(2|1-\xi|)$ tail. The $+0.5/|1-\xi|$ correction and the delta-term structure (with the $[0,2]$ domain for the subtraction) are both correctly implemented. The plus-prescription is restored by the code's per-row delta subtraction, matching the paper's $[\,\cdot\,]_{+(1)}^{[0,1]}$ and $[\,\cdot\,]_{+(1)}^{(-\infty,\infty)}$ split. No discrepancies found.
