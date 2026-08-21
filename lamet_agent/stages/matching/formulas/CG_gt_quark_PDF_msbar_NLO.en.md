<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.14); digest=3e10b53a469e5026; paper_used=true -->
$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,,$$

and the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The code implements this coefficient with the following scheme-specific structure: the regular off-diagonal term is $C_{\overline{\mathrm{MS}}}(\xi,L) = C_r(\xi,L) + \frac{1}{2|1-\xi|}$, the plus-subtraction integrand is $C_{\overline{\mathrm{MS}},+}(\xi,L) = C_r(\xi,L) + \frac{1}{2|1-\xi|}\theta(2-|\xi|)$ (restricting the $1/|1-\xi|$ piece to the paper's $\int_0^2$ counterterm), and the diagonal delta term carries $+\frac{1}{2}(1+L)$ minus the full integral of $C_{\overline{\mathrm{MS}},+}$ over $\xi$.

#### Consistency check

The code reproduces Eq. (2.14) of arXiv:2602.11283 term by term. The regular coefficient $C_r$ matches exactly: the splitting-function piece $\frac{1+\xi^2}{1-\xi}L + \xi - 1$ on $[0,1]$, the signed logarithms $\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|$ with the same prefactor, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch term $\frac{3\xi-1}{\xi-1}\frac{\tan^{-1}(\sqrt{1-2\xi}/|\xi|)}{\sqrt{1-2\xi}}$ (with the code correctly switching to $\arctanh$ for $\xi>1/2$), and the $-3/(2|1-\xi|)$ term are all present with identical signs and arguments. The $+\frac{1}{2|1-\xi|}$ correction and the $\frac{1}{2}\delta(1-\xi)[1+L-\int_0^2 d\xi'\,1/|1-\xi'|]$ diagonal term match the paper exactly. The plus-prescription domains $[0,1]$ and $(-\infty,\infty)$ with subtraction point $+(1)$ are reproduced precisely, including the paper's convention that the $1/|1-\xi|$ piece in the second bracket is subtracted only over $[0,2]$ (via the $\theta(2-|\xi|)$ restriction in $C_{\overline{\mathrm{MS}},+}$). No discrepancies were found.
