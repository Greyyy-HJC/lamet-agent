<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gtgpg5_msbar_nlo; arxiv=2602.11283; equations=Eq. (2.17); digest=ccdcc88a113341af; paper_used=true -->
$$C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$
C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}  + \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] \nn\\
\qquad + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.
$$

Here the plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,,
$$

with the subtraction point $x_0=1$ (i.e., $+(1)$). The first bracket is restricted to the domain $[0,1]$, while the second extends over $(-\infty,\infty)$. The arctangent branch is chosen by the position of $\xi$ relative to $1/2$: for $\xi<1/2$ the argument is $\sqrt{1-2\xi}/|\xi|$ with $\tan^{-1}$, while for $\xi>1/2$ it becomes $\sqrt{2\xi-1}/|\xi|$ with $\tanh^{-1}$; the expression is analytic at $\xi=1/2$. There is no scheme-specific finite correction beyond the plus prescription, and no $\delta(1-\xi)$ term appears explicitly in this coefficient.

#### Consistency check

The code implements `C_ratio_perp(ksi, log_scale)` with `log_scale = ln(4 y^2 P_z^2 / mu^2)`, matching the paper's $L$. The regular coefficient is $2\xi/(1-\xi)\,L$ on $[0,1]$, exactly as in the first plus-bracket. The second bracket contains $2\xi/(1-\xi)[\mathrm{sgn}(\xi)\ln|\xi|+\mathrm{sgn}(1-\xi)\ln|1-\xi|]$, the arctan/arctanh piece $(3\xi-1)/(\xi-1)\cdot\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$, and $-1/|1-\xi|$ — all matching the paper term by term. The plus prescription is implemented by making each $y$-column integrate to zero over the full domain, which reproduces the paper's $[g]^D_{+(1)}$ convention with the subtraction at $\xi=1$. The code correctly omits the $+\xi-1$ and $+\mathrm{sgn}(\xi)$ terms that appear in the unpolarized/helicity kernel but not in the transversity one, and it correctly uses $-1/|1-\xi|$ rather than $-3/(2|1-\xi|)$. The branch of the arctangent is handled exactly as described. No discrepancies were found: the code reproduces Eq. (2.17) of arXiv:2602.11283 verbatim.

