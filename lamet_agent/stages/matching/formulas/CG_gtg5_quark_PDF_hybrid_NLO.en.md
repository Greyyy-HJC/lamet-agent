<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=377815641542200b; paper_used=true -->
$$C_{\tilde\Gamma}^{\overline{\mathrm{MS}}}\big(\xi, {\mu\over p^z}\big) = \delta(\xi-1) + {\alpha_sC_F\over 2\pi}C_{\tilde\Gamma(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + {\cal O}(\alpha_s^2)\,,$$

with $C_F=4/3$, $\xi=x/y$, and $L=\ln(4y^2P_z^2/\mu^2)$. For the `gtg5` operator in the hybrid scheme, the NLO coefficient is

$$C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + \delta C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big)\,,$$

where the regular part is

$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,,$$

and the hybrid-scheme correction is

$$\delta C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}\,.$$

Here the plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,,$$

with the subtraction point $x_0=1$ (i.e., $+(1)$). The arctan term is analytic at $\xi=1/2$; for $\xi<1/2$ it uses $\arctan(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$, and for $\xi>1/2$ it uses $\arctanh(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$.

#### Consistency check

The code implements exactly the coefficient above. The regular coefficient $C^{(1)}_r$ matches Eq. (2.16) of the paper term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed logarithms $\text{sgn}(\xi)\ln|\xi|+\text{sgn}(1-\xi)\ln|1-\xi|$, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch, and the $-3/(2|1-\xi|)$ term, all on $(-\infty,\infty)$. The hybrid correction $\delta C$ matches Eq. (2.20) exactly, including the sine-integral argument $(1-\xi)z_sp^z$ and the factor $1/2$. The plus-prescription notation in the code (column-sum subtraction at $\xi=1$) reproduces the paper's $[g]^D_{+(1)}$ convention. The code's `C_hybrid` adds the correction to `C_ratio` with the same sign and prefactor as the paper. No discrepancies were found between the code and Eqs. (2.19)–(2.20) of arXiv:2602.11283.
