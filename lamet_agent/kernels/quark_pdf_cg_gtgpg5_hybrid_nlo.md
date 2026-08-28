<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gtgpg5_hybrid_nlo; arxiv=2602.11283; equations=Eq. (2.21); digest=db607a21b1d3f02b; paper_used=true -->
$$C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}  + \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] \nn\\
    \qquad + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.$$

The plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

For the hybrid scheme, the matching coefficient is

$$C_{(1)}^{\mathrm{hyb.}}\left(\xi,{\mu\over p^z},z_sp^z\right) = C^{(1)}_r\left(\xi,{\mu\over p^z}\right) + \delta C_{(1)}^{\mathrm{hyb.}}\left(\xi,z_sp^z\right)\,,$$

with

$$\delta C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = \delta C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big)= 0\,.$$

Thus, for the `gtgpg5` operator in the hybrid scheme, the matching coefficient is exactly $C^{\perp (1)}_r$ as given above, with no additional scheme-specific correction.

#### Consistency check

The code implements `quark_pdf_cg_gtgpg5_hybrid_nlo` by calling `quark_pdf_cg_gtgpg5_ratio_nlo`, which uses `C_ratio_perp`. Comparing term by term with Eq. (2.21) of arXiv:2602.11283 (which sets $\delta C^{\mathrm{hyb.}}=0$ for transversity, so the hybrid coefficient equals the ratio coefficient $C^{\perp(1)}_r$ of Eq. (2.18)):

- **Regular coefficient**: The code has $2\xi/(1-\xi)\ln(4p_z^2/\mu^2)$ for $0<\xi<1$, matching the first plus-bracket of the paper.
- **Logarithms**: The code has $2\xi/(1-\xi)[\mathrm{sgn}(\xi)\ln|\xi| + \mathrm{sgn}(1-\xi)\ln|1-\xi|]$, matching the second bracket.
- **Arctan/arctanh branch**: The code implements the $\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$ term with the correct branch: arctan for $\xi<1/2$, arctanh for $\xi>1/2$, and the analytic limit at $\xi=1/2$. This matches the paper's expression.
- **Tail term**: The code has $-1/|1-\xi|$, matching the paper.
- **Plus prescription**: The code uses the column-sum prescription (each $y$-column integrates to zero), which is equivalent to the paper's $[g]^D_{+(1)}$ definition with $x_0=1$ and domains $[0,1]$ and $(-\infty,\infty)$ as written.
- **Delta term**: No explicit $\delta(1-\xi)$ term appears in the transversity coefficient, consistent with the paper.
- **Scheme correction**: The code sets $\delta C^{\mathrm{hyb.}}=0$ for transversity, matching Eq. (2.21).

No discrepancies found. The code reproduces Eq. (2.21) of arXiv:2602.11283 exactly.

