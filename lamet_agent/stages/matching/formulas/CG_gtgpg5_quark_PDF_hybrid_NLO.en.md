<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eq. (2.21); digest=60fe98aaa0201200; paper_used=true -->
$$C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, where

$$C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}  + \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] \nn$$
$$\qquad + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.$$

The plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

For the hybrid scheme, the matching coefficient is

$$C_{(1)}^{\mathrm{hyb.}}\left(\xi,{\mu\over p^z},z_sp^z\right) = C^{(1)}_r\left(\xi,{\mu\over p^z}\right) + \delta C_{(1)}^{\mathrm{hyb.}}\left(\xi,z_sp^z\right)\,,$$

with the scheme-specific correction for the transversity case

$$\delta C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = \delta C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big)= 0\,.$$

Thus, for the `gtgpg5` operator in the hybrid scheme, the matching coefficient is exactly $C^{\perp (1)}_r$ with no additional finite correction.

#### Consistency check

The code implements `CG_gtgpg5_quark_PDF_hybrid_NLO` by delegating to `CG_gtgpg5_quark_PDF_ratio_NLO` (dropping `zspz`), which uses `C_ratio_perp`. Comparing term by term against Eq. (2.21) of arXiv:2602.11283:

- **Regular coefficient**: The code's `C_ratio_perp` contains $2\xi/(1-\xi)\ln(4p_z^2/\mu^2)$ for $0<\xi<1$ (the $[0,1]$ plus-bracket) and the second bracket with $2\xi/(1-\xi)[\text{sgn}(\xi)\ln|\xi|+\text{sgn}(1-\xi)\ln|1-\xi|]$, the arctan/arctanh piece, and $-1/|1-\xi|$. This matches the paper exactly.
- **Logarithms and arguments**: The code uses `_pdf_log_scale` giving $L=\ln(4y^2P_z^2/\mu^2)$, and the coefficient is evaluated at $\xi=x/y$, so the log argument becomes $\ln(4\xi^2 y^2 P_z^2/\mu^2)=\ln(4x^2P_z^2/\mu^2)$ after the $dy/|y|$ convolution — consistent with the paper's $\ln(4p_z^2/\mu^2)$ at $\xi=x/y$.
- **Plus-prescription**: The code uses the column-sum prescription (each $y$-column integrates to zero), which is the discretized form of $[g(\xi)]^{D}_{+(1)}$ with the paper's exact notation. The domain split into $[0,1]$ and $(-\infty,\infty)$ is reproduced: the first bracket only contributes for $0<\xi<1$, the second for all $\xi\neq1$.
- **Delta term**: No explicit $\delta(1-\xi)$ term appears in the transversity coefficient, matching the paper (the plus-prescription handles the singularity).
- **Scheme-specific correction**: The code sets `delta C_hyb = 0` for transversity, matching Eq. (2.21).

**Verdict**: The code reproduces Eq. (2.21) of arXiv:2602.11283 exactly. No discrepancies found.
