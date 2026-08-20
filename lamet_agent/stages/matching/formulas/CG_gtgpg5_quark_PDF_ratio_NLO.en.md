<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.18); digest=f859d313379e6e62; paper_used=true -->
## Matching coefficient for `gtgpg5` in the ratio scheme

The matching coefficient for the Coulomb-gauge transversity quasi-PDF with Dirac structure $\tilde\Gamma = \gamma^t\gamma_\perp^\alpha\gamma_5$, renormalized in the ratio scheme, is given by Eq. (2.18) of arXiv:2602.11283. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the NLO coefficient is

$$
C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,
$$

where

$$
C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)
= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}
+ \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big]
+ {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.
$$

The plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,,
$$

with the subtraction point $x_0 = 1$ (i.e., $+(1)$). The arctangent term is understood analytically across $\xi = 1/2$; for $\xi > 1/2$ the branch is $\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi} \to \tanh^{-1}(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$. For the transversity operator, the ratio, $\overline{\mathrm{MS}}$, and hybrid schemes all share this same coefficient (Eqs. 2.17, 2.21 of the paper).

#### Consistency check

The code implements `C_ratio_perp(ksi, log_scale)` with `log_scale = ln(4 y^2 P_z^2 / mu^2)`, matching the paper's $L$. The regular coefficient matches: the $[0,1]$ plus-bracket contains $2\xi/(1-\xi)\,L$; the $(-\infty,\infty)$ plus-bracket contains $2\xi/(1-\xi)[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|]$, the arctan/arctanh piece $(3\xi-1)/(\xi-1)\cdot\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$, and the tail $-1/|1-\xi|$. The plus prescription is implemented by column-wise subtraction over the full quasi grid, equivalent to the paper's $[g]^{D}_{+(1)}$ with $D = (-\infty,\infty)$ for the second bracket. No $\delta(1-\xi)$ term appears in the transversity coefficient, consistent with the paper. The code reproduces Eq. (2.18) exactly; no discrepancies found.
