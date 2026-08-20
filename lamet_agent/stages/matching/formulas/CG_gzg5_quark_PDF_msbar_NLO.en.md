<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.15); digest=69ff5cce34e36f1e; paper_used=true -->
For the `gzg5` operator in the $\overline{\mathrm{MS}}$ scheme, the matching coefficient is given by Eq. (2.15) of arXiv:2602.11283. With $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the NLO coefficient is

$$
C_{\gamma^z\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)\,,
$$

where the $\gamma^t$ coefficient is

$$
C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} 
+ {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,
$$

and the ratio-scheme backbone is

$$
C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] 
+ \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.
$$

The plus prescription is defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,,
$$

with the subtraction point $x_0=1$ and domains $D=[0,1]$ and $D=(-\infty,\infty)$ as indicated. The $\delta(1-\xi)$ term in the $\gamma^z$ coefficient carries coefficient $1$ (from the explicit $+\delta(1-\xi)$), while the $\gamma^t$ diagonal carries the finite conversion $1+\ln(4p_z^2/\mu^2)$.

#### Consistency check

The code reproduces Eq. (2.15) term by term. The regular coefficient `C_msbar_gz` equals `C_msbar` (which is `C_ratio + 0.5/|1-ξ|`) plus `2(1-ξ)` on $0<\xi<1$, matching the paper. The plus-subtraction integrand `C_msbar_gz_plus` correctly restricts the `0.5/|1-ξ|` piece to $[0,2]$ and the `2(1-ξ)` piece to $[0,1]$, matching the two-domain split. The diagonal delta term carries `0.5(1+log_scale)+1.0`, i.e. the $\gamma^t$ finite term plus the extra $+1$ from the $\gamma^z$ delta, exactly as written. The `_atan_piece` implements the arctan/arctanh branch correctly: arctan for $\xi<1/2$, arctanh for $\xi>1/2$, with the analytic limit at $\xi=1/2$. All logarithms have arguments $|\xi|$, $|1-\xi|$, and $4y^2P_z^2/\mu^2$ as in the paper. No discrepancies found.
