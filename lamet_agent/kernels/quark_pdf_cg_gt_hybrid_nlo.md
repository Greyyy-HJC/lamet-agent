<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gt_hybrid_nlo; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=a46959d07db575e5; paper_used=true -->
### Matching coefficient for the `gt` operator in the hybrid scheme

The matching coefficient for the Coulomb-gauge $\gamma^t$ quasi-PDF in the hybrid scheme is given by Eqs. (2.19)–(2.20) of arXiv:2602.11283. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the coefficient is

$$
C_{\gamma^t(1)}^{\mathrm{hyb.}}\big(\xi, {\mu\over p^z}, z_sp^z\big) = C^{(1)}_r\big(\xi, {\mu\over p^z}\big) + \delta C_{\gamma^t(1)}^{\mathrm{hyb.}}\big(\xi, z_sp^z\big),
$$

where the regular part is

$$
C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},
$$

and the hybrid correction is

$$
\delta C_{\gamma^t(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}.
$$

The plus prescription on a domain $D$ is defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').
$$

The arctangent term is evaluated with the branch

$$
\frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} \to
\begin{cases}
\arctan\!\big(\sqrt{1-2\xi}/|\xi|\big)/\sqrt{1-2\xi}, & \xi < 1/2,\\[2pt]
\arctanh\!\big(\sqrt{2\xi-1}/|\xi|\big)/\sqrt{2\xi-1}, & \xi > 1/2,
\end{cases}
$$

with the analytic limit $(3\xi-1)/(\xi-1)/|\xi|$ at $\xi = 1/2$.

#### Consistency check

The code implements exactly the structure of Eqs. (2.19)–(2.20): the regular coefficient $C^{(1)}_r$ matches the paper term-by-term, including the splitting-function piece with $L$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch, and the $-3/(2|1-\xi|)$ term. The hybrid correction $\delta C_{\gamma^t(1)}^{\mathrm{hyb.}}$ reproduces the paper’s form with the sine integral $\mathrm{Si}[(1-\xi)z_sp^z]$ and the factor $1/2$. The plus prescription is implemented via the column-sum method, which is equivalent to the paper’s definition $[g]^D_{+(1)}$ with the subtraction at $\xi=1$ over the stated domains. The code’s `C_ratio` and `C_hybrid` functions match the paper’s Eqs. (2.16) and (2.19)–(2.20) exactly; no discrepancies were found.

