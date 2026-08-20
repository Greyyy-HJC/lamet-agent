<!-- lamet-agent formula cache; kernel=CG_gtg5_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=297e65ae148c37e5; paper_used=true -->
### Matching coefficient for `gtg5` in the hybrid scheme

The matching coefficient for the helicity quasi-PDF with Dirac structure $\tilde\Gamma = \gamma^t\gamma_5$, renormalized in the hybrid scheme, is given by Eqs. (2.19)–(2.20) of arXiv:2602.11283. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the NLO coefficient is

$$
C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi, {\mu\over p^z}, z_sp^z\big) = C^{(1)}_r\big(\xi, {\mu\over p^z}\big) + \delta C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi, z_sp^z\big),
$$

where the regular part is

$$
C^{(1)}_r\big(\xi, {\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},
$$

and the hybrid-scheme correction is

$$
\delta C_{\gamma^t\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi, z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}.
$$

The plus prescription is defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').
$$

The arctangent term is understood analytically across $\xi = 1/2$: for $\xi < 1/2$ it uses $\arctan(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$, and for $\xi > 1/2$ it uses $\arctanh(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$, with the limit $1/|\xi|$ at $\xi = 1/2$.

#### Consistency check

The code implements exactly the coefficient above. The regular coefficient $C^{(1)}_r$ matches Eq. (2.16) term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)\,L + \xi - 1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch, and the $-3/(2|1-\xi|)$ term on $(-\infty,\infty)$. The hybrid correction matches Eq. (2.20) with the sine integral ${\rm Si}[(1-\xi)z_sp^z]$ and the factor $1/2$. The plus prescription is implemented as the column-sum subtraction, which is equivalent to the paper's definition with $x_0 = 1$. The code reproduces Eqs. (2.19)–(2.20) of arXiv:2602.11283 exactly; no discrepancies were found.
