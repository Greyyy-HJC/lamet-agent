<!-- lamet-agent formula cache; kernel=CG_gt_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=4bd606299fc2d73f; paper_used=true -->
For the `gt` operator in the hybrid scheme, the matching coefficient is given by Eqs. (2.19)–(2.20) of arXiv:2602.11283. With $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the coefficient is

$$
C_{\gamma^t(1)}^{\mathrm{hyb.}}\left(\xi,{\mu\over p^z},z_sp^z\right) = C^{(1)}_r\left(\xi,{\mu\over p^z}\right) + \delta C_{\gamma^t(1)}^{\mathrm{hyb.}}\left(\xi,z_sp^z\right),
$$

where the regular part is

$$
C^{(1)}_r\left(\xi,{\mu\over p^z}\right) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},
$$

and the hybrid correction is

$$
\delta C_{\gamma^t(1)}^{\mathrm{hyb.}}\left(\xi,z_sp^z\right) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}.
$$

The plus prescription is defined as in the paper:

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').
$$

The arctangent term is analytic at $\xi=1/2$; for $\xi>1/2$ the branch switches to $\tanh^{-1}(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$.

#### Consistency check

The code implements exactly the coefficient above. The regular part `C_ratio` matches Eq. (2.16) term by term: the splitting-function piece with $L$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch, and the $-3/(2|1-\xi|)$ term, all with the same plus-prescription domains. The hybrid correction `delta` in `C_hybrid` matches Eq. (2.20) exactly, including the factor $1/2$, the $1/|1-\xi|$ term, the sine-integral term with ${\rm Si}[(1-\xi)z_sp^z]$, and the $(-\infty,\infty)$ plus domain. The code's `_atan_piece` reproduces the paper's branch structure, including the analytic limit at $\xi=1/2$. The code's plus prescription (column-sum to zero) is equivalent to the paper's definition. No discrepancies were found between the code and Eqs. (2.19)–(2.20).
