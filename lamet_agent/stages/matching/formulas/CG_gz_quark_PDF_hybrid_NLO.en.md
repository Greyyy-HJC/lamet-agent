<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=777a174c378f8f21; paper_used=true -->
For the `gz` operator in the hybrid scheme, the matching coefficient is given by Eqs. (2.19)–(2.20) of arXiv:2602.11283. With $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the NLO coefficient is

$$
C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + \delta C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big),
$$

where the regular piece is

$$
C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},
$$

and the hybrid-scheme correction is

$$
\delta C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}.
$$

The plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').
$$

#### Consistency check

The code implements exactly the coefficient above. The regular coefficient $C^{(1)}_r$ matches Eq. (2.16) term by term: the splitting-function piece with $\ln(4p_z^2/\mu^2)$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (analytic at $\xi=1/2$), and the $-3/(2|1-\xi|)$ term, all with the same plus-prescription domains. The hybrid correction $\delta C_{\gamma^z(1)}^{\mathrm{hyb.}}$ matches Eq. (2.20) exactly, including the sine-integral argument $(1-\xi)z_sp^z$ and the $1/2$ prefactor. The code's plus prescription (making each $y$-column integrate to zero) reproduces the paper's $[g]^D_{+(1)}$ convention. No discrepancies were found between the code and Eqs. (2.19)–(2.20).
