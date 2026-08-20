<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20); digest=f932137f2422db5b; paper_used=true -->
### Matching coefficient for the `gzg5` operator in the hybrid scheme

The matching coefficient for the helicity quasi-PDF with $\tilde\Gamma = \gamma^z\gamma_5$ in the hybrid scheme is, at NLO,

$$
C_{\gamma^z\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + \delta C_{\gamma^z\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big),
$$

with $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The regular coefficient is

$$
C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] 
+ \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},
$$

where the plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').
$$

The hybrid-scheme correction is

$$
\delta C_{\gamma^z\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}.
$$

#### Consistency check

The code implements exactly the coefficient above: `C_ratio` reproduces the regular coefficient $C^{(1)}_r$ with the same splitting function, the same logarithms (with arguments $|\xi|$, $|1-\xi|$, and $4p_z^2/\mu^2$), the same arctan/arctanh branch (analytic at $\xi=1/2$), and the same $-\frac{3}{2|1-\xi|}$ term. The hybrid correction `delta` matches the paper’s $\delta C^{\mathrm{hyb.}}$ with the sine integral ${\rm Si}[(1-\xi)z_sp^z]$ and the factor $1/2$. The plus-prescription is implemented by making each $y$-column integrate to zero, which is equivalent to the paper’s $[g]^D_{+(1)}$ definition with the subtraction at $\xi=1$. The code also correctly includes the $\delta(1-\xi)$ term from the plus prescription. No discrepancies were found between the code and Eqs. (2.19)–(2.20) of arXiv:2602.11283.
