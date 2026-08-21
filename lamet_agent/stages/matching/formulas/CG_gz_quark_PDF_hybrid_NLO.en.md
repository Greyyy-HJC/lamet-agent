<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20) with the gamma^z shift of Eq. (2.15); digest=5189fda14f705511; paper_used=true -->
$$C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + 2(1-\xi) + \delta C_{(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big)$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription defined as
$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The regular coefficient is
$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$

where the arctan branch is taken for $\xi<1/2$ and the arctanh branch for $\xi>1/2$ (analytic at $\xi=1/2$). The scheme-specific correction is
$$\delta C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}$$

with $z_sp^z = z_s P^z$ and the sine-integral argument evaluated at $(1-\xi)|y|z_sp^z$ in the code. The $2(1-\xi)$ shift is the gamma^z vs gamma^t difference of Eq. (2.15), kept as a bare (non-plus-prescribed) term in the ratio/hybrid schemes.

#### Consistency check

The code reproduces Eqs. (2.19)–(2.20) with the gamma^z shift of Eq. (2.15) of arXiv:2602.11283, with the following discrepancies:

1. **The $2(1-\xi)$ shift**: The paper writes it only for MSbar with an accompanying $+\delta(1-\xi)$ (Eq. 2.15). The code applies the shift without the delta in the hybrid scheme, arguing the delta is MSbar-specific normalization. This is a deliberate deviation, not an error.

2. **The sine-integral argument**: The paper writes ${\rm Si}[(1-\xi)z_sp^z]$; the code evaluates ${\rm Si}[(1-\xi)|y|z_sp^z]$, inserting the parton momentum fraction $|y|$ into the Wilson-line scale. The paper's notation suppresses this $y$-dependence.

3. **The $1/(1-\xi)$ in the Si term**: The paper writes $2{\rm Si}[(1-\xi)z_sp^z]/[\pi(1-\xi)]$; the code uses the sign-safe denominator $1-\xi+\text{sgn}(1-\xi)\epsilon$, which matches the paper's expression in the limit $\epsilon\to0$.

4. **The $2(1-\xi)$ shift domain**: The code applies it only for $0<\xi<1$ (the physical window), whereas the paper's Eq. (2.15) writes it as a plus-function over $[0,1]$. The code's version is the bare (un-subtracted) form, consistent with the paper's statement that the plus-prescription plus delta equals the bare form.

All other terms — the splitting function $(1+\xi^2)/(1-\xi)$, the logarithms $\ln|\xi|$ and $\ln|1-\xi|$, the arctan/arctanh branch, the $-3/(2|1-\xi|)$ term, and the plus-prescription structure with domains $[0,1]$ and $(-\infty,\infty)$ — match the paper exactly.
