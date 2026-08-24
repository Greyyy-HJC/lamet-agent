<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_hybrid_NLO; arxiv=2602.11283; equations=Eqs. (2.19)-(2.20) with the gamma^z shift of Eq. (2.15); digest=585da36fd1bce6bf; paper_used=true -->
$$C_{\gamma^z\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,{\mu\over p^z},z_sp^z\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta C_{(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big)$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription defined as in the paper:
$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The regular coefficient is
$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]}  + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$
where the arctan branch is taken for $\xi<1/2$ and the arctanh branch for $\xi>1/2$ (analytic at $\xi=1/2$).

The scheme-specific hybrid correction is
$$\delta C_{\gamma^z\gamma_5(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = \delta C_{\gamma^z(1)}^{\mathrm{hyb.}}\big(\xi,z_sp^z\big) = {1\over2}\left[{1\over |1-\xi|} - {2{\rm Si}[(1-\xi)z_sp^z]\over \pi(1-\xi)}\right]_{+(1)}^{(-\infty, \infty)}$$
with $z_sp^z = z_s P^z$ and the sine-integral argument evaluated at $(1-\xi)|y|z_sp^z$ in the code (the $|y|$ factor arises from the parton momentum $yP^z$).

The $2(1-\xi)_{+(1)}^{[0,1]}$ term is the gamma^z shift of Eq. (2.15); the accompanying $+\delta(1-\xi)$ in that equation is a normalization term belonging to $\overline{\mathrm{MS}}$ alone and is not included here, as the ratio/hybrid schemes divide the normalization out.

#### Consistency check

The code reproduces Eqs. (2.19)–(2.20) with the gamma^z shift of Eq. (2.15) of arXiv:2602.11283, with the following discrepancies:

1. **Gamma^z shift domain**: The paper writes the shift as $2(1-\xi)_{+(1)}^{[0,1]}$ (Eq. 2.15). The code implements it as a bare $2(1-\xi)$ on $0<\xi<1$ (no explicit plus-subtraction), relying on the shared column-sum prescription to restore the plus form. This is equivalent for the physical window but differs in notation.

2. **Delta term**: The paper's Eq. (2.15) includes $+\delta(1-\xi)$ alongside the shift. The code deliberately omits it, arguing it is an $\overline{\mathrm{MS}}$-only normalization term. This is a genuine omission relative to the paper's written form, though the code's rationale is that the ratio/hybrid schemes divide out the normalization.

3. **Hybrid correction argument**: The paper writes ${\rm Si}[(1-\xi)z_sp^z]$ with $z_sp^z$ constant. The code evaluates ${\rm Si}[(1-\xi)|y|z_sp^z]$, inserting the $|y|$ factor from the parton momentum. This is a discrepancy in the argument of the sine integral.

4. **Arctan branch**: The paper's Eq. (2.16) writes $\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$ without specifying the branch. The code explicitly uses arctan for $\xi<1/2$ and arctanh for $\xi>1/2$, which is the correct analytic continuation and matches the paper's intent.

All other terms—the splitting function, the logarithms and their arguments, the plus-prescription domains, and the $1/|1-\xi|$ structure—match exactly between code and paper.

