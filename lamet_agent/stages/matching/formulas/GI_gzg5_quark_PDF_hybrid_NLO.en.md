<!-- lamet-agent formula cache; kernel=GI_gzg5_quark_PDF_hybrid_NLO; arxiv=2604.00143; equations=Eqs. (C6)-(C8); digest=e083728d16229637; paper_used=true -->
The matching coefficient for the `gzg5` operator in the hybrid scheme is given by Eqs. (C6)–(C8) of arXiv:2604.00143. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the NLO kernel is

$$
\mathcal{C}(x,y,\mu,P_z) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi} \left[ C^{(1)}_{\text{ratio}}(\xi,L) + C^{(1)}_{\text{hybrid}}(\xi,y,z_sP_z) \right],
$$

where the ratio-scheme piece, Eq. (C7), is

$$
C^{(1)}_{\text{ratio}}(\xi,L) = \left[ \frac{1+\xi^2}{1-\xi} \left( L + \ln\frac{4\xi(1-\xi)}{1} - 1 \right) + 1 \right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2|1-\xi|},
$$

with the plus-prescription defined as

$$
[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi'),
$$

and the hybrid correction, Eq. (C8), is

$$
C^{(1)}_{\text{hybrid}}(\xi,y,z_sP_z) = \frac{3}{2}\left[ -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_sP_z)}{\pi(1-\xi)} \right].
$$

The full coefficient is the sum of these two pieces, with the plus-prescription applied to the combined expression over the domain $(-\infty,\infty)$ at the subtraction point $x_0=1$.

#### Consistency check

The code implements exactly the structure above. The ratio piece matches Eq. (C7) term by term: the splitting function $(1+\xi^2)/(1-\xi)$, the log combination $L + \ln(4\xi(1-\xi)) - 1$ (with the code's $\ln(4y^2P_z^2/\mu^2)$ correctly reduced by $\ln 4$ to match the paper's $\ln(y^2P_z^2/\mu^2)$), the constant $+1$, and the $3/(2|1-\xi|)$ tail. The hybrid piece matches Eq. (C8) with the correct prefactor $3/2$ and the sine-integral argument $(1-\xi)|y|z_sP_z$. The plus-prescription is restored by the column-sum method, which is equivalent to the paper's $[\,\cdot\,]^{(-\infty,\infty)}_{+(1)}$ definition. The code omits the $\delta C_M$ (leading-renormalon) term and the NNLO piece of Eq. (C6), as stated in its docstring, but these are not part of the NLO coefficient being documented. No discrepancies found.
