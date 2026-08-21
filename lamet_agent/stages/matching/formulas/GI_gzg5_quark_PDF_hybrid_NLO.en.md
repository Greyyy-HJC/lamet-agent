<!-- lamet-agent formula cache; kernel=GI_gzg5_quark_PDF_hybrid_NLO; arxiv=2604.00143; equations=Eqs. (C6)-(C8); digest=b13bac72af1379b9; paper_used=true -->
$$C_{\mathrm{hybrid}}^{g_zg_5}(\xi, L, y, z_s P_z) = C_{\mathrm{ratio}}^{g_zg_5}(\xi, L) + \Delta C_{\mathrm{hybrid}}(\xi, y, z_s P_z),$$

with $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The ratio-scheme coefficient is

$$C_{\mathrm{ratio}}^{g_zg_5}(\xi, L) = \left[ \frac{1+\xi^2}{1-\xi} \left( L + \ln\frac{4\xi(1-\xi)}{1} - 1 \right) + 1 \right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2|1-\xi|} + 2(1-\xi)\quad (0<\xi<1),$$

where the plus-prescription is defined as in the paper:

$$\int_0^1 d\xi\, [g(\xi)]^{D}_{+(x_0)}\,\varphi(\xi) = \int_0^1 d\xi\, g(\xi)\big(\varphi(\xi)-\varphi(x_0)\big),$$

with $D=(-\infty,\infty)$ and $x_0=1$. The hybrid-scheme correction is

$$\Delta C_{\mathrm{hybrid}}(\xi, y, z_s P_z) = \frac{3}{2}\left[ -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_s P_z\big)}{\pi(1-\xi)} \right],$$

where $\mathrm{Si}(z)$ is the sine integral. The full coefficient is

$$C_{\mathrm{hybrid}}^{g_zg_5}(\xi, L, y, z_s P_z) = \left[ \frac{1+\xi^2}{1-\xi} \left( L + \ln\frac{4\xi(1-\xi)}{1} - 1 \right) + 1 \right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2|1-\xi|} + 2(1-\xi) + \frac{3}{2}\left[ -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_s P_z\big)}{\pi(1-\xi)} \right],$$

with the plus-prescription at $\xi=1$ applied to the first bracket, and the $\delta(1-\xi)$ term from the LO is implicit in the discretization.

#### Consistency check

The code reproduces Eqs. (C6)–(C8) of arXiv:2604.00143 term by term. The regular coefficient matches: the splitting function $(1+\xi^2)/(1-\xi)$, the log argument $L + \ln(4\xi(1-\xi))$, the constant $-1$, and the $+1$ term are all present. The $3/(2|1-\xi|)$ tail and the $2(1-\xi)$ shift for $\gamma^z$ vs $\gamma^t$ are correctly implemented. The hybrid correction matches Eq. (C8) exactly: the prefactor $3/2$, the $-1/|1-\xi|$ term, and the sine-integral term with argument $(1-\xi)|y|z_sP_z$ are all reproduced. The plus-prescription is implemented as the paper defines it, with the subtraction point at $\xi=1$ and the domain $(-\infty,\infty)$. The only discrepancy is that the code does not implement the $\delta C_M$ (leading-renormalon/mass) term and the NNLO piece mentioned in Eq. (C6), but these are explicitly documented as not implemented in the code. No other discrepancies found.
