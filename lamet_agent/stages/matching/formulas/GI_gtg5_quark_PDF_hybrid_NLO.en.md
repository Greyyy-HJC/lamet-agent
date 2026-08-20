<!-- lamet-agent formula cache; kernel=GI_gtg5_quark_PDF_hybrid_NLO; arxiv=2412.20461; equations=Eq. (24); digest=f0736a7a9f28fbbd; paper_used=true -->
$$C_{q_iq_i}^{\text{hyb-r}}\left(\xi,\frac{\mu}{yP^z}, y z_s P^z\right) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi} \left[ \mathcal{C}_{\text{ratio}}(\xi, L) + \delta\mathcal{C}_{\text{hyb}}(\xi, y z_s P^z) \right]$$

with $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The ratio-scheme part, from Eq. (23) of the paper, is

$$
\mathcal{C}_{\text{ratio}}(\xi, L) = 
\begin{cases}
\left[ \frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1} + 1 \right]_{+(1)}^{[1,\infty]} + \frac{3}{2}\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[1,\infty]}, & \xi > 1 \\[6pt]
\left[ \frac{1+\xi^2}{1-\xi}\left( \ln\frac{y^2P_z^2}{\mu^2} + \ln(4\xi(1-\xi)) - 1 \right) + 1 \right]_{+(1)}^{[0,1]} + \frac{3}{2}\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[0,1]}, & 0 < \xi < 1 \\[6pt]
\left[ -\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi} - 1 \right]_{+(1)}^{[-\infty,0]} + \frac{3}{2}\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[-\infty,0]}, & \xi < 0
\end{cases}
$$

where the plus prescription is defined as in the paper:
$$\int_{-\infty}^{\infty} dx\ \left[ f(x) \right]_{+(c)}^{[a,b]} g(x) = \int_a^b dx\ f(x)\left[ g(x) - g(c) \right].$$

The hybrid-scheme correction, from Eq. (24), is

$$
\delta\mathcal{C}_{\text{hyb}}(\xi, y z_s P^z) = \frac{3}{2} \left[ -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_s P^z\big)}{\pi(1-\xi)} \right]_{+(1)}^{[-\infty,\infty]}
$$

with $\mathrm{Si}(x) = \int_0^x \frac{\sin t}{t}\,dt$ and $z_s P^z$ the scheme-change scale.

#### Consistency check

The code implements exactly the coefficient above. The regular (non-plus) part matches: the splitting function $(1+\xi^2)/(1-\xi)$, the log arguments $\ln(\xi/(\xi-1))$ for $\xi>1$, $\ln(-\xi/(1-\xi))$ for $\xi<0$, and $\ln(y^2P_z^2/\mu^2) + \ln(4\xi(1-\xi)) - 1$ for $0<\xi<1$ all agree with the paper's Eq. (23). The constant $+1$ (with sign $\mathrm{sgn}(\xi)$) and the $3/(2|1-\xi|)$ tail are reproduced. The hybrid correction matches Eq. (24) exactly: the prefactor $3/2$, the $1/|1-\xi|$ term, and the $\mathrm{Si}$ term with argument $(1-\xi)|y|z_sP_z$ are all present. The plus prescription is applied with subtraction point $+(1)$ and the domains $[1,\infty]$, $[0,1]$, $[-\infty,0]$ for the ratio part and $[-\infty,\infty]$ for the hybrid part, exactly as in the paper. The code's $\log_scale$ convention differs from the paper's by $\ln 4$, which the code explicitly removes, so the physical log is identical. No discrepancies found.
