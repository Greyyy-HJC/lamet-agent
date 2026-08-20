<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_hybrid_NLO; arxiv=2405.20120; equations=Eq. (4.2), the gamma^t gamma_5 coefficient (V_qq_h below), with the hybrid-scheme Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x)); digest=1a0f4a912a64c998; paper_used=true -->
$$V_{qq,h}^{(1)}(x,y,\mu,P_z) = \frac{\alpha_s(\mu) C_F}{2\pi} \left[ \frac{|x|}{y} \left( \ln\frac{4P_z^2 x^2}{\mu^2} - 1 \right) + \frac{|1-x|}{1-y} \left( \ln\frac{4P_z^2 (1-x)^2}{\mu^2} - 1 \right) + \frac{|x-y|}{y(y-1)} \left( \ln\frac{4P_z^2 (x-y)^2}{\mu^2} - 1 \right) \right.$$
$$\left. + \frac{|x|}{y(y-x)} \left( \ln\frac{4P_z^2 x^2}{\mu^2} - 1 \right) + \frac{|1-x|}{(1-y)(x-y)} \left( \ln\frac{4P_z^2 (1-x)^2}{\mu^2} - 1 \right) + \frac{x+y-2xy}{|x-y|\, y(1-y)} \left( \ln\frac{4P_z^2 (x-y)^2}{\mu^2} - 1 \right) \right]^{[-\infty,\infty]}_+$$

with the plus prescription defined as in the paper:
$$[f(x,y)]^{[a,b]}_+ = f(x,y) - \delta(x-y) \int_a^b f(w,y)\, dw$$

The hybrid-scheme Wilson-line term is:
$$\frac{3\,{\rm Si}(z_s P_z (y-x))}{\pi (y-x)}$$

where ${\rm Si}(x) = \int_0^x \frac{\sin y}{y} dy$ and $\bar{x}=1-x$.

#### Consistency check

The code's `V_qq_h` implements exactly the brace of Eq. (4.2) as transcribed above: the three terms of the first line (with logs $\ln(4P_z^2 x^2/\mu^2)$, $\ln(4P_z^2 (1-x)^2/\mu^2)$, $\ln(4P_z^2 (x-y)^2/\mu^2)$, each minus 1) plus the three terms of the third line (same log structure, different denominators). The prefactor $\alpha_s C_F/(2\pi)$ is applied by the caller. The plus prescription is implemented by making each $y$-column integrate to zero over the full $x$ grid, matching the paper's bracket with $[-\infty,\infty]$ domain. The hybrid Wilson-line term is added as $3\,{\rm Si}(z_s P_z (y-x))/(\pi (y-x))$, exactly as specified. No discrepancies found between code and paper for this coefficient.
