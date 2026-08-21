<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_hybrid_NLO; arxiv=2405.20120; equations=Eq. (4.2), the gamma^t gamma_5 coefficient (V_qq_h below), with the hybrid-scheme Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x)); digest=f160b816b001ebd4; paper_used=true -->
$$V_{qq,h}^{(1)}(x,y,\mu,P_z) = \frac{\alpha_s C_F}{2\pi} \left[ \frac{|x|}{y} \left(\ln\frac{4P_z^2 x^2}{\mu^2}-1\right) + \frac{|1-x|}{1-y} \left(\ln\frac{4P_z^2 (1-x)^2}{\mu^2}-1\right) + \frac{|x-y|}{y(y-1)} \left(\ln\frac{4P_z^2 (x-y)^2}{\mu^2}-1\right) \right.$$
$$\left. + \frac{|x|}{y(y-x)} \left(\ln\frac{4P_z^2 x^2}{\mu^2}-1\right) + \frac{|1-x|}{(1-y)(x-y)} \left(\ln\frac{4P_z^2 (1-x)^2}{\mu^2}-1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)} \left(\ln\frac{4P_z^2 (x-y)^2}{\mu^2}-1\right) \right]^{[-\infty,\infty]}_+$$

where the plus function is defined as in the paper:
$$[f(x,y)]^{[a,b]}_+ = f(x,y) - \delta(x-y)\int_a^b f(w,y)\,dw,$$
and the hybrid-scheme Wilson-line term is added:
$$+\frac{3\,\mathrm{Si}(z_s P_z (y-x))}{\pi (y-x)},$$
with $\mathrm{Si}(x)=\int_0^x \frac{\sin y}{y}dy$. The logarithm scale is $l = \ln(4P_z^2 v^2/\mu^2)$ for $v = x$, $1-x$, or $x-y$, as in Eq. (4.16) of the paper.

#### Consistency check

The code's `V_qq_h` reproduces the paper's Eq. (4.2) for the $\gamma^t\gamma_5$ coefficient exactly: the three terms of the first brace and the three terms of `V_qq_t` match the paper's Eq. (4.15) verbatim, including all log arguments ($4P_z^2 x^2/\mu^2$, $4P_z^2(1-x)^2/\mu^2$, $4P_z^2(x-y)^2/\mu^2$) and the $-1$ in each. The plus prescription in the code (column-sum to zero over the full quasi grid) matches the paper's bracket with domain $[-\infty,\infty]$. The hybrid Wilson-line term in the code is exactly $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi(y-x))$, as specified. No discrepancies found.
