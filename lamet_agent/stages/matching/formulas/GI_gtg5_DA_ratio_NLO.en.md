<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_ratio_NLO; arxiv=2212.14415; equations=Eq. (4.15) V_qq,h (the gamma^t gamma_5 coefficient), with the ratio-scheme Wilson-line term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto); digest=6ecd3b66ac995ef5; paper_used=true -->
$$V_{qq,h}^{(1)}(x,y) = \frac{\alpha_s C_F}{2\pi} \left\{ \frac{|x|}{y} \left(\ln\frac{4P_z^2 x^2}{\mu^2}-1\right) + \frac{|1-x|}{1-y} \left(\ln\frac{4P_z^2 (1-x)^2}{\mu^2}-1\right) + \frac{|x-y|}{y(y-1)} \left(\ln\frac{4P_z^2 (x-y)^2}{\mu^2}-1\right) \right\}_+ + V_{qq,t}^{(1)}(x,y)$$

with the transversity piece

$$V_{qq,t}^{(1)}(x,y) = \frac{\alpha_s C_F}{2\pi} \left\{ \frac{|x|}{y(y-x)} \left(\ln\frac{4P_z^2 x^2}{\mu^2}-1\right) + \frac{|1-x|}{(1-y)(x-y)} \left(\ln\frac{4P_z^2 (1-x)^2}{\mu^2}-1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)} \left(\ln\frac{4P_z^2 (x-y)^2}{\mu^2}-1\right) \right\}$$

where the plus-prescription is defined as in the paper:

$$\left[\frac{f(\alpha)}{\alpha}\right]_+ = \frac{f(\alpha)}{\alpha} - \delta(\alpha)\int_0^1 \frac{f(\alpha')}{\alpha'}\,d\alpha'$$

with the subtraction domain being the full support $y\in[0,1]$ of the DA.

The ratio-scheme Wilson-line term is

$$V_{qq}^{\mathrm{rto}}(x,y) = \frac{3}{2|x-y|}$$

which is the $z_s\to\infty$ limit of the hybrid Si term $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi(y-x))$. The full ratio-scheme kernel is

$$C^{\mathrm{ratio}}(x,y) = \frac{1}{2}V_{qq,h}^{(1)}(x,y) + \frac{3}{2|x-y|}$$

with the $1/2$ from the code's `0.5 * coefficient` and the Wilson-line term added separately.

#### Consistency check

The code's `V_qq_h` implements exactly the first line of Eq. (4.15) as transcribed: the three log terms with arguments $4P_z^2 x^2/\mu^2$, $4P_z^2(1-x)^2/\mu^2$, and $4P_z^2(x-y)^2/\mu^2$, each with the $-1$ shift, and the prefactors $|x|/y$, $|1-x|/(1-y)$, and $|x-y|/(y(y-1))$ match the paper verbatim. The `V_qq_t` term is added on as the third line of Eq. (4.15), with the same log structure and the prefactors $|x|/(y(y-x))$, $|1-x|/((1-y)(x-y))$, and $(x+y-2xy)/(|x-y|y(1-y))$ — all matching. The plus-prescription is implemented by the column-sum subtraction over the full $y\in[0,1]$ grid, consistent with the paper's definition. The ratio-scheme Wilson-line term `V_qq_rto` is exactly $3/(2|x-y|)$, and the code adds it with the correct sign and prefactor. No discrepancies found: the code reproduces Eq. (4.15) $V_{qq,h}$ with the ratio-scheme term $3/(2|x-y|)$ of arXiv:2212.14415 exactly as written.
