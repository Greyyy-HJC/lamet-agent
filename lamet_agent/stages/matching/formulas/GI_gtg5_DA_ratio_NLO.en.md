<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_ratio_NLO; arxiv=2212.14415; equations=Eq. (4.15) V_qq,h (the gamma^t gamma_5 coefficient), with the ratio-scheme Wilson-line term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto); digest=8026c977e30dceea; paper_used=true -->
$$V_{qq,h}^{(1)}(x,y) = \frac{|x|}{y}\left(l_x-1\right) + \frac{|1-x|}{1-y}\left(l_{\bar{x}}-1\right) + \frac{|x-y|}{y(y-1)}\left(l_{xy}-1\right) + V_{qq,t}^{(1)}(x,y)$$

with

$$V_{qq,t}^{(1)}(x,y) = \frac{|x|}{y(y-x)}\left(l_x-1\right) + \frac{|1-x|}{(1-y)(x-y)}\left(l_{\bar{x}}-1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)}\left(l_{xy}-1\right)$$

and the logarithms defined as

$$l_x = \ln\frac{4P_z^2 x^2}{\mu^2}, \qquad l_{\bar{x}} = \ln\frac{4P_z^2 (1-x)^2}{\mu^2}, \qquad l_{xy} = \ln\frac{4P_z^2 (x-y)^2}{\mu^2}.$$

The full coefficient is

$$V_{qq,h}(x,y) = a_s C_F \left[ V_{qq,h}^{(1)}(x,y) \right]_+$$

where the plus-prescription is defined exactly as in the paper:

$$\left[\frac{f(\alpha)}{\alpha}\right]_+ = \frac{f(\alpha)}{\alpha} - \delta(\alpha)\int_0^1 \frac{f(\alpha')}{\alpha'}\,d\alpha'$$

with the subtraction domain being the full support of the DA, $y \in [0,1]$.

The ratio-scheme Wilson-line term is

$$V_{qq}^{\text{rto}}(x,y) = \frac{3}{2|x-y|}$$

which is the $z_s \to \infty$ limit of the hybrid scheme's sine-integral term

$$\frac{3\,\text{Si}(z_s P_z (y-x))}{\pi (y-x)}.$$

The full ratio-scheme kernel is

$$C^{\text{ratio}}(x,y) = \frac{1}{2}V_{qq,h}(x,y) + \frac{3}{2|x-y|}.$$

#### Consistency check

The code's `V_qq_h` implements exactly the first line of Eq. (4.15) as transcribed above: the three terms with $|x|/y$, $|1-x|/(1-y)$, and $|x-y|/(y(y-1))$, each multiplied by the corresponding log minus one, plus the `V_qq_t` term of the third line. The logarithms in `_da_log` match the paper's Eq. (4.16) form $\ln(4P_z^2 v^2/\mu^2)$ for $v = x$, $1-x$, and $x-y$. The plus-prescription is implemented by the column-sum subtraction in `build_matching_matrix`, which makes each $y$-column integrate to zero over the full $x$ grid, consistent with the paper's definition. The ratio-scheme Wilson-line term `V_qq_rto` is exactly $3/(2|x-y|)$, and the hybrid version `_hybrid_gi_delta` with strength $3/2$ reproduces the paper's Eq. (C8) form. The code's `_da_matrix` adds this term with a factor of $1/2$ on the coefficient, matching the paper's convention. No discrepancies found: the code reproduces Eq. (4.15) $V_{qq,h}$ with the ratio-scheme Wilson-line term $3/(2|x-y|)$ exactly as presented in arXiv:2212.14415.
