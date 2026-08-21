<!-- lamet-agent formula cache; kernel=GI_gzg5_DA_ratio_NLO; arxiv=2212.14415; equations=Eq. (4.15) V_qq,p (the gamma^z gamma_5 coefficient), with the ratio-scheme Wilson-line term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto); digest=883b8b3cb4bc0f82; paper_used=true -->
$$V_{qq,p}^{(1)}(x,y,\mu/P_z) = V_{qq,h}^{(1)}(x,y,\mu/P_z) + 2a_s C_F \left\{ \frac{|x|}{y} + \frac{|1-x|}{1-y} + \frac{|x-y|}{(y-1)y} \right\},$$

with the logarithm scale defined as in Eq. (4.16) of the paper,

$$l_v = \ln \frac{4P_z^2 v^2}{\mu^2}, \qquad v = x,\; 1-x,\; x-y,$$

and the plus-prescription at $x=y$ defined by

$$\left[f(x,y)\right]_+ = f(x,y) - \delta(x-y) \int_0^1 f(x',y)\, dx',$$

where the subtraction domain is the full DA support $x'\in[0,1]$.

The explicit regular coefficient is

$$V_{qq,p}^{(1)} = \frac{|x|}{y}(l_x-1) + \frac{|1-x|}{1-y}(l_{1-x}-1) + \frac{|x-y|}{y(y-1)}(l_{x-y}-1) + V_{qq,t}^{(1)} + 2\left\{ \frac{|x|}{y} + \frac{|1-x|}{1-y} + \frac{|x-y|}{(y-1)y} \right\},$$

with

$$V_{qq,t}^{(1)} = \frac{|x|}{y(y-x)}(l_x-1) + \frac{|1-x|}{(1-y)(x-y)}(l_{1-x}-1) + \frac{x+y-2xy}{|x-y|\,y(1-y)}(l_{x-y}-1).$$

The ratio-scheme Wilson-line term is

$$V_{qq}^{\text{rto}}(x,y) = \frac{3}{2|x-y|},$$

which is the $z_s \to \infty$ limit of the hybrid Si term, $3\,\text{Si}(z_s P_z (y-x))/(\pi(y-x))$.

#### Consistency check

The code's `V_qq_p` reproduces Eq. (4.15) exactly: the regular coefficient matches the paper's second line term-for-term, including the absolute values, the denominators, and the log arguments $l_x$, $l_{1-x}$, $l_{x-y}$ as defined in Eq. (4.16). The plus-prescription is implemented as the column-sum subtraction over the full $x\in[0,1]$ domain, matching the paper's bracket definition. The ratio-scheme Wilson-line term `V_qq_rto` is exactly $3/(2|x-y|)$, and the code's `_da_wilson_line("ratio")` adds it to the density, consistent with the paper's Eq. (4.15) and the stated ratio-scheme limit. No discrepancies found.
