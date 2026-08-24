<!-- lamet-agent formula cache; kernel=GI_gzg5_DA_hybrid_NLO; arxiv=2405.20120; equations=Eq. (4.5), the gamma^z gamma_5 coefficient (V_qq_p below), with the hybrid-scheme Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x)); digest=5654810ed7cd3fc1; paper_used=true -->
The matching coefficient for the `gzg5` operator in the hybrid scheme is given by Eq. (4.5) of arXiv:2405.20120, with the Wilson-line term $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi (y-x))$. The kernel is written as a plus-distribution in $x$ at fixed $y$, with the plus prescription defined over $[-\infty,\infty]$:

$$
\mathcal{C}^{\gamma_z\gamma_5}(x,y,\mu,P_z) = \delta(x-y) + \frac{\alpha_s(\mu) C_F}{2\pi} \left[ V_{qq,p}(x,y) + \frac{3\,\mathrm{Si}(z_s P_z (y-x))}{\pi (y-x)} \right]^{[-\infty,\infty]}_+,
$$

where the plus function is defined as in the paper:

$$
[f(x,y)]^{[a,b]}_+ = f(x,y) - \delta(x-y) \int_a^b f(w,y)\, dw,
$$

and $\mathrm{Si}(x) = \int_0^x \frac{\sin y}{y} dy$.

The regular coefficient $V_{qq,p}(x,y)$ is the sum of the $\gamma_t\gamma_5$ coefficient $V_{qq,h}$ plus the $\Delta\mathcal{C}^{\gamma_z\gamma_5}$ correction. Explicitly, with $\bar{x}=1-x$, $\bar{y}=1-y$, and the logarithms defined as $l_x = \ln(4P_z^2 x^2/\mu^2)$, $l_{\bar{x}} = \ln(4P_z^2 \bar{x}^2/\mu^2)$, $l_{x-y} = \ln(4P_z^2 (x-y)^2/\mu^2)$:

$$
V_{qq,p}(x,y) = V_{qq,h}(x,y) + 2\left[ \frac{|x|}{y} + \frac{|1-x|}{1-y} + \frac{|x-y|}{(y-1)y} \right],
$$

with

$$
V_{qq,h}(x,y) = \frac{|x|}{y}(l_x - 1) + \frac{|1-x|}{1-y}(l_{\bar{x}} - 1) + \frac{|x-y|}{y(y-1)}(l_{x-y} - 1) + V_{qq,t}(x,y),
$$

and

$$
V_{qq,t}(x,y) = \frac{|x|}{y(y-x)}(l_x - 1) + \frac{|1-x|}{(1-y)(x-y)}(l_{\bar{x}} - 1) + \frac{x+y-2xy}{|x-y|\,y(1-y)}(l_{x-y} - 1).
$$

The scheme-specific correction is the hybrid Wilson-line term $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi (y-x))$, which replaces the ratio-scheme term $3/(2|x-y|)$; the two coincide in the limit $z_s P_z \to \infty$.

#### Consistency check

The code implements exactly the coefficient above. Comparing term by term with Eq. (4.5) of the paper:

- **Regular coefficient**: The code's `V_qq_p` matches the paper's $\gamma^z\gamma_5$ coefficient, including the $\Delta\mathcal{C}$ correction. The brace in `V_qq_p` vanishes identically outside $0<x<1$, consistent with the paper's statement that the two operators differ only in the physical region.
- **Logarithms**: All log arguments are $4P_z^2 v^2/\mu^2$ with $v = x$, $1-x$, or $x-y$, matching the paper's $l_x$, $l_{\bar{x}}$, $l_{xy}$ definitions. The squares ensure well-defined logs for negative arguments, as in the paper.
- **Plus prescription**: The code uses the paper's exact bracket $[\,\cdot\,]^{[-\infty,\infty]}_+$ with the subtraction domain $[-\infty,\infty]$, matching the definition given in the text.
- **Delta term**: The LO $\delta(x-y)$ is present, and the plus prescription restores the $x=y$ singularity, consistent with the paper.
- **Scheme correction**: The hybrid Wilson-line term is exactly $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi (y-x))$, as specified. The code's `_hybrid_gi_delta` with strength $3/2$ reproduces this term, and the ratio-scheme term $3/(2|x-y|)$ is correctly subtracted in the hybrid case.

No discrepancies were found between the code and the paper for this coefficient.

