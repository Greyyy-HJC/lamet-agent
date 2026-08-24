<!-- lamet-agent formula cache; kernel=GI_gzg5_DA_ratio_NLO; arxiv=2212.14415; equations=Eq. (4.15) V_qq,p (the gamma^z gamma_5 coefficient), with the ratio-scheme Wilson-line term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto); digest=24f91e7e0b84644b; paper_used=true -->
$$V_{qq,p}^{(1)}(x,y) = V_{qq,h}^{(1)}(x,y) + 2a_s C_F \left\{ \frac{|x|}{y} + \frac{|1-x|}{1-y} + \frac{|x-y|}{(y-1)y} \right\},$$

with  
$$V_{qq,h}^{(1)}(x,y) = a_s C_F \left\{ \frac{|x|}{y} (l_x - 1) + \frac{|1-x|}{1-y} (l_{\bar{x}} - 1) + \frac{|x-y|}{y(y-1)} (l_{xy} - 1) \right\} + V_{qq,t}^{(1)}(x,y),$$

and  
$$V_{qq,t}^{(1)}(x,y) = a_s C_F \left\{ \frac{|x|}{y(y-x)} (l_x - 1) + \frac{|1-x|}{(1-y)(x-y)} (l_{\bar{x}} - 1) + \frac{x+y-2xy}{|x-y|\, y(1-y)} (l_{xy} - 1) \right\}.$$

The logarithms are defined as  
$$l_x = \ln \frac{4P_z^2 x^2}{\mu^2}, \qquad l_{\bar{x}} = \ln \frac{4P_z^2 (1-x)^2}{\mu^2}, \qquad l_{xy} = \ln \frac{4P_z^2 (x-y)^2}{\mu^2}.$$

The ratio-scheme Wilson-line term is  
$$V_{qq}^{\mathrm{rto}}(x,y) = \frac{3}{2|x-y|},$$  
which is the $z_s \to \infty$ limit of the hybrid Si term.

The full ratio-scheme kernel is  
$$C^{\mathrm{ratio}}(x,y) = \frac{1}{2} V_{qq,p}^{(1)}(x,y) + \frac{3}{2|x-y|},$$  
with the plus-prescription at $x=y$ defined as  
$$\left[ f(x,y) \right]_+ = f(x,y) - \delta(x-y) \int_0^1 f(x',y) \, dx',$$  
where the subtraction domain is the full DA support $[0,1]$.

#### Consistency check

The code implements exactly the coefficient $V_{qq,p}$ of Eq. (4.15) (second line) with the ratio-scheme Wilson-line term $3/(2|x-y|)$ from $V_{qq,rto}$. Comparing term by term:

- **Regular coefficient**: The code's `V_qq_p` matches the paper's second line exactly: it adds the brace $\{|x|/y + |1-x|/(1-y) + |x-y|/((y-1)y)\}$ to `V_qq_h`, with the same prefactor $2a_s C_F$. The paper's first line (`V_qq_h`) and third line (`V_qq_t`) are reproduced verbatim in the code.
- **Logarithms**: All three logs $l_x$, $l_{\bar{x}}$, $l_{xy}$ appear with the same arguments and prefactors as in Eq. (4.16). The code's `_da_log` uses $\ln(4P_z^2 v^2/\mu^2)$ for $v = x, 1-x, x-y$, matching the paper.
- **Plus-prescription**: The code restores the plus-prescription by making each $y$-column integrate to zero over the $x$ grid, with the subtraction domain being the full quasi grid (which for the DA is $[0,1]$). This matches the paper's bracket definition with the integral over $[0,1]$.
- **Delta term**: There is no explicit $\delta(x-y)$ term in the paper's Eq. (4.15); the code's `build_matching_matrix` includes the LO delta via interpolation, and the plus-prescription handles the singularity. No discrepancy.
- **Scheme-specific correction**: The ratio-scheme Wilson-line term $3/(2|x-y|)$ is exactly `V_qq_rto` in the code, and it is added to the density as specified. The hybrid scheme's Si term is correctly implemented as the finite-$z_s$ version, with the ratio term as its $z_s \to \infty$ limit.

No discrepancies found. The code reproduces Eq. (4.15) $V_{qq,p}$ with the ratio-scheme Wilson-line term $3/(2|x-y|)$ exactly as presented in arXiv:2212.14415.

