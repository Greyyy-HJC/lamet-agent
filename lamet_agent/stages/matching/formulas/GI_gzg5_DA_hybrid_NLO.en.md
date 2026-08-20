<!-- lamet-agent formula cache; kernel=GI_gzg5_DA_hybrid_NLO; arxiv=2405.20120; equations=Eq. (4.5), the gamma^z gamma_5 coefficient (V_qq_p below), with the hybrid-scheme Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x)); digest=9acb081d84794af9; paper_used=true -->
$$ \mathcal{C}^{\gamma_z\gamma_5}(x,y,\mu,P_z) = \mathcal{C}^{\gamma_t\gamma_5}(x,y,\mu,P_z) + \Delta\mathcal{C}^{\gamma_z\gamma_5}(x,y,\mu,P_z) $$

with the $\gamma_t\gamma_5$ kernel given by Eq. (4.5) of the paper,

$$ \mathcal{C}^{\gamma_t\gamma_5}(x,y,\mu,P_z) = \delta(x-y) + \frac{\alpha_s(\mu)C_F}{2\pi} \left[ \begin{cases} \frac{1+x-y}{y-x}\frac{\bar{x}}{\bar{y}}\ln\frac{(y-x)}{\bar{x}}+\frac{1+y-x}{y-x}\frac{x}{y}\ln\frac{(y-x)}{-x} & x<0\\ \frac{1+y-x}{y-x}\frac{x}{y}\ln\frac{4x(y-x)P_z^2}{\mu^2}+\frac{1+x-y}{y-x}\left(\frac{\bar{x}}{\bar{y}}\ln\frac{y-x}{\bar{x}}-\frac{x}{y}\right) & 0< x< y<1\\ \frac{1+x-y}{x-y}\frac{\bar{x}}{\bar{y}}\ln\frac{4\bar{x}(x-y)P_z^2}{\mu^2}+\frac{1+y-x}{x-y}\left(\frac{x}{y}\ln\frac{x-y}{x}-\frac{\bar{x}}{\bar{y}}\right) & 0<y<x< 1\\ \frac{1+y-x}{x-y}\frac{x}{y}\ln\frac{(x-y)}{x}+\frac{1+x-y}{x-y}\frac{\bar{x}}{\bar{y}}\ln\frac{(x-y)}{-\bar{x}} & 1<x \end{cases} \right. \left. +\frac{3{\rm Si}(z_sP_z(y-x))}{\pi(y-x)}\right]^{[-\infty,\infty]}_+, $$

where $\bar{x}=1-x$, and the plus function defined in a certain range $[a,b]$ is

$$ [f(x,y)]^{[a,b]}_+= f(x,y)-\delta(x-y)\int_{a}^{b} f(w,y) dw, $$

and the sine integral function

$$ {\rm Si}(x)=\int_0^x \frac{\sin y}{y}dy. $$

For the $\gamma_z\gamma_5$ operator, the additional correction term is

$$ \Delta\mathcal{C}^{\gamma_z\gamma_5}=\frac{\alpha_s(\mu)C_F}{\pi}\left[\frac{x}{y}\theta(x)\theta(y-x)+ \begin{matrix} x\leftrightarrow \bar{x} \\ y\leftrightarrow \bar{y} \end{matrix}\right]_+^{[0,1]}. $$

The code implements this coefficient as $V_{qq,p}/2$ (with the $\alpha_s C_F/(2\pi)$ prefactor factored out), where $V_{qq,p} = V_{qq,h} + 2\alpha_s C_F \{|x|/y + |1-x|/(1-y) + |x-y|/((y-1)y)\}$, and the hybrid-scheme Wilson-line term is added separately as $3{\rm Si}(z_sP_z(y-x))/(\pi(y-x))$.

#### Consistency check

The code reproduces Eq. (4.5) for the $\gamma^z\gamma_5$ coefficient ($V_{qq,p}$) with the hybrid-scheme Wilson-line term $3{\rm Si}(z_sP_z(y-x))/(\pi(y-x))$ of arXiv:2405.20120, with the following observations:

- **Regular coefficient**: The code's `V_qq_p` matches the paper's $\mathcal{C}^{\gamma_z\gamma_5}$ (including the $\Delta\mathcal{C}$ term) exactly, with the $\alpha_s C_F/(2\pi)$ prefactor factored out. The piecewise structure for $x<0$, $0<x<y<1$, $0<y<x<1$, and $1<x$ is reproduced.
- **Logarithms**: All log arguments match: $\ln\frac{(y-x)}{\bar{x}}$, $\ln\frac{(y-x)}{-x}$, $\ln\frac{4x(y-x)P_z^2}{\mu^2}$, $\ln\frac{y-x}{\bar{x}}$, $\ln\frac{4\bar{x}(x-y)P_z^2}{\mu^2}$, $\ln\frac{x-y}{x}$, $\ln\frac{(x-y)}{-\bar{x}}$ — all present with correct signs and arguments.
- **Plus prescription**: The code implements the column-sum prescription (each $y$-column integrates to zero), which matches the paper's bracket $[\cdot]^{[-\infty,\infty]}_+$ with the subtraction domain $[-\infty,\infty]$. The paper's definition of the plus function is reproduced verbatim.
- **Delta term**: The LO $\delta(x-y)$ is present via the interpolation stencil in `build_matching_matrix`, consistent with the paper's explicit $\delta(x-y)$.
- **Scheme-specific correction**: The hybrid Wilson-line term $3{\rm Si}(z_sP_z(y-x))/(\pi(y-x))$ is added in `_da_wilson_line("hybrid", ...)`, matching the paper exactly. The code's `_hybrid_gi_delta` uses the same $R = -1/|1-\xi| + 2{\rm Si}((1-\xi)|y|z_sP_z)/(\pi(1-\xi))$ structure with strength $3/2$, which after the $1/|y|$ factor and the $1/2$ from `_da_matrix` gives exactly $3{\rm Si}(z_sP_z(y-x))/(\pi(y-x))$.
- **Discrepancies**: None found. The code and paper agree term by term.
