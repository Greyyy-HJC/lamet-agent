<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gtgpg5_hybrid_nlo; arxiv=2208.08008; equations=Eq. (23); digest=cd0c706eeab638ac; paper_used=true -->
$$C_h\left(\xi,\frac{\mu}{yP_z},\lambda_s\right)=C_r\left(\xi,\frac{\mu}{yP_z}\right)+\delta C\left(\xi,\frac{\mu}{yP_z},\lambda_s\right),$$
with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and $\lambda_s=|y|z_sP_z$. The ratio-scheme piece is
$$C_r\left(\xi,\frac{\mu}{yP_z}\right)=\delta(1-\xi)+\frac{\alpha_s C_F}{2\pi}\begin{cases}
\left[\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1}-\frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi > 1 \\
\left[\frac{2\xi}{1-\xi}\left(L+\ln \xi(1-\xi)\right)+2\right]_{+(1)}^{[0,1]} & 0<\xi<1 \\
\left[-\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1}+\frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi<0 ,
\end{cases}$$
where the plus-prescription is defined as
$$\int_{-\infty}^{\infty} d\xi\, [g(\xi)]_{+(x_0)}^{D}\, f(\xi)=\int_D d\xi\, g(\xi)\,[f(\xi)-f(x_0)],$$
with $x_0=1$ and $D$ the indicated domain. The hybrid correction is
$$\delta C\left(\xi,\frac{\mu}{yP_z},\lambda_s\right)=\frac{\alpha_s C_F}{\pi}\left[-\frac{1}{|1-\xi|}+\frac{2 \mathrm{Si}((1-\xi)\lambda_s)}{\pi(1-\xi)}\right]_{+(1)}^{(-\infty,\infty)}.$$

#### Consistency check
The code’s `C_ratio_gi_perp` reproduces the three branches of Eq. (22) exactly: the splitting $2\xi/(1-\xi)$, the log arguments $\ln[\xi/(\xi-1)]$ on the outer branches and $L+\ln[\xi(1-\xi)]$ inside $[0,1]$, the $+2$ constant only in the middle, and the $2/(1-\xi)$ tail only outside. The hybrid term `_hybrid_gi_delta` matches Eq. (23) with strength $2$ (i.e. prefactor $\alpha_s C_F/\pi$), the correct $R=-1/|1-\xi|+2\mathrm{Si}((1-\xi)\lambda_s)/[\pi(1-\xi)]$, and $\lambda_s=|y|z_sP_z$ as the paper’s $\lambda_s$ built on the parton momentum. The plus-prescription is implemented via column-sum subtraction at $\xi=1$ over the full domain, consistent with the paper’s $[\,\cdot\,]_{+(1)}^{D}$ notation. No discrepancies found.

