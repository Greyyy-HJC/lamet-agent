<!-- lamet-agent formula cache; kernel=GI_gtgpg5_quark_PDF_hybrid_NLO; arxiv=2208.08008; equations=Eq. (23); digest=3b0b225c48f74a2a; paper_used=true -->
$$C_h\left(\xi,\frac{\mu}{yP_z},\lambda_s\right)=
C_r\left(\xi,\frac{\mu}{yP_z}\right)+\delta C\left(\xi,\frac{\mu}{yP_z},\lambda_s\right),$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and $\lambda_s=|y|z_sP_z$. The ratio-scheme piece is

$$C_r(\xi,L)=
\begin{cases}
\left[\dfrac{2\xi}{1-\xi}\ln\dfrac{\xi}{\xi-1}-\dfrac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi>1 \\[2ex]
\left[\dfrac{2\xi}{1-\xi}\Bigl(L+\ln[\xi(1-\xi)]\Bigr)+2\right]_{+(1)}^{[0,1]} & 0<\xi<1 \\[2ex]
\left[-\dfrac{2\xi}{1-\xi}\ln\dfrac{\xi}{\xi-1}+\dfrac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi<0,
\end{cases}$$

where the plus-prescription is defined as in the paper:

$$\int_0^1 dx\,[g(x)]_{+(x_0)}^{D}\,f(x)=\int_D dx\,g(x)\bigl[f(x)-f(x_0)\bigr],$$

with $x_0=1$ and $D$ the indicated domain. The hybrid correction is

$$\delta C(\xi,\lambda_s)=\frac{\alpha_s C_F}{\pi}\left[-\frac{1}{|1-\xi|}+\frac{2\,\mathrm{Si}\bigl((1-\xi)\lambda_s\bigr)}{\pi(1-\xi)}\right]_{+(1)}^{(-\infty,\infty)}.$$

#### Consistency check

The code implements exactly the three-branch structure of Eq. (22) and the $\delta C$ of Eq. (23), with the same logs, the same $+2$ constant in $0<\xi<1$, the same $\mathrm{Si}$ argument $(1-\xi)\lambda_s$ with $\lambda_s=|y|z_sP_z$, and the same prefactor $\alpha_s C_F/\pi$ for the hybrid term. The plus-prescription is enforced numerically by column-sum subtraction, matching the paper’s $[\,\cdot\,]_{+(1)}^{D}$ convention. No discrepancies were found.
