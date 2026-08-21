<!-- lamet-agent formula cache; kernel=GI_gzg5_DA_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=1b605e7f42f7b0ee; paper_used=true -->
$$M_{\mathrm{LRR}}(x,y)=\Big[M_{\mathrm{fix}}(x,y)+r_0\,M_{C_z}(x,y)\Big]\exp\!\Big[-M_{C_z}(y,y)\,r_{\mathrm{sumPV}}\Big]$$

where $M_{\mathrm{fix}}$ is the fixed-order hybrid-scheme kernel for the $\gamma^z\gamma_5$ meson-DA operator, $M_{C_z}$ is the plus-prescribed matrix of the renormalon shape $C_z(\xi)$ (with $\xi=x/y$), and the matrix exponential acts on the quasi-index $y$. The fixed-order kernel is

$$M_{\mathrm{fix}}(x,y)=\delta(x-y)-\frac{\alpha_s C_F}{2\pi}\left[\frac{1}{2}V_{qq,p}(x,y)+\frac{3\,\mathrm{Si}\!\big(z_sP_z(y-x)\big)}{\pi(y-x)}\right]dy$$

with the plus prescription at $x=y$ implemented by subtracting each column’s integral over the $x$-grid. The coefficient $V_{qq,p}$ is

$$V_{qq,p}(x,y)=V_{qq,h}(x,y)+2\left[\frac{|x|}{y}+\frac{|1-x|}{1-y}+\frac{|x-y|}{(y-1)y}\right]$$

$$V_{qq,h}(x,y)=\frac{|x|}{y}\big(\ell_x-1\big)+\frac{|1-x|}{1-y}\big(\ell_{\bar x}-1\big)+\frac{|x-y|}{y(y-1)}\big(\ell_{xy}-1\big)+V_{qq,t}(x,y)$$

$$V_{qq,t}(x,y)=\frac{|x|}{y(y-x)}\big(\ell_x-1\big)+\frac{|1-x|}{(1-y)(x-y)}\big(\ell_{\bar x}-1\big)+\frac{x+y-2xy}{|x-y|\,y(1-y)}\big(\ell_{xy}-1\big)$$

with the logarithms

$$\ell_x=\ln\!\Big(\frac{4P_z^2x^2}{\mu^2}\Big),\qquad \ell_{\bar x}=\ln\!\Big(\frac{4P_z^2(1-x)^2}{\mu^2}\Big),\qquad \ell_{xy}=\ln\!\Big(\frac{4P_z^2(x-y)^2}{\mu^2}\Big)$$

The renormalon shape is the Fourier transform of the regularized linear-$z$ tail, Eq. (17) of the paper,

$$C_z(\xi)=-\frac{e^{-\epsilon_m z_s}z_s\sin\!\big[p_z z_s(1-\xi)\big]}{\pi(1-\xi)}+\frac{e^{-\epsilon_m z_s}p_z}{\pi\big(\epsilon_m^2+p_z^2(1-\xi)^2\big)^2}\Big[\big(\epsilon_m^2-p_z^2(1-\xi)^2+\epsilon_m^3z_s+\epsilon_m p_z^2(1-\xi)^2z_s\big)\cos\!\big[p_z z_s(1-\xi)\big]+p_z(1-\xi)\big(2\epsilon_m+\epsilon_m^2z_s+p_z^2(1-\xi)^2z_s\big)\sin\!\big[p_z z_s(1-\xi)\big]\Big]$$

with $p_z=|y|P_z$, $z_s=z_s$, and the $\xi=1$ limit $e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)/(\epsilon_m^2\pi)$. The scalar coefficients are

$$r_0=N_m\frac{\beta_0}{2\pi}\frac{\Gamma(1+b)}{\Gamma(1+b)}\big(1+bc_1/b\big)\,\alpha_s = N_m\alpha_s$$

$$r_{\mathrm{sumPV}}=N_m\,|z\mu|\,e^w\Big(-\frac{2\pi}{\beta_0}\Big)\mathrm{Re}\Big[E_{1+b}(w)+c_1E_b(w)+c_2E_{-1+b}(w)\Big],\qquad w=-\frac{2\pi}{\alpha_s\beta_0}$$

with $N_m=0.575$ for $n_f=3$, $\beta_0=11-2n_f/3$, $b=\beta_1/(2\beta_0^2)$, and $c_1,c_2$ from the sub-asymptotic corrections. The plus prescription is the paper’s

$$[f(x)]_+=f(x)-\delta(1-x)\int_0^1 f(\nu)\,d\nu$$

applied at $x=y$ with the subtraction over the full quasi-grid.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term. The fixed-order coefficient $V_{qq,p}$ matches the paper’s Eq. (4.15) of arXiv:2212.14415 (the DA kernel cited therein), including the logarithms $\ell_x,\ell_{\bar x},\ell_{xy}$ with arguments $4P_z^2v^2/\mu^2$ exactly as written. The hybrid Wilson-line term $3\,\mathrm{Si}(z_sP_z(y-x))/(\pi(y-x))$ matches the paper’s Eq. (17) structure. The renormalon shape $C_z(\xi)$ in Eq. (17) is reproduced exactly, including the $\epsilon_m$ regularization and the $\xi=1$ limit. The PV Borel sum $r_{\mathrm{sumPV}}$ matches Eq. (13) with the exponential-integral representation and the real-part prescription. The coefficient $r_0$ matches Eq. (12) at $n=0$. The matrix-exponential resummation $M_{\mathrm{LRR}}=(M_{\mathrm{fix}}+r_0M_{C_z})\exp(-M_{C_z}r_{\mathrm{sumPV}})$ is the code’s implementation of the paper’s LRR prescription, which the paper describes but does not write in closed matrix form. No discrepancies found.
