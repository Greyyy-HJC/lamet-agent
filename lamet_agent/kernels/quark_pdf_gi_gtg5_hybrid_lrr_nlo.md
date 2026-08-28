<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gtg5_hybrid_lrr_nlo; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=ef7e5216c3ffc113; paper_used=true -->
$$C_{\rm gtg5}^{(\rm hybrid,LRR)}(\xi,L,y,P_z,\mu,z_s) = \left[\,C_{\rm gtg5}^{(\rm hybrid,NLO)}(\xi,L,y,P_z,\mu,z_s) + r_0\, C_z(\xi,y,P_z,z_s)\,\right]^{D}_{+(1)} \exp\!\left(-\,r_{\rm sumPV}\, C_z(\xi,y,P_z,z_s)\right)$$

with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription defined as in the paper:
$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu,$$
with domain $D=(-\infty,\infty)$ and subtraction point $x_0=1$.

The fixed-order part is the NLO hybrid coefficient (Eq. (24) of the companion paper, same as $\gamma^t$):
$$C_{\rm gtg5}^{(\rm hybrid,NLO)}(\xi,L,y,P_z,\mu,z_s) = C_{\rm ratio}(\xi,L) + \frac{3}{2}\left[-\frac{1}{|1-\xi|} + \frac{2\,{\rm Si}\!\left((1-\xi)|y|z_sP_z\right)}{\pi(1-\xi)}\right],$$
where
$$C_{\rm ratio}(\xi,L) = \begin{cases}
\frac{1+\xi^2}{1-\xi}\left(L - \ln 4 + \ln(4\xi(1-\xi)) - 1\right) + 1 + \frac{3}{2|1-\xi|}, & 0<\xi<1,\\[4pt]
{\rm sgn}(\xi)\left[\frac{1+\xi^2}{1-\xi}\ln\left|\frac{\xi}{\xi-1}\right| + 1\right] + \frac{3}{2|1-\xi|}, & \xi<0 \text{ or } \xi>1.
\end{cases}$$

The renormalon resummation uses the shape (Eq. (17), with $\epsilon_m$ the long-distance regulator):
$$C_z(\xi,y,P_z,z_s) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2 z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s}z_s\sin[\bar{\xi}z_sp_z]}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s}p_z}{(\epsilon_m^2+p_z^2\bar{\xi}^2)^2}\left[(\epsilon_m^2-\bar{\xi}^2p_z^2+\epsilon_m^3z_s+\epsilon_mp_z^2\bar{\xi}^2z_s)\cos[\bar{\xi}z_sp_z] - \bar{\xi}p_z(2\epsilon_m+\bar{\xi}^2p_z^2z_s+\epsilon_m^2z_s)\sin[\bar{\xi}z_sp_z]\right]\right),$$
with $p_z=|y|P_z$, $\bar{\xi}=1-\xi$, $z_s=z_s$, and the $\xi=1$ limit taken as $e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)/(\epsilon_m^2\pi)$.

The scalar coefficients are $r_0 = N_m\,(\beta_0/2\pi)^0\,\Gamma(1+b)/\Gamma(1+b)\,(1+bc_1/b)\,\alpha_s = N_m\,\alpha_s$ (with $N_m=0.575$ for $n_f=3$) and
$$r_{\rm sumPV} = N_m\,|z\mu|\,e^w\left(-\frac{2\pi}{\beta_0}\right){\rm Re}\left[E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w)\right],\quad w = -\frac{2\pi}{\alpha_s\beta_0},$$
where $E_\nu$ is the exponential integral, $\beta_0=9$, $b=\beta_1/(2\beta_0^2)$, $c_1=(\beta_1^2-\beta_0\beta_2)/(4b\beta_0^3)$, $c_2$ from the paper's Eq. (12) tail, and $\alpha_s=\alpha_s(\mu)$ at two-loop running.

The matrix implementation is $M_{\rm LRR} = (M_{\rm fix} + r_0 M_{C_z})\exp(-M_{C_z}r_{\rm sumPV})$, where $M_{C_z}$ is the plus-prescribed discretization of $C_z(\xi)/|y|$ with the column-sum subtraction, and the exponential is a matrix exponential acting on the quasi-index.

#### Consistency check
The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term: the splitting function $(1+\xi^2)/(1-\xi)$, the logarithms $\ln(y^2P_z^2/\mu^2)$ and $\ln(4\xi(1-\xi))$ (with the $\ln 4$ constant removed as documented), the $\pm1$ constants, the $3/(2|1-\xi|)$ tail, the Si-term with strength $3/2$, the plus-prescription with domain $(-\infty,\infty)$ and subtraction at $x_0=1$, the renormalon shape $C_z$ of Eq. (17) with $\epsilon_m$ regulator, and the PV Borel sum $r_{\rm sumPV}$ of Eq. (13). The only discrepancy is notational: the paper writes the plus-prescription as $[\,\cdot\,]^{D}_{+(x_0)}$ with $D$ as a superscript and $x_0$ as a subscript, which the code implements numerically via column-sum-to-zero; the code's $r_0$ uses $N_m=0.575$ (paper's $n_f=3$ value) and the paper's Eq. (12) with $n=0$, which matches. No other discrepancies found.

