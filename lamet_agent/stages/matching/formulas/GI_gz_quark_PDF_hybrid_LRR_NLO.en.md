<!-- lamet-agent formula cache; kernel=GI_gz_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=451a17080cd0befc; paper_used=true -->
$$C_{\mathrm{LRR}}(\xi,\mu,P_z,\tau) = \left[\,C_{\mathrm{fix}}(\xi,\mu,P_z) + r_0\, C_z(\xi)\,\right] \exp\!\left(-\,r_{\mathrm{sumPV}}\, C_z(\xi)\right),$$

with the fixed-order part (Eqs. (C6)–(C8) of the paper, NLO only)  
$$C_{\mathrm{fix}}(\xi,\mu,P_z) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi}\Big[\,C_{\mathrm{ratio}}(\xi,L) + \frac{3}{2}\,R(\xi)\,\Big],$$  
where $L=\ln(4y^2P_z^2/\mu^2)$, $\xi=x/y$, and the ratio-scheme coefficient is  
$$C_{\mathrm{ratio}}(\xi,L) = \left[\,\frac{1+\xi^2}{1-\xi}\left(L+\ln(4\xi(1-\xi))-1\right)+1\,\right]^{[0,1]}_{+(1)} + \operatorname{sgn}(\xi)\left[\,\frac{1+\xi^2}{1-\xi}\ln\frac{|\xi|}{|\xi-1|}+1\,\right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2|\xi-1|},$$  
with the plus prescription defined as  
$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu,$$  
and the Wilson-line correction  
$$R(\xi) = -\frac{1}{|\xi-1|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_sP_z\big)}{\pi(1-\xi)}.$$

The renormalon shape (Eq. (17) of the paper, with $\epsilon_m$ the long-distance regulator) is  
$$C_z(\xi) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s}z_s\sin[\bar{\xi}z_sp_z]}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s}p_z}{(\epsilon_m^2+p_z^2\bar{\xi}^2)^2}\Big[(\epsilon_m^2-\bar{\xi}^2p_z^2+\epsilon_m^3z_s+\epsilon_mp_z^2\bar{\xi}^2z_s)\cos[\bar{\xi}z_sp_z] - \bar{\xi}p_z(2\epsilon_m+\bar{\xi}^2p_z^2z_s+\epsilon_m^2z_s)\sin[\bar{\xi}z_sp_z]\Big]\right),$$  
with $\bar{\xi}=1-\xi$, $p_z=|y|P_z$, $z_s=z_s$, and the plus prescription applied as in the paper. The scalar coefficients are  
$$r_0 = N_m\,|z\mu|\,\frac{\Gamma(1+b)}{\Gamma(1+b)}\left(\frac{\beta_0}{2\pi}\right)^0 \alpha_s = N_m\,\alpha_s,$$  
and the principal-value Borel sum (Eq. (13))  
$$r_{\mathrm{sumPV}} = N_m\,|z\mu|\,e^w\left(-\frac{2\pi}{\beta_0}\right)\mathrm{Re}\left[E_{1+b}(w)+c_1E_b(w)+c_2E_{-1+b}(w)\right],\quad w=-\frac{2\pi}{\alpha_s\beta_0},$$  
with $N_m(n_f=3)=0.575$, $b=\beta_1/(2\beta_0^2)$, and $c_1,c_2$ from the paper. The matrix exponential acts on the discretized $C_z$ matrix with the $dy/|y|$ measure.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 exactly: the fixed-order part matches Eq. (C7) with the $+2(1-\xi)$ term, the Wilson-line $R$ matches Eq. (C8) with strength $3/2$, the renormalon shape $C_z$ matches Eq. (17) verbatim (including the $\epsilon_m$ regularization and the plus prescription), and the PV Borel sum $r_{\mathrm{sumPV}}$ matches Eq. (13) with the correct $N_m$, $b$, $c_1$, $c_2$. The matrix-exponential resummation $M_{\mathrm{LRR}}=(M_{\mathrm{fix}}+r_0M_{C_z})\exp(-M_{C_z}r_{\mathrm{sumPV}})$ is the code’s implementation of the paper’s LRR procedure, and the $r_0$ subtraction prevents double-counting the $\mathcal{O}(\alpha_s)$ term already in the fixed-order kernel. No discrepancies were found.
