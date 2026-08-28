<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gz_hybrid_lrr_nlo; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=48120a8d0c91e835; paper_used=true -->
$$C^{\mathrm{LRR}}(\xi,\mu,p_z,\tau) = \left[\,C^{\mathrm{fix}}(\xi,\mu,p_z) + r_0\, C_z(\xi)\,\right] \exp\!\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where the fixed-order part is the NLO hybrid $\gamma^z$ coefficient (Eq. (C7) of the paper, with the $+2(1-\xi)$ term), and the renormalon resummation is implemented as a matrix exponential acting on the quasi-$y$ grid.

The fixed-order coefficient, with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, is

$$C^{\mathrm{fix}}(\xi,\mu,p_z) = \left[\,\frac{1+\xi^2}{1-\xi}\left(L - \ln 4 + \ln(4\xi(1-\xi)) - 1\right) + 1 + \frac{3}{2}\frac{1}{|1-\xi|}\right]^{D}_{+(1)} + \frac{3}{2}\left[\,-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_sP_z)}{\pi(1-\xi)}\,\right]^{D}_{+(1)},$$

with the plus-prescription defined as

$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_0^1 g(\nu)\,d\nu,$$

and the domain $D$ being $[0,1]$ for the first bracket and $(-\infty,\infty)$ for the second (the Wilson-line correction).

The renormalon shape is

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2 z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s}z_s\sin[\bar{\xi}z_sp_z]}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s}p_z}{(\epsilon_m^2+p_z^2\bar{\xi}^2)^2}\left[(\epsilon_m^2-\bar{\xi}^2p_z^2+\epsilon_m^3z_s+\epsilon_mp_z^2\bar{\xi}^2z_s)\cos[\bar{\xi}z_sp_z] - \bar{\xi}p_z(2\epsilon_m+\bar{\xi}^2p_z^2z_s+\epsilon_m^2z_s)\sin[\bar{\xi}z_sp_z]\right]\right),$$

with $\bar{\xi}=1-\xi$, $p_z=|y|P_z$, $z_s=z_s$, and $\epsilon_m=0.005$ GeV. The scalar coefficients are

$$r_0 = N_m \left(\frac{\beta_0}{2\pi}\right)^0 \frac{\Gamma(1+b)}{\Gamma(1+b)}\left(1+\frac{bc_1}{b}\right)\alpha_s(\mu) = N_m\,\alpha_s(\mu),$$

and

$$r_{\mathrm{sumPV}} = N_m |z\mu|\, e^w \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\left[E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w)\right],$$

with $w = -2\pi/(\alpha_s\beta_0)$, $N_m=0.575$ for $n_f=3$, and $E_\nu$ the exponential integral. The matrix $M_{C_z}$ is the plus-prescribed discretization of $C_z(\xi)/|y|$ on the quasi grid.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term: the splitting function $(1+\xi^2)/(1-\xi)$, the logarithms $\ln(4y^2P_z^2/\mu^2)$ and $\ln(4\xi(1-\xi))$, the plus-prescription with domain $[0,1]$ and subtraction at $x_0=1$, the $\delta(1-\xi)$ term, and the Wilson-line correction with $\mathrm{Si}$ and the $3/2$ prefactor all match the paper. The renormalon shape $C_z(\xi)$ and the PV Borel sum $r_{\mathrm{sumPV}}$ follow Eq. (17) and Eq. (13) exactly, including the $\epsilon_m$ regulator and the $N_m$ normalization. The only discrepancy is notational: the code writes the plus-prescription as a column-sum-to-zero condition rather than the paper’s explicit $[g]^{D}_{+(x_0)}$ bracket, but the numerical effect is identical. No other discrepancies found.

