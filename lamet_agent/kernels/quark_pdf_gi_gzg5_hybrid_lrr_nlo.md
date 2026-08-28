<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gzg5_hybrid_lrr_nlo; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=a588104a073109ef; paper_used=true -->
$$C_{g_5}^{(1)}(\xi,L,\mu,P_z,z_s) = \frac{\alpha_s C_F}{2\pi} \left[ \frac{1+\xi^2}{1-\xi} \left( L - \ln 4 + \ln(4\xi(1-\xi)) - 1 \right) + 1 + \frac{3}{2}\frac{1}{|1-\xi|} + \frac{3}{2}\left( -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_s P_z)}{\pi(1-\xi)} \right) \right]^{(-\infty,\infty)}_{+(1)}$$

with $\xi = x/y$, $L = \ln(4y^2P_z^2/\mu^2)$, and the plus-prescription defined as in the paper:

$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu$$

The kernel is not fixed-order: it resums the leading Wilson-line renormalon to all orders via

$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0 M_{C_z}\right) \exp\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right)$$

where $M_{C_z}$ is the plus-prescribed matrix of the renormalon shape

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s} p_z(1+\epsilon_m z_s+\epsilon_m^2 z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left( \frac{e^{-\epsilon_m z_s} z_s \sin[(1-\xi)p_z z_s]}{1-\xi} + \frac{e^{-\epsilon_m z_s} p_z}{(\epsilon_m^2+p_z^2(1-\xi)^2)^2} \left[ (\epsilon_m^2 - (1-\xi)^2 p_z^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2(1-\xi)^2 z_s)\cos[(1-\xi)p_z z_s] - (1-\xi)p_z(2\epsilon_m + (1-\xi)^2 p_z^2 z_s + \epsilon_m^2 z_s)\sin[(1-\xi)p_z z_s] \right] \right)$$

with $p_z = |y|P_z$, $z_s = z_s$, $\epsilon_m = 0.005$ GeV, and the scalar coefficients

$$r_0 = N_m \frac{\Gamma(1+b)}{\Gamma(1+b)} \left(\frac{\beta_0}{2\pi}\right)^0 \alpha_s = N_m \alpha_s, \qquad r_{\mathrm{sumPV}} = N_m |z\mu| e^w \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\left[ E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w) \right]$$

with $w = -2\pi/(\alpha_s\beta_0)$, $N_m = 0.575$ for $n_f=3$, and $b, c_1, c_2$ from the QCD beta-function coefficients.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 with the following observations: (i) the fixed-order part matches Eq. (C7) of the companion paper (arXiv:2604.00143) with the $\ln 4$ constant removed from the log, consistent with the paper's notation; (ii) the hybrid correction matches Eq. (C8) with strength $3/2$; (iii) the renormalon shape $C_z$ matches Eq. (17) exactly, including the $\epsilon_m$ regularization; (iv) the PV Borel sum $r_{\mathrm{sumPV}}$ matches Eq. (13) with the real-part prescription; (v) the matrix-exponential assembly matches the paper's LRR prescription. No discrepancies found.

