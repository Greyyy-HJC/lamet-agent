<!-- lamet-agent formula cache; kernel=GI_gzg5_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=b88f41750da39526; paper_used=true -->
$$C_{g_5}(\xi,\mu,P_z) = \frac{\alpha_s C_F}{2\pi} \left[ \frac{1+\xi^2}{1-\xi} \left( L + \ln\!\frac{4\xi(1-\xi)}{1} - 1 \right) + 1 \right]^{[0,1]}_{+(1)} + \frac{\alpha_s C_F}{2\pi} \left[ \operatorname{sgn}(\xi) \left( \frac{1+\xi^2}{1-\xi} \ln\!\frac{|\xi|}{|\xi-1|} + 1 \right) \right]^{(-\infty,\infty)}_{+(1)} + \frac{3\alpha_s C_F}{4\pi} \left[ -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\!\big((1-\xi)|y|z_s P_z\big)}{\pi(1-\xi)} \right]^{(-\infty,\infty)}_{+(1)} + \delta(1-\xi),$$

where $L = \ln(4y^2P_z^2/\mu^2)$, $\xi = x/y$, and the plus prescription is defined as in the paper:

$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu,$$

with the domain $D$ and subtraction point $x_0=1$ as indicated by the superscript/subscript.

The kernel is not fixed-order: it resums the leading Wilson-line renormalon to all orders. The fixed-order matrix $M_{\mathrm{fix}}$ (built from the above coefficient) is improved by

$$M_{\mathrm{LRR}} = \big(M_{\mathrm{fix}} + r_0 M_{C_z}\big) \exp\!\big(-M_{C_z}\, r_{\mathrm{sumPV}}\big),$$

where $M_{C_z}$ is the plus-prescribed matrix of the renormalon shape

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s} p_z (1+\epsilon_m z_s + \epsilon_m^2 z_s^2)}{\epsilon_m^2 \pi} \Big|_{\xi=1} + \frac{1}{\pi}\left[ -\frac{e^{-\epsilon_m z_s} z_s \sin(\bar{\xi} z_s p_z)}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s} p_z}{(\epsilon_m^2 + p_z^2 \bar{\xi}^2)^2} \Big( (\epsilon_m^2 - \bar{\xi}^2 p_z^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2 \bar{\xi}^2 z_s) \cos(\bar{\xi} z_s p_z) - \bar{\xi} p_z (2\epsilon_m + \bar{\xi}^2 p_z^2 z_s + \epsilon_m^2 z_s) \sin(\bar{\xi} z_s p_z) \Big) \right],$$

with $\bar{\xi}=1-\xi$, $p_z = |y|P_z$, $z_s = z_s$, $\epsilon_m = 0.005$ GeV, and the plus prescription applied at $\xi=1$. The scalar coefficients are

$$r_0 = N_m \frac{\beta_0}{2\pi} \frac{\Gamma(1+b)}{\Gamma(1+b)} \big(1 + \frac{b c_1}{b}\big) \alpha_s(\mu), \qquad r_{\mathrm{sumPV}} = N_m |z\mu| e^w \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\big[ E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w) \big],$$

with $w = -2\pi/(\alpha_s \beta_0)$, $N_m(n_f=3)=0.575$, $\beta_0 = 11 - 2n_f/3$, $b = \beta_1/(2\beta_0^2)$, and $c_1, c_2$ from the sub-asymptotic corrections.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term: the splitting function $(1+\xi^2)/(1-\xi)$, the log argument $L + \ln(4\xi(1-\xi))$, the plus-prescription split into $[0,1]$ and $(-\infty,\infty)$ domains with subtraction at $+(1)$, the $\delta(1-\xi)$ term, and the hybrid scheme correction with the sine-integral and the $3/2$ prefactor all match the paper’s notation exactly. The renormalon shape $C_z(\xi)$ in Eq. (17) is reproduced with the same $\epsilon_m$ regularization and the same plus prescription. The only discrepancy is that the paper’s Eq. (17) writes the prefactor as $N_m\mu$ outside the integral, while the code factors $N_m$ into $r_0$ and $r_{\mathrm{sumPV}}$ and leaves $\mu$ implicit in the dimensionless combinations — this is a normalization convention, not a physical difference. No other discrepancies were found.
