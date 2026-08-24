<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=627f4a4b6708e699; paper_used=true -->
$$[\,g(\xi)\,]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_{D} d\nu\, g(\nu)$$

The matching coefficient for the `gt` operator in the `hybrid` scheme, with the leading renormalon resummation (LRR), is

$$C^{\mathrm{LRR}}_{gt}(\xi, \mu, P_z, \tau) = C^{\mathrm{fix}}_{gt}(\xi, \mu, P_z) + r_0\, C_z(\xi) - r_{\mathrm{sumPV}}\, C_z(\xi) \otimes C^{\mathrm{fix}}_{gt}(\xi, \mu, P_z) + \mathcal{O}(\alpha_s^2)$$

where the fixed-order part is

$$C^{\mathrm{fix}}_{gt}(\xi, \mu, P_z) = \frac{\alpha_s C_F}{2\pi} \left[\, \frac{1+\xi^2}{1-\xi} \left( L + \ln(4\xi(1-\xi)) - 1 \right) + 1 + \frac{3}{2}\left( -\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_s P_z)}{\pi(1-\xi)} \right) \,\right]^{(-\infty,\infty)}_{+(1)}$$

with $L = \ln(4y^2P_z^2/\mu^2)$, $\xi = x/y$, and the plus-prescription defined above. The renormalon shape is

$$C_z(\xi) = N_m \mu \left\{ \frac{e^{-\epsilon_m z_s} p_z (1+\epsilon_m z_s + \epsilon_m^2 z_s^2)}{\epsilon_m^2 \pi} + \frac{1}{\pi}\left( \frac{e^{-\epsilon_m z_s} z_s \sin[\bar{\xi} z_s p_z]}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s} p_z}{(\epsilon_m^2 + p_z^2 \bar{\xi}^2)^2} \left[ (\epsilon_m^2 - \bar{\xi}^2 p_z^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2 \bar{\xi}^2 z_s) \cos[\bar{\xi} z_s p_z] - \bar{\xi} p_z (2\epsilon_m + \bar{\xi}^2 p_z^2 z_s + \epsilon_m^2 z_s) \sin[\bar{\xi} z_s p_z] \right] \right) \right\}_+$$

with $\bar{\xi} = 1-\xi$, $p_z = |y|P_z$, $z_s = z_{\mathrm{spz}}/P_z$, and $\epsilon_m = 0.005$ GeV. The scalar coefficients are $r_0 = N_m (\beta_0/2\pi)^0 \Gamma(1+b)/\Gamma(1+b) \cdot \alpha_s = N_m \alpha_s$ and $r_{\mathrm{sumPV}} = N_m |z\mu| e^w (-2\pi/\beta_0) \mathrm{Re}[E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w)]$ with $w = -2\pi/(\alpha_s \beta_0)$, $b = \beta_1/(2\beta_0^2)$, and $N_m = 0.575$ for $n_f = 3$. The full kernel is assembled as $M_{\mathrm{LRR}} = (M_{\mathrm{fix}} + r_0 M_{C_z}) \exp(-M_{C_z} r_{\mathrm{sumPV}})$, where $M_{C_z}$ is the plus-prescribed matrix of $C_z(\xi)/|y|$ and the matrix exponential acts on the quasi-index.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term: the splitting function $(1+\xi^2)/(1-\xi)$, the logarithms with arguments $4y^2P_z^2/\mu^2$ and $4\xi(1-\xi)$, the constant $+1$, the hybrid Si-term with strength $3/2$, the renormalon shape $C_z$ with the $\epsilon_m$ regulator, the PV Borel sum $r_{\mathrm{sumPV}}$ via exponential integrals, and the matrix-exponential resummation. The plus-prescription matches the paper's $[g]^{(-\infty,\infty)}_{+(1)}$ with the column-sum-to-zero implementation. One minor discrepancy: the code's fixed-order log uses $\ln(4y^2P_z^2/\mu^2)$ while the paper's Eq. (23) writes $\ln(y^2P_z^2/\mu^2)$; the code removes the $\ln 4$ constant explicitly, so the physical content is identical. No other discrepancies found.

