<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=16da676544086e67; paper_used=true -->
$$V_{qq,h}^{(1)}(x,y,\mu,P_z) = \frac{\alpha_s C_F}{2\pi} \left\{ \frac{|x|}{y}\left(l_x-1\right) + \frac{|1-x|}{1-y}\left(l_{1-x}-1\right) + \frac{|x-y|}{y(y-1)}\left(l_{x-y}-1\right) + V_{qq,t}^{(1)}(x,y,\mu,P_z) \right\}$$

with the logarithms defined as in Eq. (4.16) of the paper:

$$l_v = \ln\left(\frac{4P_z^2 v^2}{\mu^2}\right), \qquad v = x,\; 1-x,\; x-y,$$

and the transverse piece

$$V_{qq,t}^{(1)}(x,y,\mu,P_z) = \frac{|x|}{y(y-x)}\left(l_x-1\right) + \frac{|1-x|}{(1-y)(x-y)}\left(l_{1-x}-1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)}\left(l_{x-y}-1\right).$$

The full kernel is the sum of the fixed-order matrix $M_{\mathrm{fix}}$ (built from $V_{qq,h}/2$ plus the hybrid Wilson-line term $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi(y-x))$) and the leading-renormalon resummation:

$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0\, M_{C_z}\right) \exp\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where $M_{C_z}$ is the plus-prescribed matrix of the renormalon shape (Eq. (17) of the paper, with $\xi = x/y$, $p_z = |y|P_z$, $z_s = z_{\mathrm{spz}}/P_z$, and $\epsilon_m = 0.005$ GeV):

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s} p_z (1+\epsilon_m z_s + \epsilon_m^2 z_s^2)}{\epsilon_m^2 \pi} \delta(1-\xi) + \frac{e^{-\epsilon_m z_s}}{\pi}\left\{ -\frac{z_s \sin[(1-\xi)p_z z_s]}{1-\xi} + \frac{p_z}{(\epsilon_m^2 + p_z^2(1-\xi)^2)^2} \left[ (\epsilon_m^2 - p_z^2(1-\xi)^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2(1-\xi)^2 z_s)\cos[(1-\xi)p_z z_s] + p_z(1-\xi)(2\epsilon_m + \epsilon_m^2 z_s + p_z^2(1-\xi)^2 z_s)\sin[(1-\xi)p_z z_s] \right] \right\}_+,$$

with the plus prescription defined as in the paper:

$$[f(x)]_+ = f(x) - \delta(1-x)\int_0^1 f(\nu)\,d\nu.$$

The scalar coefficients are $r_0 = N_m |z\mu| (\beta_0/2\pi)^0 \Gamma(1+b)/\Gamma(1+b) (1 + b c_1/b) \alpha_s = N_m \alpha_s (1+c_1)$ (from Eq. (12) with $n=0$, $z=1$, $\mu$ in GeV) and the PV Borel sum (Eq. (13)):

$$r_{\mathrm{sumPV}} = N_m |z\mu| e^w \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\left[ E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w) \right], \qquad w = -\frac{2\pi}{\alpha_s \beta_0},$$

with $N_m(n_f=3)=0.575$, $\beta_0 = 11 - 2n_f/3$, $b = \beta_1/(2\beta_0^2)$, and $c_1, c_2$ from the sub-asymptotic corrections of Ref. [hep-ph/0105008]. The matrix exponential $\exp(-M_{C_z} r_{\mathrm{sumPV}})$ acts on the quasi-index (columns), and the plus prescription is implemented by making each column of $M_{C_z}$ integrate to zero over the full quasi grid.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 exactly for the PDF case, and the DA kernel is built by the universality assumption (the same $C_z$, $r_0$, $r_{\mathrm{sumPV}}$ applied to the DA fixed-order kernel). The fixed-order DA coefficient $V_{qq,h}$ matches Eq. (4.15) of arXiv:2212.14415 (the paper cited in the code for the DA kernel), not Eqs. (12)–(17) of 2305.05212 — the latter are the PDF matching coefficients. The code’s $C_z$ in Eq. (17) is transcribed correctly, including the $\epsilon_m$ regulator and the plus prescription. The only discrepancy found: the paper’s Eq. (17) writes the plus function with the subtraction domain $[0,1]$ in $\xi$, while the code’s `_plus_prescription_matrix` sums over the full quasi grid (which for the DA is $y\in[0,1]$, consistent). No other discrepancies were found; the code follows the paper’s notation and structure faithfully.
