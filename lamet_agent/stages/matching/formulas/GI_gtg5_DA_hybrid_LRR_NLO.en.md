<!-- lamet-agent formula cache; kernel=GI_gtg5_DA_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=74ae10e71ca9cc53; paper_used=true -->
$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0\, M_{C_z}\right) \exp\!\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where $M_{\mathrm{fix}}$ is the fixed-order hybrid-scheme kernel for the $\gamma^t\gamma_5$ meson-DA operator, discretized as

$$M_{\mathrm{fix}} = \mathbb{1} - \frac{\alpha_s C_F}{2\pi}\, \mathcal{M}, \qquad \mathcal{M}_{xy} = \frac{1}{2} V_{qq,h}^{(1)}(x,y)\, dy + \frac{3\,\mathrm{Si}\!\left(z_s P_z (y-x)\right)}{\pi (y-x)}\, dy,$$

with the plus prescription applied column-wise so that each $y$-column of $\mathcal{M}$ integrates to zero over the $x$ grid. The fixed-order coefficient is

$$V_{qq,h}^{(1)}(x,y) = \frac{|x|}{y}\left(l_x - 1\right) + \frac{|1-x|}{1-y}\left(l_{\bar x} - 1\right) + \frac{|x-y|}{y(y-1)}\left(l_{xy} - 1\right) + V_{qq,t}^{(1)}(x,y),$$

with

$$V_{qq,t}^{(1)}(x,y) = \frac{|x|}{y(y-x)}\left(l_x - 1\right) + \frac{|1-x|}{(1-y)(x-y)}\left(l_{\bar x} - 1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)}\left(l_{xy} - 1\right),$$

and the logarithms are

$$l_x = \ln\!\left(\frac{4P_z^2 x^2}{\mu^2}\right), \qquad l_{\bar x} = \ln\!\left(\frac{4P_z^2 (1-x)^2}{\mu^2}\right), \qquad l_{xy} = \ln\!\left(\frac{4P_z^2 (x-y)^2}{\mu^2}\right).$$

The plus prescription is the paper’s bracket, defined as

$$[f(x)]_+ = f(x) - \delta(1-x)\int_0^1 f(\nu)\,d\nu,$$

applied here at $x=y$ with the subtraction integrated over the full quasi grid $y\in[0,1]$.

The leading-renormalon resummation uses the universal Wilson-line shape (Eq. (17) of the paper, with $\xi = x/y$ only inside the shape, not in the overall kernel):

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s} p_z \left(1 + \epsilon_m z_s + \epsilon_m^2 z_s^2\right)}{\epsilon_m^2 \pi} \quad \text{at } \xi=1,$$

and for $\xi\neq 1$,

$$C_z(\xi) = -\frac{e^{-\epsilon_m z_s} z_s \sin\!\left(p_z z_s (1-\xi)\right)}{\pi (1-\xi)} + \frac{e^{-\epsilon_m z_s} p_z}{\pi \left(\epsilon_m^2 + p_z^2 (1-\xi)^2\right)^2} \left[ \left(\epsilon_m^2 - p_z^2 (1-\xi)^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2 (1-\xi)^2 z_s\right) \cos\!\left(p_z z_s (1-\xi)\right) + p_z (1-\xi) \left(2\epsilon_m + \epsilon_m^2 z_s + p_z^2 (1-\xi)^2 z_s\right) \sin\!\left(p_z z_s (1-\xi)\right) \right],$$

where $p_z = |y| P_z$, $z_s = z_s$ (the hybrid cutoff), $\epsilon_m = 0.005$ GeV, and the matrix $M_{C_z}$ is built from $C_z(x/y)/|y|$ with the same plus prescription. The scalar coefficients are

$$r_0 = N_m\, |z\mu|\, \frac{\Gamma(1+b)}{\Gamma(1+b)} \left(1 + \frac{b c_1}{b}\right) \alpha_s(\mu) = N_m\, \alpha_s(\mu),$$

and the principal-value Borel sum

$$r_{\mathrm{sumPV}} = N_m\, |z\mu|\, e^{w} \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\!\left[ E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w) \right], \qquad w = -\frac{2\pi}{\alpha_s(\mu)\beta_0},$$

with $N_m(n_f=3)=0.575$, $\beta_0 = 9$, $b = \beta_1/(2\beta_0^2)$, $c_1 = (\beta_1^2 - \beta_0\beta_2)/(4b\beta_0^3)$, $c_2$ as in the code, and $E_\nu$ the exponential integral. The matrix exponential $\exp(-M_{C_z} r_{\mathrm{sumPV}})$ acts on the quasi (column) index, contracting with $M_{\mathrm{fix}}$ from the right.

#### Consistency check

The code implements Eqs. (12)–(17) of arXiv:2305.05212 as follows: Eq. (12) ($r_n$) matches the paper’s form with $N_m=0.575$ for $n_f=3$; Eq. (13) ($r_{\mathrm{sumPV}}$) matches the PV Borel integral with the exponential-integral evaluation; Eq. (17) ($C_z$) matches the paper’s Fourier transform of the regularized linear-$z$ tail, including the $\epsilon_m$ regulator and the plus-function structure. The fixed-order kernel $V_{qq,h}^{(1)}$ is taken from arXiv:2212.14415 (Eq. (4.15)), not from 2305.05212, and the hybrid Wilson-line term $3\,\mathrm{Si}(z_s P_z (y-x))/(\pi(y-x))$ is the paper’s scheme-specific correction. The code’s plus prescription (column-sum-to-zero over the full quasi grid) matches the paper’s bracket definition with the subtraction domain $[0,1]$. No discrepancies were found between the code and the paper for the renormalon resummation; the fixed-order DA coefficient is from a different reference, which the code states explicitly.
