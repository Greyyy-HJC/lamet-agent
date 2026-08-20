<!-- lamet-agent formula cache; kernel=GI_gtgpg5_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=e2e20f3df1bbd92c; paper_used=true -->
$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0\, M_{C_z}\right) \exp\!\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where $M_{\mathrm{fix}}$ is the fixed-order hybrid kernel of Eq. (23) of arXiv:2208.08008, and the renormalon shape $C_z(\xi)$ is the Fourier transform of the regularized linear-$z$ tail, Eq. (17) of arXiv:2305.05212 with $\xi=x/y$, $\bar\xi=1-\xi$, $p_z=|y|P_z$, $z_s=z_s^{\mathrm{hybrid}}$, and $\epsilon_m$ the long-distance regulator:

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s}z_s\sin[\bar\xi z_sp_z]}{\bar\xi} + \frac{e^{-\epsilon_m z_s}p_z}{(\epsilon_m^2+p_z^2\bar\xi^2)^2}\left[(\epsilon_m^2-\bar\xi^2p_z^2+\epsilon_m^3z_s+\epsilon_mp_z^2\bar\xi^2z_s)\cos[\bar\xi z_sp_z] - \bar\xi p_z(2\epsilon_m+\bar\xi^2p_z^2z_s+\epsilon_m^2z_s)\sin[\bar\xi z_sp_z]\right]\right),$$

with the $\xi=1$ limit $C_z(1)=e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)/(\epsilon_m^2\pi)$. The scalar renormalon numbers are $r_0 = N_m |z\mu|\,(\beta_0/2\pi)^0\,\Gamma(1+b)/\Gamma(1+b)\,(1+bc_1/b) = N_m |z\mu|\,(1+c_1)$ evaluated at $z=1$, $\mu$, and $r_{\mathrm{sumPV}} = N_m |z\mu|\, e^w\,(-2\pi/\beta_0)\,\mathrm{Re}\left[E_{1+b}(w)+c_1E_b(w)+c_2E_{-1+b}(w)\right]$ with $w=-2\pi/(\alpha_s\beta_0)$, $E_\nu$ the exponential integral, and $b,c_1,c_2$ the sub-asymptotic corrections of arXiv:hep-ph/0105008. The matrix $M_{C_z}$ is the plus-prescribed discretization of $C_z(x/y)/|y|$ with the column-sum-to-zero condition, and the matrix exponential acts on the quasi-index.

The fixed-order part $M_{\mathrm{fix}}$ is the hybrid transversity kernel of arXiv:2208.08008 Eq. (23), whose coefficient is

$$C_{\mathrm{hybrid}}^{\perp}(\xi, L, y) = C_r^{\perp}(\xi, L) + 2\left[-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_sP_z)}{\pi(1-\xi)}\right],$$

with the ratio coefficient

$$C_r^{\perp}(\xi, L) = \begin{cases} \frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} - \frac{2}{1-\xi}, & \xi>1, \\[4pt] \frac{2\xi}{1-\xi}\left(L + \ln[\xi(1-\xi)]\right) + 2, & 0<\xi<1, \\[4pt] -\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} + \frac{2}{1-\xi}, & \xi<0, \end{cases}$$

and $L=\ln(4y^2P_z^2/\mu^2)$. The plus prescription is applied at $\xi=1$ with the domain $D=(-\infty,\infty)$, following the paper's notation $[g(\xi)]^{D}_{+(1)}$, defined by $[g(\xi)]^{D}_{+(1)} = g(\xi) - \delta(1-\xi)\int_0^1 g(\nu)\,d\nu$.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 exactly. The renormalon shape $C_z(\xi)$ in Eq. (17) matches the code's `C_z_lrr` term-for-term, including the $\xi=1$ limit and the $\epsilon_m$ regularization. The scalar $r_0$ and $r_{\mathrm{sumPV}}$ match Eqs. (12) and (13) with the paper's $N_m(n_f=3)=0.575$, $b=\beta_1/(2\beta_0^2)$, and the PV prescription taking the real part of the exponential-integral combination. The matrix assembly $M_{\mathrm{LRR}}=(M_{\mathrm{fix}}+r_0M_{C_z})\exp(-M_{C_z}r_{\mathrm{sumPV}})$ is exactly the code's `_lrr_improve`. The fixed-order transversity kernel matches Eq. (23) of arXiv:2208.08008, including the branch structure and the $+2$ constant in the central region. The plus prescription in the code (column-sum-to-zero) is equivalent to the paper's $[g]^{D}_{+(1)}$ with $D=(-\infty,\infty)$ and the $\delta(1-\xi)$ subtraction. No discrepancies found.
