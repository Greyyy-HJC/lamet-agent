<!-- lamet-agent formula cache; kernel=GI_gtg5_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=5e71b269e099192d; paper_used=true -->
The matching coefficient for the `gtg5` operator in the `hybrid` scheme, as implemented by the kernel, is the NLO gauge-invariant $\gamma^t$ coefficient of Eq. (24) of arXiv:2412.20461, augmented by the leading-renormalon resummation (LRR) of Eqs. (12)–(17) of arXiv:2305.05212. With $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription $[g(\xi)]^{D}_{+(1)}$ defined by
$$
[g(\xi)]^{D}_{+(1)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu,
$$
the fixed-order part is
$$
\mathcal{C}_{\mathrm{NLO}}(\xi,L) = \left[\frac{1+\xi^2}{1-\xi}\left(L - \ln 4 + \ln(4\xi(1-\xi)) - 1\right) + 1 + \frac{3}{2|1-\xi|}\right]^{[0,1]}_{+(1)}
+ \left[\operatorname{sgn}(\xi)\left(\frac{1+\xi^2}{1-\xi}\ln\frac{|\xi|}{|\xi-1|} + 1\right) + \frac{3}{2|1-\xi|}\right]^{(-\infty,\infty)}_{+(1)}
+ \frac{3}{2}\left[-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}((1-\xi)|y|z_sP_z)}{\pi(1-\xi)}\right]^{(-\infty,\infty)}_{+(1)},
$$
where the last bracket is the hybrid scheme’s Wilson-line correction (the $\mathrm{Si}$ term), and the full kernel is
$$
M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0 M_{C_z}\right)\exp\left(-M_{C_z}\,r_{\mathrm{sumPV}}\right),
$$
with $M_{\mathrm{fix}}$ the discretized $\mathcal{C}_{\mathrm{NLO}}$ matrix, $M_{C_z}$ built from the renormalon shape
$$
C_z(\xi) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2z_s^2)}{\epsilon_m^2\pi}\delta(1-\xi) + \frac{e^{-\epsilon_m z_s}}{\pi}\left[-\frac{z_s\sin[(1-\xi)p_zz_s]}{1-\xi} + \frac{p_z}{(\epsilon_m^2+p_z^2(1-\xi)^2)^2}\left((\epsilon_m^2-p_z^2(1-\xi)^2+\epsilon_m^3z_s+\epsilon_mp_z^2(1-\xi)^2z_s)\cos[(1-\xi)p_zz_s] + p_z(1-\xi)(2\epsilon_m+\epsilon_m^2z_s+p_z^2(1-\xi)^2z_s)\sin[(1-\xi)p_zz_s]\right)\right],
$$
where $p_z=|y|P_z$, $z_s=z_s$ (the hybrid cutoff), $\epsilon_m=0.005$ GeV, and the plus-prescription is applied with the same $[\,\cdot\,]^{(-\infty,\infty)}_{+(1)}$ structure. The scalar coefficients are $r_0 = N_m(\beta_0/2\pi)^0\,\Gamma(1+b)/\Gamma(1+b)\,\alpha_s = N_m\alpha_s$ and $r_{\mathrm{sumPV}} = N_m|z\mu|e^w(-2\pi/\beta_0)\mathrm{Re}[E_{1+b}(w)+c_1E_b(w)+c_2E_{-1+b}(w)]$ with $w=-2\pi/(\alpha_s\beta_0)$, $N_m=0.575$ for $n_f=3$, and $b,c_1,c_2$ from the QCD beta function.

#### Consistency check
The code’s fixed-order coefficient matches Eq. (24) of arXiv:2412.20461 (the $\gamma^t$ hybrid kernel) exactly: the splitting function $(1+\xi^2)/(1-\xi)$, the log argument $4y^2P_z^2/\mu^2$ (with the $\ln 4$ removed to match the paper’s convention), the $+1$ constant, the $3/(2|1-\xi|)$ tail, and the $\mathrm{Si}$ term with strength $3/2$ all agree. The LRR resummation follows Eqs. (12)–(17) of arXiv:2305.05212: the renormalon shape $C_z$ is the Fourier transform of the regularized linear-$z$ tail (Eq. (17)), $r_0$ and $r_{\mathrm{sumPV}}$ are the PV Borel sums of Eqs. (12)–(13), and the matrix exponential implements the all-order resummation. One discrepancy: the paper’s Eq. (17) writes the plus-prescription with a $\delta(1-\xi)$ term and a domain $(-\infty,\infty)$, but the code’s $C_z$ omits the explicit delta (it is restored by the column-sum prescription), and the paper’s $\epsilon_m$ is a free parameter while the code fixes $\epsilon_m=0.005$ GeV. These are implementation choices, not sign or factor errors; the paper’s notation is preserved in the plus-bracket structure.
