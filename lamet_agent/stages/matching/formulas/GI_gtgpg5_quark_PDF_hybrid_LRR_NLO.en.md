<!-- lamet-agent formula cache; kernel=GI_gtgpg5_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=a7efefe3b534eb38; paper_used=true -->
$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0\, M_{C_z}\right) \exp\!\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where $M_{\mathrm{fix}}$ is the fixed-order hybrid transversity kernel of Eq. (23) of arXiv:2208.08008, and the renormalon part is built from the universal Wilson-line shape $C_z(\xi)$ (Eq. (17) of arXiv:2305.05212), the scalar $r_0 = N_m |z\mu|\, \alpha_s$ (Eq. (12) with $n=0$), and the principal-value Borel sum $r_{\mathrm{sumPV}} = dPVasym(1,\mu,n_f,\alpha_s)$ (Eq. (13)). The matrix $M_{C_z}$ is the plus-prescribed discretization of $C_z(\xi)/|y|$ on the quasi grid, with the plus prescription defined by the column-sum-to-zero condition.

The fixed-order part is the transversity hybrid coefficient of arXiv:2208.08008 Eq. (23), which in the notation $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, $p_z=yP_z$, $\lambda_s = z_s P_z$ reads:

$$C_{\mathrm{fix}}(\xi, L, y) = C_{\mathrm{ratio}}(\xi, L) + 2\left[-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\!\left((1-\xi)|y|\lambda_s\right)}{\pi(1-\xi)}\right],$$

with the ratio part (Eq. (22)):

$$C_{\mathrm{ratio}}(\xi, L) = \begin{cases} \frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} - \frac{2}{1-\xi}, & \xi>1,\\[4pt] \frac{2\xi}{1-\xi}\left(L + \ln[\xi(1-\xi)]\right) + 2, & 0<\xi<1,\\[4pt] -\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} + \frac{2}{1-\xi}, & \xi<0. \end{cases}$$

The renormalon shape (Eq. (17), with $\bar\xi=1-\xi$, $M=\epsilon_m$, $z_s=z_s$, $p_z=|y|P_z$, and the regularization $z\to z e^{-\epsilon_m|z|}$) is:

$$C_z(\xi) = \begin{cases} \frac{e^{-\epsilon_m z_s} p_z (1+\epsilon_m z_s + \epsilon_m^2 z_s^2)}{\epsilon_m^2 \pi}, & \xi=1,\\[6pt] -\frac{e^{-\epsilon_m z_s} z_s \sin(\bar\xi z_s p_z)}{\pi \bar\xi} + \frac{e^{-\epsilon_m z_s} p_z}{\pi(\epsilon_m^2 + p_z^2 \bar\xi^2)^2}\Big[(\epsilon_m^2 - p_z^2\bar\xi^2 + \epsilon_m^3 z_s + \epsilon_m p_z^2\bar\xi^2 z_s)\cos(\bar\xi z_s p_z) + p_z\bar\xi(2\epsilon_m + \epsilon_m^2 z_s + p_z^2\bar\xi^2 z_s)\sin(\bar\xi z_s p_z)\Big], & \xi\neq 1. \end{cases}$$

The plus prescription is applied exactly as in the paper: the coefficient is split into plus-brackets over the domains $[0,1]$ and $(-\infty,\infty)$, with the subtraction point at $\xi=1$:

$$[g(\xi)]^{D}_{+(1)} = g(\xi) - \delta(1-\xi)\int_0^1 g(\nu)\,d\nu,$$

and the code implements this by making each $y$-column of the discretized matrix integrate to zero.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 term by term. The renormalon coefficients $r_n$ (Eq. (12)) match with $N_m(n_f=3)=0.575$, $b=\beta_1/(2\beta_0^2)$, $c_1=(\beta_1^2-\beta_0\beta_2)/(4b\beta_0^4)$, and the $c_2$ term from the paper's Eq. (12) tail. The PV Borel sum (Eq. (13)) matches the paper's expression with the exponential-integral representation and the real-part prescription. The shape $C_z$ (Eq. (17)) matches the paper's Eq. (17) exactly, including the $\epsilon_m$ regularization and the $\xi=1$ limit. The plus prescription matches the paper's definition with $x_0=1$ and the domain split as written. The only discrepancy is notational: the paper writes the plus function as $[\,\cdot\,]^{D}_{+(x_0)}$ with $x_0=1$, while the code's column-sum implementation is equivalent but does not display the subscript/superscript explicitly; no numerical or structural difference exists. The fixed-order transversity part is from arXiv:2208.08008, not from arXiv:2305.05212, but the renormalon resummation is exactly Eqs. (12)–(17) of the latter.
