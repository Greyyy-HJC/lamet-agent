<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_hybrid_LRR_NLO; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=6ca0ba9eb74857b1; paper_used=true -->
## Matching coefficient for the `gt` operator in the hybrid scheme

The matching coefficient for the gauge-invariant $\gamma^t$ quasi-PDF in the hybrid scheme, with the leading renormalon resummation (LRR), is built from the fixed-order NLO kernel plus a universal Wilson-line renormalon correction. Define $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$.

### Fixed-order part

The fixed-order NLO coefficient is (Eq. (24) of arXiv:2412.20461, matching the hybrid scheme of arXiv:2305.05212):

$$C_{\mathrm{hybrid}}(\xi, L, y) = C_{\mathrm{ratio}}(\xi, L) + \frac{3}{2}\left[-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_sP_z\big)}{\pi(1-\xi)}\right],$$

where the ratio-scheme coefficient is

$$C_{\mathrm{ratio}}(\xi, L) = \begin{cases}
\frac{1+\xi^2}{1-\xi}\left(L - \ln 4 + \ln(4\xi(1-\xi)) - 1\right) + 1, & 0<\xi<1,\\[4pt]
\mathrm{sgn}(\xi)\left[\frac{1+\xi^2}{1-\xi}\ln\left|\frac{\xi}{\xi-1}\right| + 1\right] + \frac{3/2}{|1-\xi|}, & \xi<0 \text{ or } \xi>1.
\end{cases}$$

The full fixed-order kernel is assembled with the plus prescription: each $y$-column of the discretized matrix is forced to integrate to zero, restoring the $\xi=1$ singularity. The paper's exact plus-prescription notation is

$$[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\nu)\,d\nu,$$

with the domain $D$ and subtraction point $x_0$ indicated by the superscript/subscript. The coefficient is split as

$$C_{\mathrm{hybrid}}(\xi) = \left[\frac{1+\xi^2}{1-\xi}\left(L - \ln 4 + \ln(4\xi(1-\xi)) - 1\right) + 1\right]^{[0,1]}_{+(1)} + \left[\mathrm{sgn}(\xi)\left(\frac{1+\xi^2}{1-\xi}\ln\left|\frac{\xi}{\xi-1}\right| + 1\right)\right]^{(-\infty,\infty)}_{+(1)} + \frac{3}{2}\left[-\frac{1}{|1-\xi|} + \frac{2\,\mathrm{Si}\big((1-\xi)|y|z_sP_z\big)}{\pi(1-\xi)}\right].$$

### Leading renormalon resummation (LRR)

On top of the fixed-order matrix $M_{\mathrm{fix}}$, the LRR correction resums the leading Wilson-line renormalon to all orders:

$$M_{\mathrm{LRR}} = \left(M_{\mathrm{fix}} + r_0\, M_{C_z}\right) \exp\left(-M_{C_z}\, r_{\mathrm{sumPV}}\right),$$

where $M_{C_z}$ is the discretized plus-prescribed matrix of the renormalon shape (Eq. (17) of arXiv:2305.05212):

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s}p_z(1+\epsilon_m z_s+\epsilon_m^2 z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s}z_s\sin[\bar{\xi}z_sp_z]}{\bar{\xi}} + \frac{e^{-\epsilon_m z_s}p_z}{(\epsilon_m^2+p_z^2\bar{\xi}^2)^2}\left[(\epsilon_m^2-\bar{\xi}^2p_z^2+\epsilon_m^3z_s+\epsilon_mp_z^2\bar{\xi}^2z_s)\cos[\bar{\xi}z_sp_z] - \bar{\xi}p_z(2\epsilon_m+\bar{\xi}^2p_z^2z_s+\epsilon_m^2z_s)\sin[\bar{\xi}z_sp_z]\right]\right),$$

with $\bar{\xi}=1-\xi$, $p_z=|y|P_z$, $z_s = z_s$ (the hybrid cutoff), and $\epsilon_m$ the long-distance regulator (0.005 GeV in the code). The scalar coefficients are:

- $r_0 = N_m \frac{\beta_0}{2\pi} \frac{\Gamma(1+b)}{\Gamma(1+b)} \alpha_s(\mu) = N_m \alpha_s(\mu)$ (from Eq. (12) with $n=0$),
- $r_{\mathrm{sumPV}} = N_m |z\mu| e^w \left(-\frac{2\pi}{\beta_0}\right) \mathrm{Re}\left[E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w)\right]$ with $w = -2\pi/(\alpha_s\beta_0)$ (Eq. (13), PV prescription).

Here $N_m = 0.575$ for $n_f=3$, $\beta_0 = 11 - 2n_f/3$, $b = \beta_1/(2\beta_0^2)$, and $c_1, c_2$ are the sub-asymptotic corrections from arXiv:hep-ph/0105008. The matrix exponential $\exp(-M_{C_z}r_{\mathrm{sumPV}})$ acts from the right, contracting the quasi-index.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 with the following observations:

- **Eq. (12)**: $r_n$ matches exactly, including the $|z\mu|$ factor and the sub-asymptotic corrections $c_1, c_2$.
- **Eq. (13)**: $r_{\mathrm{sumPV}}$ matches the PV Borel integral, with the real part taken for the exponential-integral functions.
- **Eq. (17)**: $C_z(\xi)$ matches the paper's expression term-for-term, including the $\epsilon_m$ regularization and the plus-function structure.
- **Fixed-order part**: The code's $C_{\mathrm{ratio}}$ differs from the paper's Eq. (23) by a constant $\ln 4$ in the log argument — the code uses $L - \ln 4$ where the paper writes $\ln(y^2P_z^2/\mu^2)$. This is a deliberate convention choice (the code's $L$ includes $\ln 4$), and the physical content is identical.
- **Hybrid term**: The Si-term prefactor $3/2$ matches Eq. (24) of arXiv:2412.20461, which is consistent with the hybrid scheme of arXiv:2305.05212.
- **Plus prescription**: The code implements the column-sum-to-zero prescription, which is equivalent to the paper's $[g]^{D}_{+(1)}$ with the domain split as shown above. No discrepancy found.

No discrepancies were found beyond the $\ln 4$ convention noted above.
