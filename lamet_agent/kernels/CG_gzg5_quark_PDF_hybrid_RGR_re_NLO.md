<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_hybrid_RGR_re_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=e05ac695c72b48ba; paper_used=true -->
$$C^{(1)}_{\rm RGR}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{(1)}_{\rm hybrid}\left(\xi,\frac{\mu_0(x)}{|x|P_z}\right)\right]_{+(1)}^{[-\infty,\infty]} \otimes \mathcal{E}\left(\mu_0(x),\mu\right), \qquad \mu_0(x) = 2\kappa x P_z,$$

where $\xi = x/y$, $L = \ln(4y^2P_z^2/\mu^2)$, and the fixed-order hybrid kernel is

$$C^{(1)}_{\rm hybrid}\left(\xi,\frac{\mu}{|x|P_z}\right) = C^{(1)}_{\rm ratio}\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left[-\frac{1}{|1-\xi|}+\frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right]_{+(1)}^{[-\infty,\infty]},$$

with the ratio-scheme coefficient (Eq. (2.16) of the paper, plus the $\gamma^z$ shift $2(1-\xi)$ on $0<\xi<1$):

$$C^{(1)}_{\rm ratio}\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\begin{cases}
\left(\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+1-\frac{3}{2(1-\xi)}\right)_{+(1)}^{[1,\infty]} & \xi>1,\\[4pt]
\left(\frac{1+\xi^2}{1-\xi}\left[-\ln\frac{\mu^2}{4x^2P_z^2}+\ln\left(\frac{1-\xi}{\xi}\right)-1\right]+1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[0,1]} & 0<\xi<1,\\[4pt]
\left(-\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[-\infty,0]} & \xi<0,
\end{cases}$$

plus the $\gamma^z$ shift $2(1-\xi)$ on $0<\xi<1$ (the paper's Eq. (2.15) for MSbar, extended to ratio/hybrid per the code). The plus prescription is defined as in the paper: $[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D g(\xi')d\xi'$, with the subtraction point $x_0=1$ and domain $D$ as indicated.

The RGR evolution operator is a path-ordered matrix exponential over $\ln\mu^2$:

$$\mathcal{E}(\mu_0,\mu) = \mathcal{P}\exp\left[\int_{\ln\mu_0^2}^{\ln\mu^2} \frac{d\ln\mu'^2}{2}\left(\frac{\alpha_s(\mu')}{4\pi}P^{(1)}_{\rm NS} + \left(\frac{\alpha_s(\mu')}{4\pi}\right)^2 P^{(2)}_{\rm NS}\right)\right],$$

where $P^{(1)}_{\rm NS}$ is the LO non-singlet splitting function and $P^{(2)}_{\rm NS}$ is the two-loop (NLL) non-singlet splitting function for the *full* helicity channel $(q+\bar{q})/2$ (the code uses `_p_nlo_full_helicity`, which is the valence kernel plus the extra $n_f$ structure). Rows with $\mu_0(x) < \mu_{\rm min}$ (the paper's $x_{\rm min}$) are set to zero, since $\alpha_s(2xP_z)$ is out of perturbative control.

#### Consistency check

The code reproduces the paper's App. 'A Method Solving RG Equation' (Eq. matchingRGI) structure: the per-row scale $\mu_0(x)=2\kappa xP_z$, the DGLAP evolution from $\mu_0$ to $\mu$, and the cutoff at small $x$ all match the paper's description. The fixed-order hybrid kernel matches the paper's Eq. (2.19)–(2.20) with the $\gamma^z$ shift of Eq. (2.15) applied to the ratio backbone. The plus prescription and its domain split ($[0,1]$, $[1,\infty]$, $[-\infty,0]$, and the overall $[-\infty,\infty]$ for the Wilson-line correction) follow the paper's notation exactly. The one discrepancy: the paper's Eq. (2.15) writes the $\gamma^z$ shift as $[2(1-\xi)]_+ + \delta(1-\xi)$ for MSbar, while the code applies the bare $2(1-\xi)$ (without the delta) in the ratio/hybrid schemes, arguing the delta is an MSbar normalization term. This is a deliberate scheme-interpretation choice, not an error, but it differs from a literal reading of the paper. No other discrepancies found.

