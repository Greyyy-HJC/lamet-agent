<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gtgpg5_hybrid_rgr_nlo_re; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=2b21f4164e344a55; paper_used=true -->
$$C_{\rm RGR}^{\rm gtgpg5}\left(\xi,\frac{\mu}{|x|P_z}\right) = \sum_{x_i} \Theta\!\left(2\kappa x_i P_z - \mu_{\rm min}\right) \left[ \mathcal{U}(\mu_0(x_i),\mu) \, C^{\rm hyb,(1)}_{\rm gtgpg5}\!\left(\xi,\frac{\mu_0(x_i)}{|x|P_z}\right) \right]_{x_i},$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the fixed-order hybrid kernel is (Eq. (2.21) of the paper, with $\delta C_{\rm hyb}=0$ so it equals the ratio-scheme coefficient of Eq. (2.18)):

$$C^{\rm hyb,(1)}_{\rm gtgpg5}\!\left(\xi,\frac{\mu}{|x|P_z}\right) = C^{\rm ratio,(1)}_{\perp}(\xi,L),$$

with the regular coefficient (the code’s `C_ratio_perp`):

$$C^{\rm ratio,(1)}_{\perp}(\xi,L) = \frac{2\xi}{1-\xi}L + \frac{2\xi}{1-\xi}\Big[\operatorname{sgn}(\xi)\ln|\xi|+\operatorname{sgn}(1-\xi)\ln|1-\xi|\Big] + \frac{3\xi-1}{\xi-1}\frac{\arctan\!\big(\sqrt{|1-2\xi|}/|\xi|\big)}{\sqrt{|1-2\xi|}} - \frac{1}{|1-\xi|},$$

with the branch of the arctan/arctanh term chosen by $\xi\lessgtr 1/2$ (analytic at $\xi=1/2$). The plus prescription is applied by the column-sum subtraction in `build_matching_matrix`, which makes each $y$-column integrate to zero; in the paper’s notation this is the bracket $[\,\cdot\,]^{[-\infty,\infty]}_{+(1)}$ (the paper writes the domain as superscript and the subtraction point as subscript, e.g. $[g]^{[0,1]}_{+(1)}$). There is no $\delta(1-\xi)$ term beyond the LO identity.

The resummation: each row $x_i$ is matched at its own scale $\mu_0(x_i)=2\kappa x_i P_z$ (with $\kappa$ the scale-variation knob, $c'$ in the paper), then evolved to $\mu$ by the path-ordered matrix exponential

$$\mathcal{U}(\mu_0,\mu) = \mathcal{P}\exp\!\left[\int_{\ln\mu_0^2}^{\ln\mu^2} \frac{d\ln t^2}{2}\Big(\frac{\alpha_s(t)}{4\pi}P^{(0)} + \Big(\frac{\alpha_s(t)}{4\pi}\Big)^2 P^{(1)}_{\rm NS}\Big)\right],$$

using the LO and NLO non-singlet splitting functions; the code uses the transversity NLO kernel $P^{(1)}_{\perp}$ (built on $4\nu/(1-\nu)$ with the $C_F^2$, $C_AC_F$, $C_Fn_f$ terms of the two-loop transversity anomalous dimension). Rows with $\mu_0(x_i)<\mu_{\rm min}$ (the paper’s $x_{\rm min}$) are set to zero. No $Z_\psi$ factor is applied (the hybrid scheme cancels it by ratios).

#### Consistency check

The code reproduces App. “A Method Solving RG Equation” (Eq. matchingRGI) of arXiv:2209.01236: the per-row scale $\mu_0=2\kappa xP_z$ matches the paper’s $Q_{\rm eff}=2xP_zc'$, the evolution operator is the DGLAP solution of Eq. (matchingRGI), and the cutoff $\mu_{\rm min}$ implements the paper’s $x_{\rm min}$. The fixed-order input matches Eq. (2.18) of the paper (the transversity ratio coefficient) with the same log argument $L=\ln(4y^2P_z^2/\mu^2)$, the same $2\xi/(1-\xi)$ splitting, the same arctan/arctanh branch, and the same $-1/|1-\xi|$ tail. The plus prescription is the paper’s $[\,\cdot\,]^{[-\infty,\infty]}_{+(1)}$ (domain superscript, subtraction point subscript), restored by column-sum subtraction. No discrepancies found: the code’s `C_ratio_perp` matches Eq. (2.18) term by term, and the RGR construction follows the paper’s method exactly.

