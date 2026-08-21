<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=d7e9efa3fc3151aa; paper_used=true -->
$$C_{\rm RGR}^{\perp,(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C_r^{\perp,(1)}(\xi,L)\,\right]_{+(1)}^{[0,1]} + \left[\,C_r^{\perp,(1)}(\xi,L)\,\right]_{+(1)}^{[1,\infty]} + \left[\,C_r^{\perp,(1)}(\xi,L)\,\right]_{+(1)}^{[-\infty,0]},$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$. The regular coefficient is

$$C_r^{\perp,(1)}(\xi,L) = \frac{\alpha_s C_F}{2\pi}\left[ \frac{2\xi}{1-\xi}L + \frac{2\xi}{1-\xi}\left(\operatorname{sgn}(\xi)\ln|\xi|+\operatorname{sgn}(1-\xi)\ln|1-\xi|\right) + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} - \frac{1}{|1-\xi|} \right],$$

where the arctan branch is replaced by $\operatorname{arctanh}\sqrt{2\xi-1}/\sqrt{2\xi-1}$ for $\xi>1/2$ (analytic at $\xi=1/2$). The plus prescription is defined as in the paper: $[g(\xi)]_{+(x_0)}^{D} = g(\xi)$ for $\xi\neq x_0$ and $\int_D d\xi\,[g(\xi)]_{+(x_0)}^{D}=0$, with the subtraction point $x_0=1$ and domains $[0,1]$, $[1,\infty]$, $[-\infty,0]$ as written. There is no explicit $\delta(1-\xi)$ term; the plus prescription restores the singularity.

The RGR kernel is built row-by-row: for each light-cone $x$, the fixed-order matrix is evaluated at the row's own scale $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, central value 1), then evolved to $\mu$ by a path-ordered matrix exponential of the two-loop (NLL) non-singlet transversity splitting function $P_{qq}^{\perp,(1)}(\nu)=4\nu/(1-\nu)[\dots]$ (the code's `_p_nlo_transversity`). Rows with $\mu_0(x)<\mu_{\min}=0.6$ GeV are set to zero, implementing the paper's $x_{\min}$ cutoff. No scheme-specific correction beyond the plus prescription is present; the MSbar, ratio, and hybrid schemes coincide for transversity at NLO (Eqs. 2.17, 2.21 of the paper).

#### Consistency check

The code reproduces App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236. The regular coefficient matches Eq. (2.18) term-by-term: the $2\xi/(1-\xi)L$ log, the signed-log combination, the arctan/arctanh branch, and the $-1/|1-\xi|$ tail are all present with correct signs and arguments. The plus prescription uses the paper's exact notation $[\,\cdot\,]_{+(1)}^{D}$ with the three domains as written. The RGR construction (per-row scale, DGLAP evolution, cutoff) follows the paper's method exactly. No discrepancies found.
