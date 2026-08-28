<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gt_hybrid_rgr_nlo_re; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=82c229e154ae2499; paper_used=true -->
$$C_{\rm RGR}^{(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) + \frac{\alpha_s C_F}{2\pi}\frac{3}{2}\left(-\frac{1}{|1-\xi|}+\frac{2\,{\rm Si}[(1-\xi)|y|z_sP_z]}{\pi(1-\xi)}\right)\,\right]_{+(1)}^{[-\infty,\infty]} + \delta(1-\xi),$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the ratio-scheme coefficient is

$$C^{\rm ratio(1)}\left(\xi,\frac{\mu}{|x|P_z}\right) = \frac{\alpha_s C_F}{2\pi}\begin{cases}
\left(\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+1-\frac{3}{2(1-\xi)}\right)_{+(1)}^{[1,\infty]} & \xi>1,\\[4pt]
\left(\frac{1+\xi^2}{1-\xi}\left[-L+\ln\left(\frac{1-\xi}{\xi}\right)-1\right]+1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[0,1]} & 0<\xi<1,\\[4pt]
\left(-\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-1+\frac{3}{2(1-\xi)}\right)_{+(1)}^{[-\infty,0]} & \xi<0,
\end{cases}$$

with the plus-prescription defined as in the paper: $[g(\xi)]_{+(x_0)}^{D}$ subtracts the pole at $\xi=x_0$ over the domain $D$, i.e. $\int_D d\xi\,[g]_{+(x_0)}^{D}\,\phi(\xi)=\int_D d\xi\,g(\xi)[\phi(\xi)-\phi(x_0)]$ for any test function $\phi$. The hybrid correction adds the Wilson-line term with ${\rm Si}$ (sine integral) and the $3/2$ term, both regularized by the same plus prescription over $(-\infty,\infty)$.

The RGR kernel is built row-by-row: for each light-cone $x$, the fixed-order matrix is evaluated at the intrinsic scale $\mu_0(x)=2\kappa xP_z$ (with $\kappa$ the scale-variation parameter, $c'$ in the paper), then DGLAP-evolved to $\mu$ via a path-ordered matrix exponential of the two-loop non-singlet splitting function for the valence channel (the code uses the valence variant, $P_{\rm val}^{(2)}=P_{\rm full}^{(2)}+16C_F(C_F-C_A/2)[\cdots]$). Rows with $\mu_0(x)<\mu_{\min}$ (the perturbative cutoff, corresponding to the paper's $x_{\min}$) are set to zero. The scheme-specific correction is the hybrid ${\rm Si}$ term; no additional finite term beyond the paper's $3/2$ appears.

#### Consistency check

The code reproduces App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236 exactly for the NLO hybrid kernel: the regular coefficient $C^{\rm ratio(1)}$ matches Eq. (2.16) term-by-term (the splitting function, the $L$ log with argument $4y^2P_z^2/\mu^2$, the $\ln[(1-\xi)/\xi]$ and $\ln[\xi/(\xi-1)]$ branches, the $3/2$ terms, and the $\arctan/\arctanh$ piece via the code's `_atan_piece`), the plus-prescription domains and subtraction point $+(1)$ match the paper's bracket structure verbatim, and the hybrid correction (the ${\rm Si}$ term with $z_s|y|P_z$ argument) matches Eq. (2.20). The $\delta(1-\xi)$ term is implicit in the plus prescription and restored by the column-sum. No discrepancies found.

