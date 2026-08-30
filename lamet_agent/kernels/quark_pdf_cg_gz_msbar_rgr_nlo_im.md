<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gz_msbar_rgr_nlo_im; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=6d40d7cd2b353b29; paper_used=true -->
$$C_{\rm RGR}\left(\xi,\frac{\mu}{|x|P_z}\right) = \sum_{x_i} \delta_{x_i,x} \, \mathcal{U}\left(\mu_0(x_i)\to\mu\right) \otimes C^{(1)}\left(\xi,\frac{\mu_0(x_i)}{|x_i|P_z}\right),$$
with $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the per-row scale $\mu_0(x)=2\kappa xP^z$ ($\kappa$ the scale-variation knob).  
The fixed-order input is the NLO $\overline{\rm MS}$ kernel for the Coulomb-gauge $\gamma^z$ operator, Eq. (2.15) of the paper:
$$C^{(1)}_{\overline{\rm MS},\gamma^z}(\xi,L) = C^{(1)}_{\overline{\rm MS},\gamma^t}(\xi,L) + 2(1-\xi)_+ + \delta(1-\xi),$$
where the $\gamma^t$ kernel is
$$C^{(1)}_{\overline{\rm MS},\gamma^t}(\xi,L) = C^{(1)}_{\rm ratio}(\xi,L) + \frac{1}{2}\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[0,2]},$$
and the ratio-scheme coefficient is
$$C^{(1)}_{\rm ratio}(\xi,L) = \frac{\alpha_s C_F}{2\pi}\left[\left(\frac{1+\xi^2}{1-\xi}\right)\left(L+\ln\frac{\xi}{1-\xi}\right) + \xi - 1 + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} - \frac{3}{2|1-\xi|}\right]_{+(1)}^{[0,1]},$$
with the branch switching to $\operatorname{arctanh}\sqrt{2\xi-1}$ for $\xi>1/2$.  
The plus prescription is the paper's $[g(\xi)]^{D}_{+(x_0)}$, with $x_0=1$ and domain $D$ as indicated; the code restores it by subtracting each column's integral, and the $\delta(1-\xi)$ term is carried on the diagonal.  
The resummation evolves each row from $\mu_0(x)$ to $\mu$ via the path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function $P_{qq}^{(1)}+P_{qq}^{(2)}$ (the code uses the full unpolarized $q+\bar q$ channel), and rows with $\mu_0(x)<\mu_{\min}$ are set to zero, implementing the paper's $x_{\min}$ cutoff.

#### Consistency check
The code reproduces App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236 in structure: the per-row scale $\mu_0=2xP^z$, the DGLAP evolution operator, and the cutoff all match the paper's prescription. The fixed-order input matches Eq. (2.15) term by term: the $2(1-\xi)_+$ and $\delta(1-\xi)$ additions to $\gamma^t$, the $0.5/|1-\xi|$ MSbar correction, and the ratio-scheme logs and arctan/arctanh branch are all present with the correct signs and arguments. The plus prescription uses the paper's $[\,\cdot\,]^{D}_{+(1)}$ notation with the correct domains ($[0,1]$ for the ratio piece, $[0,2]$ for the MSbar correction, and $(-\infty,\infty)$ for the hybrid channel). No discrepancies were found between the code and the paper's notation or the implemented terms.

