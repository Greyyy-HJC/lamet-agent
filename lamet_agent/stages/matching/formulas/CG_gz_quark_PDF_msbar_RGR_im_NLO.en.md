<!-- lamet-agent formula cache; kernel=CG_gz_quark_PDF_msbar_RGR_im_NLO; arxiv=2209.01236; equations=App. 'A Method Solving RG Equation' (Eq. matchingRGI); digest=781546975d934f6e; paper_used=true -->
$$C_{\rm RGR}\left(\xi,\frac{\mu}{|x|P_z}\right) = \left[\,C^{(1)}\left(\xi,\frac{\mu_0}{|x|P_z}\right)\right]_{+(1)}^{[-\infty,\infty]} + \delta(1-\xi)\,\left[1 + \frac{\alpha_s(\mu_0)C_F}{2\pi}\left(\frac{3}{2}\ln\frac{\mu_0^2}{4x^2P_z^2} + \frac{5}{2}\right)\right]$$

where $\xi = x/y$, $L = \ln(4y^2P_z^2/\mu^2)$, and the per-row scale is $\mu_0 = 2\kappa xP_z$ with $\kappa$ the scale-variation parameter. The regular coefficient is

$$C^{(1)}(\xi,L) = \frac{\alpha_s(\mu_0)C_F}{2\pi}\left[\frac{1+\xi^2}{1-\xi}\left(L + \ln\frac{1-\xi}{\xi} - 1\right) + \xi - 1 + \frac{3\xi-1}{\xi-1}\frac{\arctan\sqrt{1-2\xi}}{\sqrt{1-2\xi}} - \frac{3}{2(1-\xi)}\right]$$

for $0<\xi<1$, with the $\arctan$ replaced by $\arctanh$ for $\xi>1/2$ (analytic across $\xi=1/2$). The plus-prescription is defined as in the paper: $[g(\xi)]^{D}_{+(x_0)} = g(\xi) - \delta(1-\xi)\int_D d\xi'\, g(\xi')$, with the domain $D=[-\infty,\infty]$ and subtraction point $x_0=1$. The scheme-specific correction relative to the ratio scheme is the $+\frac{1}{2|1-\xi|}$ term (included in the $-\frac{3}{2(1-\xi)}$ above) plus the diagonal $\delta(1-\xi)$ term shown.

The resummation is implemented by evaluating the fixed-order kernel at each row's own scale $\mu_0(x)$, then evolving to $\mu$ via a path-ordered matrix exponential of the two-loop (NLL) non-singlet splitting function $P_{qq}^{(1)}$ (the full unpolarized $q+\bar{q}$ channel). Rows with $\mu_0(x) < \mu_{\min}$ (the perturbative cutoff, corresponding to the paper's $x_{\min}$) are set to zero.

#### Consistency check

The code reproduces the structure of App. 'A Method Solving RG Equation' (Eq. matchingRGI) of arXiv:2209.01236: the per-row scale $\mu_0=2\kappa xP_z$, the DGLAP evolution operator, and the cutoff at small $x$ all match the paper's prescription. The regular coefficient matches Eq. (2.16) of the paper (the ratio-scheme kernel) with the MSbar correction $+1/(2|1-\xi|)$ as in Eq. (2.14). The plus-prescription domain $[-\infty,\infty]$ and subtraction point $+(1)$ match the paper's notation exactly. The $\delta(1-\xi)$ term carries the correct finite part. No discrepancies found between the code and the paper for this kernel.
