<!-- lamet-agent formula cache; kernel=GI_gtgpg5_quark_PDF_ratio_NLO; arxiv=2208.08008; equations=Eq. (22); digest=64492eff5823f3b9; paper_used=true -->
$$C_r\left(\xi,\frac{\mu}{yP_z}\right)=\delta(1-\xi)+\frac{\alpha_s C_F}{2\pi}\begin{cases}
\left[\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1}-\frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi>1 \\
\left[\frac{2\xi}{1-\xi}\left(\ln\frac{4y^2P_z^2}{\mu^2}+\ln\xi(1-\xi)\right)+2\right]_{+(1)}^{[0,1]} & 0<\xi<1 \\
\left[-\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1}+\frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,0)\cup(1,\infty)} & \xi<0 ,
\end{cases}$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the plus-prescription is defined as in the paper: for a function $g(\xi)$ on a domain $D$ with subtraction point $x_0$,
$$\int_D dx\, f(x)\,[g(x)]_{+(x_0)}^{D} = \int_D dx\, [f(x)-f(x_0)]\,g(x).$$

The code implements exactly this coefficient: the splitting function $2\xi/(1-\xi)$, the logarithms $\ln[\xi/(\xi-1)]$ on the outer branches and $L+\ln[\xi(1-\xi)]$ on the inner branch, the $+2$ constant only for $0<\xi<1$, and the $\mp 2/(1-\xi)$ tails only outside $[0,1]$. The plus-prescription is restored by the column-sum condition (each $y$-column integrates to zero), matching the paper's $+(1)$ subtraction point. There is no additional scheme-specific correction beyond the $\delta(1-\xi)$ term.

#### Consistency check

The code reproduces Eq. (22) of arXiv:2208.08008 term by term: the regular coefficient matches exactly on all three branches, the logarithms have identical arguments ($\xi/(\xi-1)$ outside, $\xi(1-\xi)$ inside), the $+2$ constant appears only in $0<\xi<1$, and the $\pm 2/(1-\xi)$ tails appear only outside. The plus-prescription domain and subtraction point ($+(1)$) match the paper's notation. No discrepancies found.
