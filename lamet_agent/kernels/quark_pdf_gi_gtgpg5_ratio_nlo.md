<!-- lamet-agent formula cache; kernel=quark_pdf_gi_gtgpg5_ratio_nlo; arxiv=2208.08008; equations=Eq. (22); digest=5a350573392aa453; paper_used=true -->
For the `gtgpg5` operator in the ratio scheme, the matching coefficient is given by Eq. (22) of arXiv:2208.08008. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the coefficient is

$$
C_r\left(\xi, \frac{\mu}{yP_z}\right) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi}
\begin{cases}
\left[\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} - \frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,\infty)} & \xi > 1 \\[6pt]
\left[\frac{2\xi}{1-\xi}\left(L + \ln \xi(1-\xi)\right) + 2\right]_{+(1)}^{[0,1]} & 0 < \xi < 1 \\[6pt]
\left[-\frac{2\xi}{1-\xi}\ln\frac{\xi}{\xi-1} + \frac{2}{1-\xi}\right]_{+(1)}^{(-\infty,\infty)} & \xi < 0
\end{cases}
$$

where the plus-prescription is defined as in the paper:

$$
\int_0^1 dx\, [g(x)]^{D}_{+(x_0)} f(x) = \int_0^1 dx\, g(x) \left[f(x) - f(x_0)\right], \quad x_0 = 1,
$$

with the domain $D$ being either $[0,1]$ or $(-\infty,\infty)$ as indicated. The $\delta(1-\xi)$ term is the LO contribution. There is no additional scheme-specific finite correction beyond the $+2$ constant in the $0<\xi<1$ branch.

#### Consistency check

The code `C_ratio_gi_perp` reproduces Eq. (22) exactly. The regular coefficient matches: the splitting $2\xi/(1-\xi)$ appears in all branches, the $2/(1-\xi)$ tail only outside $[0,1]$, and the $+2$ constant only inside. The logarithms match: $\ln(\xi/(\xi-1))$ on the outer branches and $L + \ln(\xi(1-\xi))$ inside, with $L$ correctly identified as $\ln(4y^2P_z^2/\mu^2)$. The plus-prescription is implemented via the column-sum method, which is equivalent to the paper's $[\,\cdot\,]_{+(1)}^{D}$ definition. The $\delta(1-\xi)$ term is present as the LO identity. No discrepancies found.

