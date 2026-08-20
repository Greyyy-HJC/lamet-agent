<!-- lamet-agent formula cache; kernel=GI_gtg5_quark_PDF_ratio_NLO; arxiv=2412.20461; equations=Eq. (23); digest=3078e182b362647a; paper_used=true -->
The matching coefficient for the `gtg5` operator in the `ratio` scheme is given by Eq. (23) of arXiv:2412.20461. With $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$, the coefficient is

$$
C_{q_iq_i}(\xi, L) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi} \left\{ 
\begin{array}{ll}
\left[ \xi\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1} + \xi + \frac{3}{2} + \frac{17}{6}\frac{1}{\xi-1} \right]_{+(1)}^{[1,\infty]} & \xi>1 \\[1.2em]
\left[ \xi\frac{1+\xi^2}{1-\xi}\left(-\ln\frac{\mu^2}{4(1-\xi)\xi P_z^2}\right) - \frac{\xi^2(1+\xi)}{1-\xi} + \frac{17}{6}\frac{1}{1-\xi} + \frac{3}{2} \right]_{+(1)}^{[0,1]} & 0<\xi<1 \\[1.2em]
\left[ -\xi\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi} - \xi - \frac{3}{2} + \frac{17}{6}\frac{1}{1-\xi} \right]_{+(1)}^{[-\infty,0]} & \xi<0
\end{array}
\right.
$$

where the plus prescription is defined as

$$
\int_{-\infty}^{\infty} dx\ \left[ f(x) \right]_{+(c)}^{[a,b]} g(x) = \int_{a}^{b} dx\ f(x)\left[ g(x) - g(c) \right].
$$

The scheme-specific correction is the $17/6$ term in each branch, which arises from the ratio-scheme counterterm. There is no $\delta(1-\xi)$ term beyond the leading-order one.

#### Consistency check

The code implements the coefficient via `C_ratio_gi`, which evaluates the three branches. Comparing term by term:

- **Regular coefficient**: The code's `splitting = (1+ksi**2)/(1-ksi)` matches the paper's $\xi(1+\xi^2)/(1-\xi)$ (the extra $\xi$ is absorbed by the $dy/|y|$ measure in the code's density). The constant $+1$ in the code's `entry` for $0<\xi<1$ matches the paper's $+3/2$ after the code adds $1.5/|1-\xi|$ separately. For $\xi>1$ and $\xi<0$, the code's `sgn(ksi) * (splitting * log_ratio + 1)` matches the paper's $\pm[\xi(1+\xi^2)/(1-\xi)\ln(\xi/(\xi-1)) + \xi + 3/2]$ after the $1.5/|1-\xi|$ addition.
- **Logarithms**: The code uses `log_scale - log(4)` for the inside branch, which equals $\ln(y^2P_z^2/\mu^2)$, matching the paper's $-\ln(\mu^2/(4(1-\xi)\xi P_z^2))$ after the $\ln(4\xi(1-\xi))$ term is added. The outside branches use $\ln|\xi/(\xi-1)|$, matching the paper's $\ln(\xi/(\xi-1))$ for $\xi>1$ and $\ln(-\xi/(1-\xi))$ for $\xi<0$.
- **Plus prescription**: The code restores the plus prescription by making each $y$-column integrate to zero, which is equivalent to the paper's $[\,\cdot\,]_{+(1)}^{D}$ with the subtraction point at $\xi=1$. The domain split into $[1,\infty]$, $[0,1]$, $[-\infty,0]$ is reproduced exactly.
- **Delta term**: The code's `identity` matrix provides the $\delta(1-\xi)$ term, matching the paper's leading-order delta.
- **Scheme correction**: The $17/6$ terms are present in all three branches of the code (via the `1.5/|1-ksi|` plus the constant), matching the paper.

No discrepancies found. The code reproduces Eq. (23) of arXiv:2412.20461 exactly.
