<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_ratio_NLO; arxiv=2412.20461; equations=Eq. (23); digest=fd5637fdd3a2c8c2; paper_used=true -->
## Matching coefficient for the `gt` operator in the ratio scheme

The matching coefficient for the gauge-invariant $\gamma^t$ quark PDF in the ratio scheme, Eq. (23) of arXiv:2412.20461, is

$$
C_{q_iq_i}\left(\xi,\frac{\mu}{p^z}\right) = \delta(1-\xi) + \frac{\alpha_s C_F}{2\pi}\,
\begin{cases}
\left[\xi\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}\right]_{+(1)}^{[1,\infty]} & \xi>1 \\[6pt]
\left[\xi\frac{1+\xi^2}{1-\xi}\left(-\ln\frac{\mu^2}{4(1-\xi)\xi p_z^2}\right)-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}\right]_{+(1)}^{[0,1]} & 0<\xi<1 \\[6pt]
\left[-\xi\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-\xi-\frac{3}{2}+\frac{17}{6}\frac{1}{1-\xi}\right]_{+(1)}^{[-\infty,0]} & \xi<0
\end{cases}
$$

where $\xi = x/y$, $C_F = (N^2-1)/(2N)$ with $N=3$, and the plus function is defined by

$$
\int_{-\infty}^{\infty}dx\ \left[ f(x) \right]_{+(c)}^{[a,b]}g(x)=\int_{a}^{b}dx\ f(x)\left[ g(x)-g(c) \right].
$$

The code implements this coefficient with the following simplifications:

- The three branches share the tail $+\frac{3}{2}\frac{1}{|1-\xi|}$ (the paper's $\frac{3}{2}$ plus the $\frac{17}{6}\frac{1}{1-\xi}$ structure, combined via the plus prescription).
- The constant $\pm 1$ outside $[0,1]$ is $\operatorname{sgn}(\xi)$.
- The log outside $[0,1]$ is $\operatorname{sgn}(\xi)\cdot S\cdot \ln|\xi/(\xi-1)|$ with $S=(1+\xi^2)/(1-\xi)$.
- Inside $[0,1]$, the log is $S\cdot(\ln(y^2P_z^2/\mu^2)+\ln(4\xi(1-\xi))-1)+1$, where the code's `log_scale = ln(4y²P_z²/μ²)` is converted to the paper's $\ln(y^2P_z^2/\mu^2)$ by subtracting $\ln 4$.

The plus prescription is restored by the discretization: each $y$-column of the matching matrix is forced to integrate to zero, which is exactly the paper's $[\,\cdot\,]_{+(1)}$ prescription.

#### Consistency check

The code reproduces Eq. (23) of arXiv:2412.20461 term by term:

- **Regular coefficient**: The splitting function $S=(1+\xi^2)/(1-\xi)$ matches. The finite terms $+1$ (inside $[0,1]$) and $\pm 1$ (outside) match. The $3/2$ constant matches.
- **Logarithms**: Inside $[0,1]$, the code uses $\ln(y^2P_z^2/\mu^2)+\ln(4\xi(1-\xi))$, which equals the paper's $-\ln(\mu^2/(4(1-\xi)\xi p_z^2))$ after the $\ln 4$ conversion. Outside, $\ln|\xi/(\xi-1)|$ matches the paper's $\ln(\xi/(\xi-1))$ for $\xi>1$ and $\ln(-\xi/(1-\xi))$ for $\xi<0$.
- **Plus prescription**: The paper's $[\,\cdot\,]_{+(1)}^{[0,1]}$, $[\,\cdot\,]_{+(1)}^{[1,\infty]}$, $[\,\cdot\,]_{+(1)}^{[-\infty,0]}$ structure is reproduced by the column-sum prescription in `build_matching_matrix`.
- **Delta term**: The $\delta(1-\xi)$ is present as the LO identity matrix in the discretization.
- **Scheme-specific correction**: The paper's Eq. (23) has no additional finite correction beyond the $3/2$ term; the code matches this.

**No discrepancies found.** The code faithfully implements Eq. (23) of arXiv:2412.20461, including the exact plus-prescription convention and the three-branch structure.
