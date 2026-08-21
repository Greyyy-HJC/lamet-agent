<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_ratio_NLO; arxiv=2412.20461; equations=Eq. (23); digest=82c33a3c46628b5a; paper_used=true -->
$$C_{q_iq_i}^{\mathrm{ratio}}(\xi,L)=\delta(1-\xi)+\frac{\alpha_s C_F}{2\pi}\left\{ \left[ \xi\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}\right]_{+(1)}^{[1,\infty]} \theta(\xi-1) \right.$$
$$\left.+\left[ \xi\frac{1+\xi^2}{1-\xi}\left(-\ln\frac{\mu^2}{4(1-\xi)\xi p_z^2}\right)-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}\right]_{+(1)}^{[0,1]} \theta(\xi)\theta(1-\xi) \right.$$
$$\left.+\left[ -\xi\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-\xi-\frac{3}{2}+\frac{17}{6}\frac{1}{1-\xi}\right]_{+(1)}^{[-\infty,0]} \theta(-\xi) \right\}$$
$$-\frac{\alpha_s T_F}{2\pi}\left\{ \left[\frac{1}{3}-\frac{1}{3}\ln\left(\frac{\mu^2}{4p_z^2}\right)\right]\delta(1-\xi)+\frac{1}{3}\left[\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[0,2]}+\frac{1}{|\xi-1|}\theta(-\xi)+\frac{1}{|1-\xi|}\theta(\xi-2)\right]\right\}\frac{\langle x\rangle_g}{\langle x\rangle_i},$$

with the plus prescription defined as

$$\int_{-\infty}^{\infty}dx\ \left[ f(x) \right]_{+(c)}^{[a,b]}g(x)=\int_{a}^{b}dx\ f(x)\left[ g(x)-g(c) \right].$$

Here $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, $C_F=(N^2-1)/(2N)$, $T_F=1/2$, and the scheme-specific correction is the $\delta(1-\xi)$ term proportional to $\langle x\rangle_g/\langle x\rangle_i$.

#### Consistency check

The code implements Eq. (23) of arXiv:2412.20461. Comparing term by term:

- **Regular coefficient**: The code's three branches match the paper's three regions ($\xi>1$, $0<\xi<1$, $\xi<0$) exactly, including the splitting function $S=(1+\xi^2)/(1-\xi)$, the constants $+3/2$ and $+17/6/(1-\xi)$, and the sign structure $\mathrm{sgn}(\xi)$ for the outer regions.
- **Logarithms**: Inside $[0,1]$, the code uses $\ln(y^2P_z^2/\mu^2)+\ln(4\xi(1-\xi))-1$, which matches the paper's $-\ln(\mu^2/(4(1-\xi)\xi p_z^2))$ after the $\ln 4$ convention difference noted in the code. Outside, the code's $\ln|\xi/(\xi-1)|$ matches the paper's $\ln(\xi/(\xi-1))$ and $\ln(-\xi/(1-\xi))$ respectively.
- **Plus prescription**: The code restores the plus prescription by column-summing to zero, which is equivalent to the paper's $[\,\cdot\,]_{+(1)}^{D}$ with the stated definition. The domains $[1,\infty]$, $[0,1]$, $[-\infty,0]$ are reproduced.
- **Delta term**: The $\delta(1-\xi)$ term with the $\ln(\mu^2/4p_z^2)$ and the $\langle x\rangle_g/\langle x\rangle_i$ ratio is present in both.
- **Scheme correction**: The $1/|1-\xi|$ terms with the $\theta$ functions match the paper's structure.

No discrepancies found. The code reproduces Eq. (23) exactly, up to the documented $\ln 4$ convention shift in the log scale.
