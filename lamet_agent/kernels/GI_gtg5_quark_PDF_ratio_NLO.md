<!-- lamet-agent formula cache; kernel=GI_gtg5_quark_PDF_ratio_NLO; arxiv=2412.20461; equations=Eq. (23); digest=18af8b8b1e4048c8; paper_used=true -->
$$C_{q_iq_i}^{\mathrm{ratio}}(\xi,L)=\delta(1-\xi)+\frac{\alpha_s C_F}{2\pi}\left\{ \begin{array}{rcl} &\left[\xi\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}\right]_{+(1)}^{[1,\infty]} & \mbox{for}\ 1<\xi \\ &\left[\xi\frac{1+\xi^2}{1-\xi}\left(-\ln\frac{\mu^2}{4(1-\xi)\xi p_z^2}\right)-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}\right]_{+(1)}^{[0,1]} & \mbox{for}\  0<\xi<1 \\ &\left[-\xi\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-\xi-\frac{3}{2}+\frac{17}{6}\frac{1}{1-\xi}\right]_{+(1)}^{[-\infty,0]} & \mbox{for}\ \xi<0 \end{array}\right.$$

$$-\frac{\alpha_s T_F}{2\pi}\left\{ \left[\frac{1}{3}-\frac{1}{3}\ln\left(\frac{\mu^2}{4p_z^2}\right)\right]\delta(1-\xi)+\frac{1}{3}\left[\left[\frac{1}{|1-\xi|} \right]_{+(1)}^{[0,2]}+\frac{1}{|\xi-1|}\theta(-\xi)+\frac{1}{|1-\xi|}\theta(\xi-2) \right] \right\}\frac{\langle x\rangle_g}{\langle x\rangle_i},$$

with the plus function defined as
$$\int_{-\infty}^{\infty}dx\ \left[ f(x) \right]_{+(c)}^{[a,b]}g(x)=\int_{a}^{b}dx\ f(x)\left[ g(x)-g(c) \right].$$

Here $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, $C_F=(N^2-1)/(2N)$, $T_F=1/2$, and the $\delta(1-\xi)$ term is the scheme-specific correction from the ratio scheme's denominator.

#### Consistency check

The code's `C_ratio_gi` implements the $\xi>1$, $0<\xi<1$, and $\xi<0$ branches of the first curly bracket in Eq. (23) exactly: the splitting function $S=(1+\xi^2)/(1-\xi)$, the log arguments $\ln[\xi/(\xi-1)]$ and $\ln[-\xi/(1-\xi)]$, the constants $+1$ (inside $[0,1]$) and $\pm1$ (outside), and the $3/[2(1-\xi)]$ tail all match. The code's `log_scale` is $\ln(4y^2P_z^2/\mu^2)$, and it subtracts $\ln 4$ to recover the paper's $\ln(y^2P_z^2/\mu^2)$ inside the $[0,1]$ branch — consistent. The plus prescription is restored by the column-sum in `build_matching_matrix`, matching the paper's $[\,\cdot\,]_{+(1)}^{D}$ with $D=[1,\infty]$, $[0,1]$, $[-\infty,0]$. However, the code omits the entire second curly bracket: the $\delta(1-\xi)$ term with $\ln(\mu^2/4p_z^2)$ and the $\frac{1}{3}[[1/|1-\xi|]_{+(1)}^{[0,2]}+\cdots]$ term proportional to $\langle x\rangle_g/\langle x\rangle_i$. This is a real discrepancy — the code implements only the $C_F$ part of Eq. (23), not the $T_F$ mixing term.

