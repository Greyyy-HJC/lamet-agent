<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_hybrid_NLO; arxiv=2412.20461; equations=Eq. (24); digest=d2cbc5eb0db1406b; paper_used=true -->
$$C_{q_iq_i}^{\text{hyb-r}}\left(\xi,\frac{\mu}{yP^z}, y z_s P^z\right) = C_{q_iq_i}\left(\xi,\frac{\mu}{yP^z}\right) + \delta C_{q_iq_i}(\xi, y z_s P^z)$$

with $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The ratio-scheme part is

$$
\begin{aligned}
C_{q_iq_i}\left(\xi,\frac{\mu}{yP^z}\right) = &\delta(1-\xi) \\
&+\frac{\alpha_s C_F}{2\pi}\left\{
\begin{array}{ll}
\left[\xi\frac{1+\xi^2}{1-\xi}\ln\frac{\xi}{\xi-1}+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}\right]_{+(1)}^{[1,\infty]} & \xi>1 \\[2mm]
\left[\xi\frac{1+\xi^2}{1-\xi}\left(-\ln\frac{\mu^2}{4(1-\xi)\xi y^2P_z^2}\right)-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}\right]_{+(1)}^{[0,1]} & 0<\xi<1 \\[2mm]
\left[-\xi\frac{1+\xi^2}{1-\xi}\ln\frac{-\xi}{1-\xi}-\xi-\frac{3}{2}+\frac{17}{6}\frac{1}{1-\xi}\right]_{+(1)}^{[-\infty,0]} & \xi<0
\end{array}\right. \\
&-\frac{\alpha_s T_F}{2\pi}\left\{\left[\frac{1}{3}-\frac{1}{3}\ln\left(\frac{\mu^2}{4y^2P_z^2}\right)\right]\delta(1-\xi)+\frac{1}{3}\left[\left[\frac{1}{|1-\xi|}\right]_{+(1)}^{[0,2]}+\frac{1}{|\xi-1|}\theta(-\xi)+\frac{1}{|1-\xi|}\theta(\xi-2)\right]\right\}\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}},
\end{aligned}
$$

where the plus function is defined by

$$
\int_{-\infty}^{\infty}dx\ \left[ f(x) \right]_{+(c)}^{[a,b]}g(x)=\int_{a}^{b}dx\ f(x)\left[ g(x)-g(c) \right].
$$

The hybrid correction is

$$
\delta C_{q_iq_i}(\xi, y z_s P^z) = \left(-\frac{17\alpha_s C_F}{24\pi}+\frac{\alpha_s T_F}{12\pi}\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}}\right)\left[ \frac{1}{|\xi-1|}-\frac{2 \mathrm{Si}((1-\xi) y z_s P^z)}{\pi(1-\xi)} \right]^{[-\infty,\infty]}_{+(1)}.
$$

#### Consistency check

The code's `C_ratio_gi` implements the $\xi>1$, $0<\xi<1$, and $\xi<0$ branches of the ratio-scheme coefficient. Comparing term by term with the paper's Eq. (23) (which the code labels as such, and which is the basis for Eq. (24)):

- **Regular coefficient**: The code's `splitting = (1+ksi^2)/(1-ksi)` matches the paper's $\xi(1+\xi^2)/(1-\xi)$ after the code's `density` divides by $|y|$ (the $dy/|y|$ measure). The code's `+1.5/|1-ksi|` matches the paper's $+\frac{17}{6}\frac{1}{1-\xi}$ only if the paper's $\frac{17}{6}$ is a typo for $\frac{3}{2}$ — the code uses $1.5$, not $17/6$. This is a **discrepancy**: the paper has $\frac{17}{6}$ in the $0<\xi<1$ branch and $\frac{3}{2}$ in the other branches; the code uses $1.5$ everywhere.
- **Logarithms**: The code's `lamet_log = log_scale - log(4)` with `log_scale = ln(4 y^2 P_z^2/mu^2)` gives $\ln(y^2P_z^2/\mu^2)$, matching the paper's $-\ln(\mu^2/(4(1-\xi)\xi y^2P_z^2))$ after the $\ln(4\xi(1-\xi))$ term is added. The outside-branch logs $\ln(\xi/(\xi-1))$ and $\ln(-\xi/(1-\xi))$ match the code's `log_ratio` with the sign convention.
- **Plus prescription**: The code's `build_matching_matrix` restores the plus prescription by column-sum subtraction, which matches the paper's $[\,\cdot\,]_{+(1)}^{D}$ with $D$ the domain. The paper splits into three domains $[1,\infty]$, $[0,1]$, $[-\infty,0]$; the code's `C_ratio_gi` does not explicitly split but the `if eps < ksi < 1-eps` branch covers $[0,1]$ and the `else` covers the other two with the sign function. The domains are consistent.
- **Delta term**: The paper has a $\delta(1-\xi)$ term with coefficient $-\frac{\alpha_s T_F}{2\pi}\left[\frac{1}{3}-\frac{1}{3}\ln(\mu^2/(4y^2P_z^2))\right]\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}}$. The code's `C_ratio_gi` has **no delta term** — it is absent from the code entirely.
- **Hybrid correction**: The code's `_hybrid_gi_delta` with `strength=1.5` gives $\frac{3}{2}\left[-1/|1-\xi| + 2\mathrm{Si}((1-\xi)|y|z_sP_z)/(\pi(1-\xi))\right]$, which matches the paper's $\delta C_{q_iq_i}$ **only if** the paper's prefactor $\left(-\frac{17\alpha_s C_F}{24\pi}+\frac{\alpha_s T_F}{12\pi}\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}}\right)$ is replaced by $\frac{3\alpha_s C_F}{2\pi}$ — the code's `strength` is $1.5$, not the paper's combination of $17/24$ and $T_F$ terms. This is a **discrepancy**: the code's hybrid correction is the non-singlet $\bar{C}_{q_iq_i}^{hyb-r}$ from the paper's Eq. (new_match), not the singlet $\delta C_{q_iq_i}$ from Eq. (35X).

**Summary**: The code does **not** reproduce Eq. (24) of arXiv:2412.20461. It reproduces the non-singlet matching kernel $\bar{C}_{q_iq_i}^{hyb-r}$ (the paper's Eq. after `new_match`) with the $\frac{3}{2}$ strength, missing the singlet-specific $\frac{17}{6}$ coefficient, the $T_F\langle x\rangle_g/\langle x\rangle_{q_i}$ term, and the $\delta(1-\xi)$ term. The code's `C_ratio_gi` also uses $1.5$ where the paper has $\frac{17}{6}$ in the $0<\xi<1$ branch.
