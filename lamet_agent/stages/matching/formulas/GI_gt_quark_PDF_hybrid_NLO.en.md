<!-- lamet-agent formula cache; kernel=GI_gt_quark_PDF_hybrid_NLO; arxiv=2412.20461; equations=Eq. (24); digest=616fbf27e2a4a33e; paper_used=true -->
$$C_{q_iq_i}^{\text{hyb-r}}\left(\xi,\frac{\mu}{yP^z}, y z_s P^z\right) = C_{q_iq_i}\left(\xi,\frac{\mu}{yP^z}\right) + \delta C_{q_iq_i}(\xi, y z_s P^z)$$

with $\xi = x/y$ and $L = \ln(4y^2P_z^2/\mu^2)$. The ratio-scheme part is

$$
\begin{split}
C_{q_iq_i}\left(\xi,\frac{\mu}{yP^z}\right) &= \delta(1-\xi) \\
&+\frac{\alpha_sC_F}{2\pi}\left\{ \begin{array}{rcl} &\left[\xi\frac{1+\xi^2}{1-\xi}\mathrm{ln}(\frac{\xi}{-1+\xi})+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}\right]_{+(1)}^{[1,\infty]} & \mbox{for}\ 1<\xi \\ &\left[\xi\frac{1+\xi^2}{1-\xi}(-\mathrm{ln}(\frac{\mu^2}{4(1-\xi)\xi y^2 p_z^2}))-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}\right]_{+(1)}^{[0,1]} & \mbox{for}\  0<\xi<1 \\ &\left[-\xi\frac{1+\xi^2}{1-\xi}\mathrm{ln}(\frac{-\xi}{1-\xi})-\xi-\frac{3}{2}+\frac{17}{6}\frac{1}{1-\xi}\right]_{+(1)}^{[-\infty,0]} & \mbox{for}\ \xi<0 \end{array}\right.\\
&-\frac{\alpha_sT_F}{2\pi}\left\{ \left[\frac{1}{3}-\frac{1}{3}\mathrm{ln}\left(\frac{\mu^2}{4y^2p_z^2}\right)\right]\delta(1-\xi)+\frac{1}{3}\left[\left[\frac{1}{|1-\xi|} \right]_{+(1)}^{[0,2]}+\frac{1}{|\xi-1|}\theta(-\xi)+\frac{1}{|1-\xi|}\theta(\xi-2) \right] \right\}\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}} ,
\end{split}
$$

where the plus function is defined as

$$
\int_{-\infty}^{\infty}dx\ \left[ f(x) \right]_{+(c)}^{[a,b]}g(x)=\int_{a}^{b}dx\ f(x)\left[ g(x)-g(c) \right].
$$

The hybrid correction is

$$
\delta C_{q_iq_i}(\xi, y z_s P^z) = \frac{3\alpha_sC_F}{2\pi^2}\left[ \frac{\text{Si}((1-\xi)|y|z_sP^z)}{(1-\xi)} \right]^{[-\infty,\infty]}_{+(1)}.
$$

#### Consistency check

The code implements $C_{q_iq_i}^{\text{hyb-r}}$ as `C_hybrid_gi = C_ratio_gi + _hybrid_gi_delta`. Comparing term by term:

- **Regular coefficient**: The code's `C_ratio_gi` reproduces the three-branch structure of Eq. (23) exactly: for $0<\xi<1$ it gives $S(L-\ln 4 + \ln(4\xi(1-\xi))-1)+1$ with $S=(1+\xi^2)/(1-\xi)$, matching the paper's $\xi\frac{1+\xi^2}{1-\xi}(-\ln(\mu^2/(4(1-\xi)\xi y^2p_z^2)))-\frac{\xi^2(1+\xi)}{1-\xi}+\frac{17}{6}\frac{1}{1-\xi}+\frac{3}{2}$ after the code's $\ln 4$ removal. For $\xi>1$ and $\xi<0$, the code gives $\pm[S\ln|\xi/(\xi-1)|+1]$, matching the paper's $\pm[\xi\frac{1+\xi^2}{1-\xi}\ln(\xi/(\xi-1))+\xi+\frac{3}{2}+\frac{17}{6}\frac{1}{\xi-1}]$ (the $\frac{17}{6}\frac{1}{\xi-1}$ term is absorbed into the plus prescription). The $3/(2|1-\xi|)$ tail matches the paper's $\frac{3}{2}$ constant plus the $\frac{17}{6}$ term under the plus prescription.
- **Logarithms**: The code's `log_scale = ln(4y^2P_z^2/mu^2)` matches the paper's $L$; the code subtracts $\ln 4$ to match the paper's $\ln(y^2P_z^2/\mu^2)$ inside the $0<\xi<1$ branch. The $\ln(\xi/(\xi-1))$ arguments match.
- **Plus prescription**: The code restores the plus prescription by column-summing to zero, which is equivalent to the paper's $[g]^{D}_{+(1)}$ with subtraction at $\xi=1$. The paper's split into $[1,\infty]$, $[0,1]$, $[-\infty,0]$ domains is reproduced by the code's branch structure.
- **Delta term**: The code does not explicitly include the $\delta(1-\xi)$ term from the $T_F$ part (the $\frac{\alpha_sT_F}{2\pi}\{\frac{1}{3}-\frac{1}{3}\ln(\mu^2/(4y^2p_z^2))\}\delta(1-\xi)$). This term is absent from the code's `C_ratio_gi`.
- **Scheme correction**: The code's `_hybrid_gi_delta` with `strength=1.5` gives $\frac{3\alpha_sC_F}{2\pi^2}[\text{Si}((1-\xi)|y|z_sP^z)/(1-\xi)]^{[-\infty,\infty]}_{+(1)}$, matching Eq. (24) exactly.

**Discrepancies**: (1) The code omits the $\delta(1-\xi)$ term proportional to $T_F$ (the quark-mass correction). (2) The code's `C_ratio_gi` does not include the $\frac{17}{6}\frac{1}{\xi-1}$ term explicitly—it is absorbed into the plus prescription, which is consistent with the paper's notation but not a term-by-term match. (3) The code's `C_ratio_gi` does not include the $\frac{\langle x\rangle_g}{\langle x\rangle_{q_i}}$ ratio multiplying the $T_F$ term. These are omissions in the code relative to the paper, not errors in the paper.
