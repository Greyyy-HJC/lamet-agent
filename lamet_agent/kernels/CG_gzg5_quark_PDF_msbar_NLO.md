<!-- lamet-agent formula cache; kernel=CG_gzg5_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.15); digest=2ee72e173eaea368; paper_used=true -->
$$C_{\gamma^z\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) + 2(1-\xi)_{+(1)}^{[0,1]} + \delta(1-\xi)\,,$$
with
$$C_{\gamma^t(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{(1)}_r\big(\xi,{\mu\over p^z}\big) + {1\over 2|1-\xi|} + {1\over2}\delta(1-\xi) \left[ 1 + \ln { 4p_z^2\over\mu^2} - \int_0^2 d\xi' {1\over |1-\xi'|}\right]\,,$$
where
$$C^{(1)}_r\big(\xi,{\mu\over p^z}\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}$$
and the plus prescription is defined as
$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

#### Consistency check

The code implements $C_{\gamma^z\gamma_5(1)}^{\overline{\mathrm{MS}}}$ as $C_{\gamma^t(1)}^{\overline{\mathrm{MS}}} + 2(1-\xi)$ on $0<\xi<1$ plus a $\delta(1-\xi)$ term, matching the paper’s structure. The regular coefficient $C^{(1)}_r$ is reproduced exactly: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed logarithms, the $\text{sgn}(\xi)$ term, the arctan/arctanh branch (with the analytic limit at $\xi=1/2$), and the $-3/(2|1-\xi|)$ term. The plus-prescription domains match: $[0,1]$ for the splitting piece and $(-\infty,\infty)$ for the remainder, with the $0.5/|1-\xi|$ term restricted to $[0,2]$ in the subtraction. The diagonal $\delta(1-\xi)$ coefficient is $0.5(1+\ln(4p_z^2/\mu^2)) - 0.5\int_0^2 d\xi'\,1/|1-\xi'| + 1$, where the final $+1$ is the extra $\delta(1-\xi)$ from the $\gamma^z$ channel. No discrepancies found.

