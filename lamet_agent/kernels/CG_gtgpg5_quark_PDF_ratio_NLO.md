<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_ratio_NLO; arxiv=2602.11283; equations=Eq. (2.18); digest=963add6c2825829b; paper_used=true -->
$$C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,$$

with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$, the ratio-scheme coefficient is

$$
C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}  + \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.
$$

Here the plus functions on a domain $D$ are defined as

$$
\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,,
$$

with the subtraction point $x_0=1$ (i.e. $+(1)$). The arctan branch applies for $\xi<1/2$ and the arctanh branch for $\xi>1/2$, with the analytic limit at $\xi=1/2$. There is no additional scheme-specific finite correction: for the transversity operator, $\overline{\mathrm{MS}}$, ratio, and hybrid schemes all share this same coefficient (Eqs. 2.17, 2.21 of the paper).

#### Consistency check

The code's `C_ratio_perp` reproduces Eq. (2.18) term by term: the splitting function $2\xi/(1-\xi)$ multiplying $L$ in the $[0,1]$ plus-bracket; the signed-log combination $\text{sgn}(\xi)\ln|\xi|+\text{sgn}(1-\xi)\ln|1-\xi|$ with the same prefactor in the $(-\infty,\infty)$ bracket; the shared arctan/arctanh piece $(3\xi-1)/(\xi-1)\cdot\tan^{-1}(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$; and the $-1/|1-\xi|$ tail. The plus prescription is implemented as the column-sum subtraction at $\xi=1$ over the full domain, matching the paper's $[g]^D_{+(1)}$ definition. No discrepancies found.

