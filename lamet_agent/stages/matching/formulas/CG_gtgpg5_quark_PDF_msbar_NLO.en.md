<!-- lamet-agent formula cache; kernel=CG_gtgpg5_quark_PDF_msbar_NLO; arxiv=2602.11283; equations=Eq. (2.17); digest=3c7d7add79ec2020; paper_used=true -->
$$C_{\gamma^t\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C_{\gamma^z\gamma_\perp^\alpha\gamma_5(1)}^{\overline{\mathrm{MS}}}\big(\xi,{\mu\over p^z}\big) = C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big) \,,$$

where, with $\xi=x/y$ and $L=\ln(4y^2P_z^2/\mu^2)$,

$$C^{\perp (1)}_r\big(\xi,{\mu\over p^z}\big)= \left[{2\xi \over 1-\xi}\ln{4p_z^2\over\mu^2} \right]_{+(1)}^{[0,1]}  + \Bigg\{{2\xi \over 1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] \nn$$
$$\qquad + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {1\over |1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)}\,.$$

The plus functions on a domain $D$ are defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')\,.$$

The arctangent branch is chosen by the position of $\xi$ relative to $1/2$: for $\xi<1/2$ the term is $\frac{3\xi-1}{\xi-1}\frac{\arctan(\sqrt{1-2\xi}/|\xi|)}{\sqrt{1-2\xi}}$, for $\xi>1/2$ it is $\frac{3\xi-1}{\xi-1}\frac{\arctanh(\sqrt{2\xi-1}/|\xi|)}{\sqrt{2\xi-1}}$, and at $\xi=1/2$ the analytic limit $\frac{3\xi-1}{\xi-1}\frac{1}{|\xi|}$ is taken. There is no scheme-specific finite correction beyond the plus prescription, and no $\delta(1-\xi)$ term beyond what the plus prescription generates.

#### Consistency check

The code's `C_ratio_perp` reproduces Eq. (2.17) of arXiv:2602.11283 term by term: the splitting function $2\xi/(1-\xi)$ multiplying both the log $L$ and the signed-log combination, the arctan/arctanh branch with the correct prefactor $(3\xi-1)/(\xi-1)$, the $-1/|1-\xi|$ tail, and the plus prescription split into $[0,1]$ and $(-\infty,\infty)$ domains with subtraction at $\xi=1$. The code's `_atan_piece` matches the paper's branch structure exactly, including the analytic limit at $\xi=1/2$. The code's `_pdf_density` correctly implements the $dy/|y|$ measure of Eq. (2.17)'s factorization. No discrepancies were found between the code and the paper for this coefficient.
