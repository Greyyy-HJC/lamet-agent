<!-- lamet-agent formula cache; kernel=quark_pdf_cg_gz_ratio_nlo; arxiv=2602.11283; equations=Eq. (2.16) with the gamma^z shift of Eq. (2.15); digest=cf4ae0940b07c996; paper_used=true -->
$$C_{\gamma^z(1)}^{\mathrm{ratio}}\big(\xi,L\big) = C^{(1)}_r\big(\xi,L\big) + 2(1-\xi) \quad \text{for } 0<\xi<1,$$

where $\xi=x/y$, $L=\ln(4y^2P_z^2/\mu^2)$, and the regular coefficient is

$$C^{(1)}_r\big(\xi,L\big) = \left[\frac{1+\xi^2}{1-\xi}\ln{4p_z^2\over\mu^2} + \xi - 1\right]_{+(1)}^{[0,1]} + \Bigg\{\frac{1+\xi^2}{1-\xi} \Big[\text{sgn}(\xi)\ln|\xi| + \text{sgn}(1-\xi)\ln|1-\xi|\Big] + \text{sgn}(\xi) + {3 \xi-1\over \xi-1} \frac{\tan^{-1}\left(\sqrt{1-2 \xi}/|\xi|\right)}{\sqrt{1-2 \xi}} - {3\over 2|1-\xi|} \Bigg\}_{+(1)}^{(-\infty, \infty)},$$

with the plus functions defined as

$$\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x').$$

The $\gamma^z$ shift $2(1-\xi)$ is taken from Eq. (2.15) of the paper, which pairs it with a $+\delta(1-\xi)$ term in the $\overline{\mathrm{MS}}$ scheme. In the ratio scheme, that delta is omitted because the scheme divides out the normalization; the shift remains plus-prescribed by the shared discretization. The arctan/arctanh branch is chosen by where $\xi$ sits relative to $1/2$: for $\xi<1/2$ use $\arctan(\sqrt{1-2\xi}/|\xi|)/\sqrt{1-2\xi}$, for $\xi>1/2$ use $\arctanh(\sqrt{2\xi-1}/|\xi|)/\sqrt{2\xi-1}$, and at $\xi=1/2$ the analytic limit is $1/|\xi|$.

#### Consistency check

The code reproduces Eq. (2.16) with the $\gamma^z$ shift of Eq. (2.15) of arXiv:2602.11283, with the following notes:

- **Regular coefficient**: The code's `C_ratio` matches the paper's $C^{(1)}_r$ term by term: the splitting-function piece $(1+\xi^2)/(1-\xi)\ln(4p_z^2/\mu^2)+\xi-1$ on $[0,1]$, the signed-log term, $\text{sgn}(\xi)$, the arctan/arctanh piece, and $-3/(2|1-\xi|)$. All arguments inside logs and the arctan/arctanh match exactly.
- **Plus prescription**: The code uses the paper's exact notation $[g]^D_{+(1)}$ with domains $[0,1]$ and $(-\infty,\infty)$, and the paper's definition $\left[g(x)\right]^D_{+(x_0)} = g(x) - \delta(x-x_0)\int_D dx'\ g(x')$ is reproduced by the column-sum prescription.
- **$\gamma^z$ shift**: The code adds $2(1-\xi)$ for $0<\xi<1$, matching Eq. (2.15). The paper's $+\delta(1-\xi)$ term is deliberately omitted in the ratio scheme, as documented in the code comment; this is a scheme-specific choice, not a discrepancy.
- **No discrepancies found**: All signs, factors, log arguments, and the arctan/arctanh branch structure agree between code and paper.

