<!-- lamet-agent formula cache; kernel=da_gi_gzg5_hybrid_lrr_nlo; arxiv=2305.05212; equations=Eqs. (12)-(17); digest=e9260ed2b54685c5; paper_used=true -->
$$C_{gzg5}^{{\rm hybrid,LRR}}(x,y,\mu,P_z) = \frac{1}{2}V_{qq,p}(x,y,\mu,P_z) + \frac{3\,{\rm Si}\!\left(z_s P_z (y-x)\right)}{\pi (y-x)}$$

where $V_{qq,p}$ is the $\gamma^z\gamma_5$ DA matching kernel of Eq. (4.15) of arXiv:2212.14415,

$$V_{qq,p}(x,y,\mu,P_z) = V_{qq,h}(x,y,\mu,P_z) + 2\left[\frac{|x|}{y} + \frac{|1-x|}{1-y} + \frac{|x-y|}{(y-1)y}\right],$$

with

$$V_{qq,h}(x,y,\mu,P_z) = \frac{|x|}{y}\left(l_x-1\right) + \frac{|1-x|}{1-y}\left(l_{1-x}-1\right) + \frac{|x-y|}{y(y-1)}\left(l_{x-y}-1\right) + V_{qq,t}(x,y,\mu,P_z),$$

and

$$V_{qq,t}(x,y,\mu,P_z) = \frac{|x|}{y(y-x)}\left(l_x-1\right) + \frac{|1-x|}{(1-y)(x-y)}\left(l_{1-x}-1\right) + \frac{x+y-2xy}{|x-y|\,y(1-y)}\left(l_{x-y}-1\right).$$

The logarithms are defined as in Eq. (4.16),

$$l_v = \ln\!\left(\frac{4P_z^2 v^2}{\mu^2}\right), \qquad v = x,\;1-x,\;x-y.$$

The full kernel is the fixed-order matrix $M_{\rm fix}$ (discretized from the density above with the plus prescription at $x=y$, each $y$-column integrated to zero over $x\in[0,1]$) improved by the leading renormalon resummation,

$$M_{\rm LRR} = \left(M_{\rm fix} + r_0\, M_{C_z}\right) \exp\!\left(-M_{C_z}\, r_{\rm sumPV}\right),$$

where $M_{C_z}$ is the plus-prescribed matrix of the renormalon shape $C_z(\xi)$ (Eq. (17) of arXiv:2305.05212, with $\xi=x/y$, $p_z=|y|P_z$, $z_s=z_s$),

$$C_z(\xi) = \frac{e^{-\epsilon_m z_s} p_z (1+\epsilon_m z_s+\epsilon_m^2 z_s^2)}{\epsilon_m^2\pi} + \frac{1}{\pi}\left(\frac{e^{-\epsilon_m z_s} z_s \sin[(1-\xi)p_z z_s]}{1-\xi} + \frac{e^{-\epsilon_m z_s} p_z}{(\epsilon_m^2+p_z^2(1-\xi)^2)^2}\Big((\epsilon_m^2-p_z^2(1-\xi)^2+\epsilon_m^3 z_s+\epsilon_m p_z^2(1-\xi)^2 z_s)\cos[(1-\xi)p_z z_s] - p_z(1-\xi)(2\epsilon_m+p_z^2(1-\xi)^2 z_s+\epsilon_m^2 z_s)\sin[(1-\xi)p_z z_s]\Big)\right),$$

with the plus prescription $[f(x)]_+ = f(x) - \delta(1-x)\int_0^1 f(\nu)d\nu$ applied at $\xi=1$. The scalar coefficients are $r_0 = N_m |z\mu| \alpha_s$ (from Eq. (12) at $n=0$) and $r_{\rm sumPV} = N_m |z\mu| e^w (-2\pi/\beta_0)\,{\rm Re}\!\left[E_{1+b}(w) + c_1 E_b(w) + c_2 E_{-1+b}(w)\right]$ with $w = -2\pi/(\alpha_s\beta_0)$, $N_m(n_f=3)=0.575$, $\beta_0=9$, $b=\beta_1/(2\beta_0^2)$, and $c_1,c_2$ from the sub-asymptotic corrections of arXiv:hep-ph/0105008. The regularization parameter is $\epsilon_m = 0.005$ GeV.

#### Consistency check

The code reproduces Eqs. (12)–(17) of arXiv:2305.05212 exactly. The fixed-order part matches Eq. (4.15) of arXiv:2212.14415 (the paper's reference for the $\gamma^z\gamma_5$ DA kernel), including the logarithms $l_v = \ln(4P_z^2 v^2/\mu^2)$ and the plus prescription at $x=y$ with column-sum-to-zero over $x\in[0,1]$. The renormalon resummation follows Eqs. (12)–(17) verbatim: $r_n$ from Eq. (12), the PV Borel integral of Eq. (13) evaluated as the exponential-integral sum, and the Fourier transform of Eq. (17) with the $\epsilon_m$ regulator. The matrix exponential $\exp(-M_{C_z} r_{\rm sumPV})$ implements the all-order resummation of the leading renormalon series, with $r_0$ subtracting the $\mathcal{O}(\alpha_s)$ term to avoid double counting. No discrepancies found.

