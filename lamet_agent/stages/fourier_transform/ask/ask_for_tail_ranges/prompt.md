The supplied evidence describes a coordinate-space matrix element `h(z)` on a
physical z grid. Use exact nonnegative measured coordinates from the evidence;
the evidence may already contain the runtime-completed signed-z grid, while
tail fitting uses strictly positive z points. Real and imaginary entries carry
pointwise central values and uncertainties. The lattice spacing identifies the
measured grid resolution, and the momentum sets the scale of expected
oscillations. `zmin_fm` is the lower measured separation constraining the tail
fit; `zmax_fm` is the largest measured separation retained in that fit, not the
later extension endpoint.

Recommend compact candidate lists for whichever Fourier tail-fit boundaries are
requested by the response schema. Use only exact coordinates present in the
supplied positive-z grid. Follow this selection policy in order of priority:

1. Keep zmax_fm as large as the usable measured positive-z coverage permits,
   preferably the largest valid grid point. Include essentially all effective
   data points unless the supplied real/imaginary data give a clear numerical
   reason to exclude a final-z point; do not reduce zmax merely because the
   error grows at large z. The following are only evidence-based heuristics for
   an unsupported final-z point, not runtime rejection rules:

   - the uncertainty is exceptionally large, as a concrete diagnostic roughly
     satisfying sigma >= 2 * abs(central_value) (when the central value is near
     zero, also require a genuinely large absolute uncertainty rather than
     using this ratio alone);
   - there is an isolated singular-looking point, spike, discontinuity, or
     unsupported sign/amplitude jump relative to neighboring z points;
   - the tail is coherently and significantly nonzero over successive points
     with a large amplitude, rather than merely fluctuating within the error
     band around zero.

   Oscillations are not by themselves a reason to reject a point: an oscillating tail is
   acceptable when its envelope is decreasing and it is converging toward
   zero. In contrast, an oscillation that is consistent with zero followed by
   a later, large-amplitude excursion away from zero without a plausible local
   trend is pathological; place zmax before that unsupported excursion.
2. Use raw zmin_fm as the main scan direction. Start at the first available grid
   point at or just above 0.5 fm, then provide progressively larger grid points
   so that the tail fit can test when chi2/dof stops improving significantly.
   Recommend about 4--5 successive zmin grid points when the input coverage
   allows it, rather than only one or two. Never recommend raw zmin_fm below
   0.5 fm. Runtime later applies `tail_window_step_offset * a_s`; effective
   zmin values must remain nonnegative and lie on the input grid. Authored
   runtime ranges may include zero even though this recommendation does not.
3. Prefer candidate pairs with the same large zmax and ordered increasing
   zmin values. Recommend 2--3 zmax grid points when the input coverage allows
   it, with the largest valid zmax first and carrying the largest-zmax choice
   as the primary candidate. A shorter zmax may be included only as a clearly
   justified fallback when the largest-z data are visibly unusable; it must not
   be the default recommendation.

Every Cartesian-product pair formed from the returned lists must satisfy
`zmin_fm < zmax_fm`. Pairs with too few strictly positive measured points for
the selected model/order/component are skipped deterministically; do not assume
one universal point-count threshold. More flexible expansions generally need
more data support, so avoid minimal windows chosen only to move deeper into the
apparent long-distance region. If a shorter zmax would leave a high-zmin
candidate visibly underconstrained, omit that pair.

The returned lists are recommendations, not the selection result. Runtime
enumerates model x zmin x zmax in authored order, truncates that range prefix
by `max_schemes`, selects a successful center fit with `Q >= q_min` and largest
`logGBF`, and otherwise falls back to largest `Q`. `chi2/dof` is a diagnostic,
not an independent selection criterion. Keep the lists compact, with the grid
point nearest 0.5 fm and 4--5 successive zmin points when available.

Values under `fixed_parameters` are authored and must not be changed on the
initial attempt. On a retry, `previous_attempts` may contain Q, chi2, degrees of
freedom, chi2/dof, logGBF, fit success, or a numerical error. Exclude failed
configurations, use Q and logGBF according to the runtime rule above, and treat
chi2/dof as a diagnostic. Make conservative range changes while preserving the
large-zmax-first policy, and do not alter `scheme_scan` or its `q_min` fallback.

Use only the supplied numerical evidence; do not treat growing uncertainty alone
as proof that physically plausible long-distance data should be discarded.
