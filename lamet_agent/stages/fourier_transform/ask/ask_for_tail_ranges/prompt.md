Recommend compact candidate lists for the missing Fourier tail-fit boundaries.
Use only coordinates present in the supplied positive-z grid. Follow this
selection policy in order of priority:

1. Keep zmax_fm as large as the usable measured positive-z coverage permits,
   preferably the largest valid grid point. Include essentially all effective
   data points unless the supplied real/imaginary data show that the
   long-distance matrix element behavior is clearly pathological; do not reduce
   zmax merely because the error grows at large z. Treat the long-distance
   behavior as clearly pathological only when one or more of the following
   occurs in the final-z region, considering real and imaginary parts
   separately:

   - the uncertainty is exceptionally large, as a concrete diagnostic roughly
     satisfying sigma >= 2 * abs(central_value) (when the central value is near
     zero, also require a genuinely large absolute uncertainty rather than
     using this ratio alone);
   - there is an isolated singular-looking point, spike, discontinuity, or
     unsupported sign/amplitude jump relative to neighboring z points;
   - the tail is coherently and significantly nonzero over successive points
     with a large amplitude, rather than merely fluctuating within the error
     band around zero.

   Oscillations are not by themselves pathological: an oscillating tail is
   acceptable when its envelope is decreasing and it is converging toward
   zero. In contrast, an oscillation that is consistent with zero followed by
   a later, large-amplitude excursion away from zero without a plausible local
   trend is pathological; place zmax before that unsupported excursion.
2. Use zmin_fm as the main scan direction. Start at the first available grid
   point at or just above 0.5 fm, then provide progressively larger grid points
   so that the tail fit can test when chi2/dof stops improving significantly.
   Recommend about 4--5 successive zmin grid points when the input coverage
   allows it, rather than only one or two. Never recommend zmin_fm below 0.5 fm.
3. Prefer candidate pairs with the same large zmax and ordered increasing
   zmin values. Recommend 2--3 zmax grid points when the input coverage allows
   it, with the largest valid zmax first and carrying the largest-zmax choice
   as the primary candidate. A shorter zmax may be included only as a clearly
   justified fallback when the largest-z data are visibly unusable; it must not
   be the default recommendation.

Each valid pair must satisfy zmin_fm < zmax_fm, retain enough points for the
configured tail models, and use exact input-grid coordinates. Before returning
the lists, count the positive-z observations in every candidate pair and check
the number of independent real/imag observations against the number of fit
parameters for every configured order/model. Do not recommend a pair that
cannot support the configured model; for the present light-light DA NLA tail,
five parameters are fitted from two channels, so at least three positive-z
points are required. Because the implementation forms the Cartesian product
of the two lists, this check applies to every ordered zmin/zmax pair that will
be formed, not only to the largest-zmax pair. If a shorter zmax would make a
large-zmin candidate underdetermined, omit that shorter zmax or omit the
incompatible high-zmin candidate. Do not rely on the runtime scan silently
discarding underdetermined pairs. Return compact lists, but include the grid
point nearest 0.5 fm and 4--5 successive zmin points when available. Fixed
parameters are authored values and must not be changed on the initial attempt.
When previous attempts are supplied, make a conservative runtime adjustment
using their Q and chi2 diagnostics while preserving the large-zmax-first
policy; do not alter scheme_scan.
