# Fourier transform

Inspect signed physical `z` coordinates and derive the reference symmetry,
projection, normalization, and tail family from upstream provenance before
fitting the long-distance tail. Fit one allowed tail candidate at a time, connect it to
the data with the authored smoothing prescription, and transform every sample on
the explicit dimensionless `quasi_y_ls`. The terminal result must preserve sample
order and record units, convention, tail model, and inherited identity.
When the effective parameters contain `scheme_scan`, inspect the input once and use
`run_fourier_scan`; that terminal tool first scans authored ranges with center
fits, selects the range using the original Fourier-stage rule, and only then
fits every resample for the allowed LA/NLA and prior-width models. One shared
sample-process pool is reused after range selection. Do not also fit individual
tail candidates.
