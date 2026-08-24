Use this terminal tool only when the manifest supplies `scan`. It fits each
authored z range at the ensemble center using the first order and prior width.
Among successful fits above `q_min`, the largest logGBF selects the range; when
none passes, the successful fit with the largest Q selects it. The tool then
fits every resample for the allowed LA/NLA and prior-width models on that fixed
range and applies the same per-sample choice or evidence averaging as the
original workflow. One shared process pool handles those sample fits and
transforms. The coordinate-space diagnostics show the complete input data, but
draw the selected extrapolation only from `z_min_fm` through the extrapolation
endpoint. Their horizontal coordinate is the dimensionless Ioffe time
`lambda = z P^z / (hbar c)` used by the original plots; dashed lines mark the
selected fit window after the same conversion.
