"""Prompt text for Fourier-transform stage."""

STAGE_PROMPT = """
Run asymptotic extrapolation and Fourier transform on coordinate-space
matrix-element samples.

Do this by emitting one action at a time:
0. If the stage input issues list reports missing Fourier fields, do not guess
   them and do not call tools yet. Emit request_user_input and ask the user for
   the missing fields, summarizing the listed choices and the physical meaning
   of each option.
1. If manifest.metadata.fourier_input is provided, call
   load_renormalized_matrix_element_samples on that path. Pass input_format='npz'
   for NPZ files or input_format='h5' for HDF5 files. For HDF5 inputs, pass
   h5_group when the desired group cannot be inferred from the file name.
   If metadata.fourier specifies coord_key, re_key, im_key, or resample_mode,
   pass those values to the loader.
   If an upstream stage already produced store['matrix_element_data'], skip the
   loader and run the transform directly on that EnsembleData.
2. run_fourier_transform with explicit k_grid (list or compact {start, stop, num/step}),
   method, order, observable, coord_unit, and pz_gev/a_fm when needed.
   If metadata.fourier specifies im_flip_for_ft, Lambda0,
   posterior_prior_error_scale, fit_error_mode, or save_path, pass those values
   to run_fourier_transform.
   If the manifest gives scheme_scan, pass it through. If it omits any of
   zmin_values/zmax_values/min_width/z_ext_max/smooth, the tool will fill the
   missing scan values by choosing large stable zmax values before visible
   jitter or sharply growing error bars. It then fixes each zmax, scans zmin
   from smaller to larger coordinates, and chooses zmin candidates where the
   selected method/order/observable tail fit has stable chi2/dof and Q.
   Missing z_ext_max defaults to the largest input-data lambda plus 8,
   converted back to the input coordinate unit. Missing y_range,
   roughness_weight, and model_average default to [-2,2], 1.0, and true.
   Use observable to select the large-distance form: pion_quark_quasi_pdf,
   nucleon_quark_unpolarized_quasi_pdf, nucleon_quark_transversity_quasi_pdf,
   pion_gluon_quasi_pdf, nucleon_gluon_quasi_pdf, meson_quasi_da,
   pion_quark_quasi_gpd, or nucleon_quark_quasi_gpd.
   For GPD observables, pass pz_prime_gev when P'^z differs from P^z.
   order can be 'LA' or 'NLA'.
   Lambda0 optionally sets the lower bound of the fitted large-distance
   exponential scale Lambda; default is 0.1 GeV for physical-z inputs.
   posterior_prior_error_scale inflates the sample-average fit parameter
   errors when using that posterior as a weak prior for bootstrap/jackknife
   sample fits; default is 3.0.
   fit_error_mode controls the tail-fit data covariance: 'diagonal' uses
   pointwise standard deviations, while 'covariance' uses the full covariance
   estimated from bootstrap/jackknife samples. The default is 'diagonal'.
   Fit windows must include at least as many coordinate points as the selected
   model has parameters.
   Prefer scheme_scan when the manifest provides it, so the tool can score and
   model-average the fit-range choices numerically.
3. summarize_fourier_result.
4. plot_fourier_result.
5. plot_fourier_extension_quality_result for the best weighted scheme unless
   the user asks for a different scheme. This writes separate real- and
   imaginary-part extension plots.
6. finish, reporting the NPZ artifact path, plot paths, best scheme, scheme
   weights, fit chi2/dof, roughness scores, fit failure counts, and the
   statistical/systematic uncertainty arrays.

Use only the listed tools. Do not write numerical code in the model response.
""".strip()
