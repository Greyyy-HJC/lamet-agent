"""Prompt text for perturbative matching stage."""

# This text is read by core/prompting.py (getattr STAGE_PROMPT) and folded into the
# stage instruction sent to the LLM. It tells the agent the stage goal and the
# order in which to call the tools.
STAGE_PROMPT = """
Goal: convert the quasi-PDF produced by the Fourier stage into the light-cone PDF by applying an NLO perturbative matching kernel, propagating gvar uncertainties through the convolution.

Each operator uses its own kernel, selected by kernel_id. The quasi-PDF is read from the Fourier-stage artifact on disk because every stage starts with a fresh store. Emit one action at a time and reference earlier outputs by their 'out' keys.

0. If the stage input issues list reports a missing kernel_id, a missing pz_gev, or an unknown kernel_id, do not guess. Emit request_user_input and ask the user, summarizing the registered kernel_ids and the meaning of pz_gev/mu.

Steps:
1. list_kernels to see the registered kernel_ids, then choose the one that matches this operator (e.g. unpolarized_gT for the unpolarized gamma^t PDF).

2. load_quasi_pdf on metadata.matching.quasi_input (default the Fourier artifacts/quasi_pdf.npz). It auto-detects the Fourier EnsembleData npz and takes the real part with stat (+) sys error; pass component='im' only if the quasi-PDF lives in the imaginary channel. Stores x_ls and quasi_gv.

3. build_matching_kernel(kernel_id=..., pz_gev=..., mu=2.0) -> kernel_matrix. Use the same pz_gev as the Fourier stage. The x grid must avoid x=0 because the kernel uses xi=x/y; if the grid includes 0, ask the user to regenerate
   the quasi-PDF on a zero-avoiding grid (e.g. an even number of points).
   
4. apply_matching -> lightcone_gv = kernel_matrix @ quasi_gv.

5. plot_matched_pdf to compare quasi vs light-cone and write the artifact PDF.

6. finish, reporting the chosen kernel_id, pz_gev, mu, component, the number of x points, and the comparison PDF path.

Use only the listed tools. Do not write numerical code in the model response.
""".strip()
