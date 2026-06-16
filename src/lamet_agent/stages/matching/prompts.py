"""Prompt text for perturbative matching stage."""

# This text is read by core/prompting.py (getattr STAGE_PROMPT) and folded into the
# stage instruction sent to the LLM. It tells the agent the stage goal and the
# order in which to call the tools.
STAGE_PROMPT = """
Convert the quasi-PDF produced by the Fourier stage into the light-cone PDF by
applying an NLO perturbative matching kernel and propagating gvar uncertainties
through the convolution.

Each operator uses its own kernel, selected by kernel_id. The quasi-PDF is read
from the Fourier-stage artifact on disk because every stage starts with a fresh
store.

0. If the stage input issues list reports a missing kernel_id, missing pz_gev,
   or unknown kernel_id, ask the user instead of guessing.
1. list_kernels to see the registered kernel_ids, then choose the one that
   matches this operator.
2. load_quasi_pdf on metadata.matching.quasi_input. It auto-detects the Fourier
   EnsembleData npz and takes the real part with stat (+) sys error; pass
   component='im' only if the quasi-PDF lives in the imaginary channel.
3. build_matching_kernel(kernel_id=..., pz_gev=..., mu=2.0). Use the same pz_gev
   as the Fourier stage. If the x grid includes 0, ask the user to regenerate
   the quasi-PDF on a zero-avoiding grid because the kernel uses xi=x/y.
4. apply_matching.
5. plot_matched_pdf to compare quasi vs light-cone and write the artifact PDF.
   metadata.matching.plot may set xlim/ylim.
6. finish, reporting the chosen kernel_id, pz_gev, mu, component, number of x
   points, and comparison PDF path.
""".strip()
