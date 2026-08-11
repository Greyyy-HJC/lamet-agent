# PROJECT_LOG

## 2026-05-31

- Initialized a minimal non-`temp` project scaffold from `TODO.md`.
- Added essential runtime placeholders under `src/lamet_agent/`.
- Added minimal configs/examples/tests/docs placeholders.
- Rewrote `README.md` to align with `PLAN.md`.
- Updated manifest contract to use correlator inputs + Python kernel functions.
- Added kernel callable resolution and validation in CLI `validate` and `run`.
- Added fake-data-oriented manifest example and validation tests.
- Simplified package structure to a minimal flat layout (`cli`, `manifest`, `kernels`).
- Removed unnecessary placeholder modules and removed `docs/`, `configs/`, and `runs/`.
- Kept fake-data generation at `examples/fake_data/generate_fake_data.py`.
- Added `prompts.py`, `skills.py`, and `agent.py` for minimal staged agent runtime.
- Wired CLI `run` command to execute `run_agent` with resumable stage loop.
- Documented per-file responsibilities in `README.md`.

## 2026-06-01

- Refactored runtime layout to `core/` plus `stages/*` packages.
- Added five stage packages: `correlator`, `renorm`, `fourier`, `matching`, `extrapolation`.
- Added per-stage `prompts.py`, `skills.py`, and `functions.py` placeholders.
- Moved prompt assembly and stage routing into `src/lamet_agent/core/`.
- Rewired `agent.py` and `cli.py` to use the new `core` API.
- Removed legacy flat `src/lamet_agent/prompts.py` and `src/lamet_agent/skills.py`.
- Updated README structure/responsibilities and added an English agent workflow section.
- Added unit coverage for stage routing and stage prompt resolution.

## 2026-06-03

- Implemented the `correlator_analysis` stage as the first worked example.
- Added `core/plotting.py`: self-contained LaMETLat-style plotting with a 2pt
  fit-on-data figure (C2pt + effective mass with model-average band).
- Rewrote `stages/correlator/functions.py` with copied LaMETLat numerics
  (read_pt2, bootstrap/jackknife resampling, pt2 ground-state fit) plus new
  `scan_tmin` and logGBF-weighted `model_average`; exposed a `STAGE_TOOLS`
  registry. Added an `svdcut` (default 1e-2) to stabilize the correlated 2pt fit.
- Added `STAGE_SKILL` strategy text and `tool_catalog()` to the stage `skills.py`
  and expanded the stage `prompts.py` with the call_tool/finish action protocol.
- Added `core/tools.py` and reworked `agent.py` into a pluggable responder
  (`mock`/`external`) with an intra-stage tool-execution loop; `core/prompting.py`
  now injects skill, tool catalog, and tool observations.
- Added `matplotlib` to the `analysis` extras; ignored `runs/` outputs.
- Validated end-to-end on `examples/fake_data/data/fake_2pt.h5`: recovers
  E0 = 0.4501(12) (true 0.45) via the wired loop and writes fit-on-data PDFs.
- Replaced the `max_steps` stage cap with an explicit `stages` selection
  (`--stages` CLI option); running a later stage standalone now surfaces missing
  inputs per stage via `input_issues`. Added `core/tools.validate_stage_inputs`.

## 2026-06-03

- Added a `deepseek` responder (`--model deepseek`): each step posts the full
  stage prompt to the DeepSeek chat-completions API in JSON mode (stdlib
  `urllib`, no new deps) and parses one action, so a real LLM drives the loop and
  sees tool observations before deciding the next action. The key is read from
  `--api-key-file` (default `api.key`, gitignored) or `DEEPSEEK_API_KEY`.
- Removed the interim `codex exec` responder and its CLI surface to keep the
  responder set minimal (`mock`/`external`/`deepseek`).

## 2026-06-03 (correlator agent freedom)

- Replaced `scan_tmin` with `fit_window` (appendable single-window fits) so the
  agent can explore arbitrary `[tmin, tmax)` ranges in the first half (`t <
  Lt/2`); soft warnings when windows extend past `Lt//2`.
- Extended `model_average` and `plot_fit_on_data` with `window_indices` subset
  selection; plots read `E0_avg` for the final result.
- Updated `core/plotting.py`: per-window colored fit bands on C2pt and meff,
  plus a horizontal model-averaged E0 band on meff.
- Refreshed correlator `STAGE_SKILL` / `STAGE_PROMPT` / tool catalog for Lt/2
  symmetry and flexible window selection; default `max_tool_steps` raised to 30.
- Added `tests/unit/test_correlator_tools.py`.

## 2026-06-03 (agent verbose trace)

- Added `core/trace.py` and `run_agent(..., verbose=True)` / CLI `--verbose` to
  print each cycle's prompt, model action, and tool observation before the final
  JSON summary.

## 2026-06-03 (ds_stage1 fixes)

- Force correlator plot PDFs under `cwd/artifacts/` via `resolve_plot_save_path`;
  `plot_fit_on_data` accepts optional `save_path` and rewrites any LLM path to a
  stem under `artifacts/`.
- Default legend `loc="upper right"` in `core/plotting.py`.
- Refactored DeepSeek loop to per-stage multi-turn messages (`build_stage_static_prompt`
  once, `format_tool_observation` per step) to avoid resending static context each
  cycle; verbose trace prints `[Stage context]` once and `[Observation for LLM]`
  deltas thereafter.

## 2026-06-03 (fit_window constraints and CLI summary)

- `fit_window` enforces `tmin >= 1`, `tmax - tmin >= 2*nstate`, at most six
  appended windows, and hard rejection when the window extends past `Lt//2`.
- Agent tool loop maps `ValueError` from stage tools to error observations.
- CLI `run` always echoes a compact JSON summary (no `actions`/`stage_results`
  on stdout); correlator prompts/skills updated for the six-window cap.

## 2026-06-03 (redundant code cleanup)

- Removed dead code: unused `Callable` import, legacy `AgentTrace.prompt()`,
  unused `build_stage_context` helpers in all stage `functions.py` modules.
- Merged LLM entry points into `_request_llm_action` (mock + DeepSeek); removed
  standalone `call_llm_api`.
- Updated README agent workflow and AGENTS.md plotting conventions to match the
  current session-based loop and `core/plotting.py`.
- Added unit test for `model=external` JSONL transcript replay.

## 2026-06-03 (3pt ratio correlator stage)

- Extended `stages/correlator/functions.py` with 3pt read/ratio/resample/fit/plot
  tools (`read_pt3`, `compute_pt3_ratio`, `resample_ratio_to_gvar`,
  `fit_pt3_window`, `plot_pt3_fit_on_data`); `read_pt2` now stores imag samples.
- Added `plot_pt3_ratio_fit_on_data` in `core/plotting.py`; agent routes 3pt plot
  paths through `artifacts/` like 2pt.
- Updated correlator stage prompts/skills and 3pt input validation (3pt requires 2pt).
- Expanded `tests/unit/test_correlator_tools.py` for 3pt fit, dof checks, and plots.

## 2026-06-03 (3pt window cap and multi-tsep manifest)

- Capped 3pt fit trials with `MAX_PT3_FIT_WINDOWS = 2` (2pt still allows 6).
- `workflow_smoke_manifest.json` registers fake 3pt HDF5 for tsep 4, 6, 8, 10.
- Prompts/skills: load all 3pt paths, agent picks `tsep_ls`/`tau_cut`, subset for
  model_average (avoid averaging poor Q windows).

## 2026-06-03 (3pt priors from 2pt model average)

- `fit_pt3_window` defaults to `use_pt2_avg_prior=True`, pinning E0, log(dE1), z0,
  z1 from `*_avg` store keys after 2pt `model_average`.
- Prompts require 2pt BMA on E0, log(dE1), z0, z1 before 3pt fits.

## 2026-06-04 (widen 3pt ratio priors from 2pt posteriors)

- 3pt ratio fits now use 2pt posterior means with uncertainties scaled by
  `PT2_PRIOR_ERROR_SCALE = 5` (`_pt2_posterior_as_prior`) for BMA and single-window paths.

## 2026-06-04 (3pt ratio plot tau windows)

- Ratio data error bars: tau indices ``1 .. tsep-1`` (`_pt3_ratio_data_tau_slice`).
- Fit `fill_between` bands unchanged: each window's ``[tau_cut, tsep + 1 - tau_cut)``.

## 2026-06-04 (3pt ratio plateau reference band)

- Grey reference band now shows model-averaged ``R_plat`` from ``O00_re_avg`` and
  ``E0_avg`` (`asymptotic_ratio_real_gvar`), not raw ``O00``; plateau ``~ O00/(2*E0)``.

## 2026-06-04 (correlator agent tool ergonomics)

- Removed stage-end ``finalize_correlator_plots``.
- Agent drops unknown tool kwargs; ``Lt`` inferred from store for 3pt/plot tools.
- ``fit_pt3_window`` autofills missing ``E0_avg`` / ``z0_avg`` via ``_ensure_pt2_avg_priors``.
- 3pt ratio priors anchor only ``E0`` and ``z0`` from 2pt BMA; ``log(dE*)``, ``z1+``, ``O_ij`` use ``pt3_ratio_prior``.
- ``read_pt3`` / ``compute_pt3_ratio`` / ``resample_ratio_to_gvar`` / ``plot_fit_on_data``
  accept ignored legacy ``out=``.

## 2026-06-04 (slim agent.py)

- Moved LLM sessions and DeepSeek HTTP from ``agent.py`` to ``core/llm.py``
  (``make_llm_session``, ``LlmSession``).
- Moved tool-call preparation (``resolve_tool_args``, ``filter_tool_kwargs``,
  ``prepare_tool_args``) into ``core/tools.py``; dropped redundant agent-side
  ``Lt`` pre-inference (correlator tools infer ``Lt`` when omitted).
- ``agent.py`` now holds stage orchestration only (~200 lines).

## 2026-06-04 (AGENTS.md sync)

- Rewrote ``AGENTS.md`` module map and stage-integration guidance to match the
  current five-package layout (``STAGE_TOOLS``, ``core/stages.py``).
- Removed references to deleted paths and docs (``reporting.py``, ``extensions/``,
  ``planners/``, ``workflows.py``, ``loaders.py``, ``SPEC.md``, ``DEVELOPMENT.md``,
  ``CLAUDE.md``, ``incoming/``, ``docs/analysis_model.md``).
- Documented active docs: ``README.md``, ``PLAN.md``, ``PROJECT_LOG.md``.

## 2026-06-04 (remove TODO.md)

- Removed ``TODO.md``; implementation backlog lives in ``PLAN.md`` and ``PROJECT_LOG.md``.
- Updated ``AGENTS.md`` so active documentation no longer references ``TODO.md``.

## 2026-06-04 (NLO matching kernel)

- Simplified ``src/lamet_agent/kernels.py`` around a direct ``unpolarized_matching_kernel_nlo_gT`` implementation; removed the one-off helper stack while preserving the discrete plus prescription, delta term, and helicity alias.

## 2026-06-08 (3pt tau convention and raw correlator conversion)

- ``read_pt3`` now treats 3pt HDF5 tau rows as ``0..tsep``, so compatible datasets have shape ``(tsep + 1, n_cfg)``.
- Updated fake 3pt generation/tests to the same ``(tsep + 1, n_cfg)`` convention.
- Converted raw ``temp/raw_data`` ensembles into read-compatible HDF5 under ignored ``data/correlators/``; ``conversion_report.json`` records zero-diff source checks and reader checks.

## 2026-06-08 (CG PDF correlator bundle selection fix)

- Rebuilt ignored ``data_cg_pdf/correlators/`` so retained 3pt HDF5 files are ``free`` only and drop raw column 0, which stores tau rather than a gauge configuration.
- Reduced converted 2pt files to ``SS`` datasets matching retained free 3pt ensemble/tag/momentum where available.
- Updated ``examples/workflow_cg_pdf_manifest.json`` to point at the retained free ``HISQa060_X`` files; read and ratio smoke checks now use aligned ``n_cfg=109`` samples.

## 2026-06-08 (CG PDF HISQa060_XYZ 2pt alias)

- Rebuilt ignored ``data_cg_pdf/correlators/`` with an explicit 2pt alias: ``HISQa060_XYZ`` ``CG52bxyzp00_CG52bxyzp00``/``PX0PY0PZ0`` uses raw ``CG52bxyzp20_CG52bxyzp20``/``PX0PY0PZ0`` SS 2pt data.
- ``conversion_report.json`` now records the alias and has no missing matching 2pt entries for retained free 3pt files.

## 2026-06-08 (CG PDF matrix-element samples)

- Converted matching free raw matrix-element bootstrap samples from ``temp/raw_matrix_elements`` into two-column ``data_cg_pdf/matrix_elements/qtmdpdf`` txt files.
- Output real samples use raw sample column 1; imaginary samples are written as zero.  The conversion report records 191 files and zero reload differences.

## 2026-06-08 (CG PDF matrix-element plotting example)

- Added ``examples/cg_pdf_data/read_cg_pdf_matrix_elements.py`` to read converted ``data_cg_pdf/matrix_elements/bare_qpdf`` samples and plot ``HISQa060_X`` ``P=0``/``P=5`` z-dependence.
- The example writes ``examples/cg_pdf_data/cg_pdf_a060_x_p0_p5.pdf`` for quick inspection.

## 2026-06-08 (CG PDF plotting display mode)

- Updated ``examples/cg_pdf_data/read_cg_pdf_matrix_elements.py`` to show separate ``HISQa060_X`` ``P=0`` and ``P=5`` figures interactively instead of saving a combined PDF.

## 2026-06-08 (CG PDF correlator metadata catalog)

- Rewrote ``data_cg_pdf/correlators/conversion_report.json`` as a manifest-oriented correlator metadata catalog while preserving the old conversion statistics under ``conversion_summary``.
- The catalog records 6 usable 2pt targets and 23 3pt files with pion metadata, HDF5 dataset selectors, shapes, and available ``bz`` slices for later manifest authoring.

## 2026-06-08 (CG PDF bare matrix-element path)

- Updated the CG PDF matrix-element reader to load the flattened ``data_cg_pdf/bare_matrix_elements`` directory after moving converted bare sample text files out of the old ``data_cg_pdf/matrix_elements`` tree.

## 2026-06-08 (Correlator bare-matrix export)

- Added a correlator batch grid tool that fits ``O00/(2*E0)`` over selected ``z`` values, chooses windows from bootstrap sample 0, and exports per-z bootstrap sample text files plus a PDF/report under ``artifacts``.
- Updated ``examples/workflow_cg_pdf_manifest.json`` to run the ``HISQa060_X`` ``CG52bxp00`` ``P=0`` ``X`` grid for ``z=0..24`` through the new batch path.

## 2026-06-08 (CG PDF jackknife bare fits)

- Refactored the correlator bare-matrix grid tool to select windows on sample-average data, then refit jackknife samples with the sample-average 3pt posterior as prior.
- Updated the CG PDF manifest to use jackknife samples, ``svdcut=1e-6``, ``Lt//4`` 2pt windows, and ``tau_cut=1..4`` for the all-z ``HISQa060_X`` ``P=0`` stage.

## 2026-06-08 (Correlator joint-fit batch path)

- Reworked the CG PDF correlator batch tool to use joint 2pt+ratio fits for sample-average window selection and per-sample refits.
- Added fit logs and sample-0 ratio fit-on-data PDFs under artifacts, with per-sample priors built from sample-average joint posteriors widened by 3x.
## 2026-06-08 (Correlator fit strategies and split logs)

- Added explicit chained vs joint strategy selection for the correlator bare-matrix grid tool and marked the smoke/CG PDF manifests accordingly.
- Moved reusable bootstrap/jackknife helpers into ``lamet_agent.core.resampling`` while keeping correlator compatibility imports.
- Split batch fit logs into sample-average tuning and per-sample files, with shared ``log_nonlinear_fit_quality`` Good/Bad records.

## 2026-06-08 (CG qPDF P=0/P=5 manifests)

- Renamed CG qPDF workflow test references to ``examples/workflow_cg_qpdf_p0_manifest.json`` after the P=0 manifest rename.
- Added ``examples/workflow_cg_qpdf_p5_manifest.json`` for the ``HISQa060_X`` ``CG52bxp30`` ``P=5`` ``X`` grid with the same joint-fit jackknife workflow settings as P=0.

## 2026-06-08 (CG qPDF p5 fit-log diagnostics)

- Added momentum labels to sample-0 ratio fit-on-data PDF stems under ``artifacts/fit_logs`` to prevent P=0/P=5 overwrite collisions.
- Added direct ``O00/(2*E0)`` bands to sample-0 ratio fit plots and restricted the P=5 diagnostic manifest ``tsep_ls`` to ``[8]``.

## 2026-06-08 (Correlator overlap rescale controls)

- Added agent-driven ``correlator_rescale`` support for 2pt, chained ratio, and joint 2pt+ratio fits so tiny correlator magnitudes can be fit with scaled overlap parameters while preserving ``O00/(2*E0)`` outputs.
- Added ``inspect_correlator_scale`` diagnostics plus prompt/tool-catalog guidance for choosing a power-of-ten rescale that brings fitted 2pt data into the ``0.0001..0.01`` range.
- Logged physical overlap diagnostics by converting scaled fit overlaps back with ``sqrt(correlator_rescale)`` and added rescale invariance/unit coverage.

## 2026-06-08 (Ratio plot denominator correction)

- Updated 3pt ratio fit-on-data plots to always display ratios in the forward-denominator convention by multiplying data and fit bands by the ground-state periodic/forward 2pt factor, keeping grey bands at O00/(2*E0).

## 2026-06-08 (CG qPDF p5 momentum inspection fix)

- Made ``inspect_correlator_scale`` accept selector dictionaries and report the resolved ``source_sink``, ``gamma``, and ``momentum`` so nonzero-momentum 2pt files do not fall back to ``PX0PY0PZ0``.
- Updated correlator batch-mode prompt/catalog guidance to pass the exact HDF5 momentum key from ``Metadata.correlator_grid`` before fitting.

## 2026-06-10 (Ensemble resampling before 3pt/2pt ratios)

- Moved bootstrap/jackknife resampling to ``read_pt2`` / ``read_pt3`` with shared bootstrap indices for one ensemble; ``compute_pt3_ratio`` now divides resampled correlators and ``resample_ratio_to_gvar`` only converts samples to gvar.
- Fixed ``fit_bare_matrix_grid`` (joint and chained) to resample 2pt and 3pt separately with the same indices before forming ratios, instead of resampling ratios built from raw configs.

## 2026-06-10 (Correlator stage agentic refactor)

- Rewrote ``stages/correlator/functions.py`` (~3000 -> ~900 lines) around an agentic inspect -> tune-on-average -> apply-to-samples flow, collapsing the manual low-level tools and the duplicated joint/chained monoliths into shared physics/scan/refit/IO helpers.
- Replaced the 12-tool registry with four tools: ``inspect_correlator_scale``, ``tune_ground_state``, ``tune_bare_matrix``, and ``fit_bare_matrix_grid``. The grid tool now tunes one shared window once (on a representative ``tune_z``) and applies it to every z and every resampled sample, instead of selecting a window per z; it accepts an explicit ``pt2_window``/``pt3_window`` or ``model_average=true`` to BMA-combine the window grid with per-z logGBF weights.
- Removed ceremonial validators and repeated ``int(...)`` re-casting; constrained ints at tool boundaries for readability.
- Shortened ``prompts.py`` to the four-tool flow and trimmed ``skills.py`` to physics facts plus the new tool catalog (removed prompt/skill overlap); updated ``core/tools._PLOT_TOOLS`` so the new plotting tools get ``artifacts_dir``/``save_path`` injection.
- Rewrote ``tests/unit/test_correlator_tools.py`` for the new API (fake-data end-to-end grid coverage for single-window, explicit-window chained, and model-average modes); full unit suite passes (81 tests).
- Behavior change: bare matrix elements now use one shared fit window across all z by default (previously per-z selection), so re-running real data may shift results vs the prior ``runs/ds_pdf_stage1`` per-z windows.

## 2026-06-10 (OpenAI backend alongside DeepSeek)

- Generalized ``core/llm.py`` to OpenAI-compatible providers via a ``PROVIDERS`` table (base URL, default model, API-key env var) so DeepSeek and OpenAI share one ``_post_chat_completion`` / ``_openai_compatible_session`` path; added ``provider_config()``.
- Added ``--model openai`` (default model ``gpt-4o-mini``) next to ``--model deepseek`` (default ``deepseek-chat``); replaced the DeepSeek-specific ``--deepseek-model``/hardcoded base URL with generic ``--llm-model``/``--base-url`` and provider-aware API-key resolution (``api.key`` file or ``DEEPSEEK_API_KEY``/``OPENAI_API_KEY``).
- Renamed ``run_agent``/``make_llm_session`` LLM params to ``llm_model``/``base_url``; updated unit tests and added OpenAI routing coverage.

## 2026-06-10 (LLM JSON repair and plot artifact names)

- Added repair retries for OpenAI-compatible LLM action parsing so malformed provider JSON is fed back for correction instead of aborting the stage immediately.
- Scoped injected plot save stems by ``run_id`` and expanded effective-mass x limits to start at ``min(meff_x) - 0.5`` to avoid adjacent-run plot overwrites and clipped first points.

## 2026-06-10 (CG qPDF ratio-renormalization flow)

- Added sample-preserving ratio/hybrid renormalization tools that read correlator bare-matrix txt grids, apply Eq. 15 with CG defaults (`zs=4`, `delta_m=m0=0`), write compatible `EnsembleData` NPZ artifacts, and plot renormalized matrix elements.
- Made the agent store persist across stages so renormalization can hand `matrix_element_data` directly to Fourier in one run; added manifest argument merging for `metadata.renormalization`.
- Updated the PX5 CG qPDF example and stage1 run script for a correlator -> renormalization -> Fourier smoke flow using the PX0 report as the ratio denominator.

## 2026-06-10 (Correlator sample fit quality logging)

- Added per-sample nonlinear fit quality logging to correlator bare-matrix grid fits so successful resampled ground-state/joint fits write Good/Bad Q, chi2/dof, and logGBF lines to the samples log.

## 2026-06-10 (Correlator grid argument defaults)

- Made ``prepare_tool_args`` fill missing correlator tool selectors and grid fields from ``metadata.correlator_grid``, so nonzero-momentum workflows keep the manifest momentum key even when the LLM omits it in later tool calls.
- Added unit coverage for PX5-style correlator argument preparation.

## 2026-06-10 (ds_pdf_complete two-step full pipeline)

- Added ``examples/workflow_cg_qpdf_complete_manifest.json`` for HISQa060_X PX5 correlator through ``perturbative_matching``, with NLA Fourier settings from ``ds_pdf_cont`` and ``metadata.matching`` for ``unpolarized_gT``.
- Added ``runs/ds_pdf_complete/run.sh``: step 1 runs ``workflow_cg_qpdf_p0_manifest.json`` (``correlator_analysis`` only); step 2 runs the complete manifest with ``correlator_analysis,renormalization,fourier_transform,perturbative_matching``, using ``runs/ds_pdf_complete/artifacts/a060_x_p0_bare_matrix_elements_report.json`` as the ratio denominator.

## 2026-06-16 (Prompt context trimming)

- Trimmed non-Fourier stage prompts by filtering stage metadata, omitting repeated correlator paths outside correlator/Fourier stages, and removing duplicated action-protocol wording from correlator, renormalization, and matching prompts.
- Kept Fourier stage prompt/context behavior unchanged while preserving the multi-turn API conversation shape; verbose traces now print a compact observation-forwarded marker instead of duplicating full observations.

## 2026-06-16 (Legacy helper cleanup)

- Removed the unused monolithic `build_stage_prompt` helper and its core export now that agent runs use `build_stage_static_prompt` plus per-turn observations.
- Removed the unused `set_my_logger` compatibility wrapper while keeping `setup_logger` as the active logging helper.

## 2026-06-16 (LLM observation filtering)

- Stopped forwarding dropped `ignored_args` payloads to the LLM while keeping them available in tool observations for trace/debug output.

## 2026-06-18 (EnsembleData NetCDF serialization)

- Added `EnsembleData.to_netcdf` / `from_netcdf` support using xarray NetCDF4 output with `auto_complex=True` so complex arrays round-trip without manual real/imag splitting.
- Stored `ensemble` and `resample` metadata in DataArray attrs, added `netCDF4` to the analysis extra, and added a focused I/O smoke test for complex NetCDF round-tripping.

## 2026-06-18 (README NetCDF intermediate I/O)

- Documented NetCDF as the standard stage-to-stage artifact format in `README.md`, including artifact naming, manifest path conventions, and Python/xarray read/write examples.
- Updated Quick Start to recommend the `[analysis]` extra for NetCDF I/O dependencies.

## 2026-06-18 (Correlator-renormalization NetCDF handoff)

- Migrated the correlator-to-renormalization handoff from JSON report/txt-grid loading to `EnsembleData` NetCDF artifacts.
- Changed ratio-scheme renormalization output from `.npz` to `.nc` while leaving Fourier and matching IO for a later coordinated update.
- Removed correlator-stage per-z bare matrix `.txt` output so the bare matrix element artifact is NetCDF-only.

## 2026-06-20 (Job-DAG manifest migration, phase 1)

- Replaced the legacy top-level manifest contract with `metadata`, global `inputs`, and per-stage `defaults`/`jobs`; `metadata.stages` is now the sole execution order.
- Added per-job isolated stores and job-id output registration so role-named downstream inputs resolve in memory without a second run.
- Migrated correlator analysis to derive paths/selectors from `correlator_ids`, scan configured nstate/strategy candidates, and write job-scoped NetCDF outputs with lattice metadata.
- Migrated renormalization to consume `target`/`denominator` job roles and apply `hybrid_ratio` using `scheme_parameters.zs_fm` and the target lattice spacing.
- Added the P0+P5 `cg_pion_pdf_manifest.json` and reduced `runs/ds_pdf_complete/run.sh` to one manifest and one run command.

## 2026-06-20 (Correlator model-average control)

- Added an authoritative correlator `model_average` manifest default so LLM tool arguments cannot accidentally switch a single-window production run into the roughly 12x more expensive full-window BMA path.

## 2026-06-20 (Fourier and matching job-DAG migration)

- Migrated Fourier and perturbative matching parameter preparation from legacy metadata fields to stage defaults, job params, role-named upstream outputs, and kernel declarations.
- Kept the Fourier numerical workflow unchanged while registering its EnsembleData as the job output and scoping its NetCDF, fit-info, plot, and report artifacts by job id.
- Added logical `unpolarized_gT` kernel resolution, in-memory Fourier-to-matching handoff, and matched-PDF NetCDF output.
- Extended `cg_pion_pdf_manifest.json` through matching and added `partial_cg_pion_pdf_manifest.json` for restart from the saved renormalization artifact.

## 2026-06-20 (Partial-run external artifact hydration)

- Auto-load declared `inputs.artifacts` into job stores before the LLM tool loop for Fourier (`input` → `load_renormalized_matrix_element_samples`) and matching (`quasi` → `load_quasi_pdf`).
- Clarified system prompt that external artifact inputs are pre-loaded so partial/resume runs do not depend on the model calling loader tools first.
- Added agent unit tests covering hydration without a manual loader action.

## 2026-06-20 (Partial-run loader path injection)

- Resolve declared artifact paths in `prepare_tool_args` when job inputs were pre-hydrated to `EnsembleData`, so redundant loader calls still receive `path`.
- Made `load_renormalized_matrix_element_samples` idempotent when `matrix_element_data` is already loaded.
- Updated Fourier stage prompt to call `run_fourier_transform` directly after pre-load.

## 2026-06-23 (Correlator FH fit scope)

- Added correlator `fit_scope` support for `ratio`, `FH`, and `ratio+FH`, including FH construction from summed ratios and joint/chained fitting through the existing bare-matrix tools.
- Updated correlator manifests, prompts, and tests so agents can scan scope choices while preserving the NetCDF `EnsembleData` output contract.

## 2026-06-23 (Correlator FH diagnostics)

- Added FH sample-0 fit diagnostic PDFs under correlator `fit_logs` for FH and `ratio+FH` grid fits.
- Stopped `tune_bare_matrix` from writing root-level `tune_*_sample0_pt3_ratio_*.pdf` diagnostics; tuning now returns ranked candidates without producing those PDFs.

## 2026-06-23 (Correlator systematic-error attrs)

- Added correlator bare-matrix `EnsembleData` attrs for per-z real/imag mean, resampling statistical error, and window-model systematic error.
- Kept stored correlator samples unchanged while reporting zero systematic spread for single-window fits and logGBF-weighted window spread for model-averaged fits.

## 2026-06-23 (Correlator and renormalization reports)

- Added concise bilingual stage reports for correlator analysis and renormalization, wired through the same post-stage runner hook used by Fourier and matching.
- Included correlator `fit_logs` descriptions and links to existing NetCDF/PDF artifacts without adding PNG companions.

## 2026-06-26 (GI PX4 x_dependence reference tables)

- Added ``temp/Fig16/read.py`` to convert Fig. 16 ``App2_*_GI_pz4.dat`` tables into ``# x y_mean y_sdev`` text files under ``data_gi_pdf/x_dependence/``.
- Wrote ``HISQa060_X_GI_PX4_Pion_PDF_NLO_LRR.txt`` (100 x points) and ``HISQa060_X_GI_PX4_Pion_qPDF.txt`` (349 x points); x grids match the CG reference layout in ``data_cg_pdf/x_dependence/``.

## 2026-06-26 (GI pion PDF manifest)

- Added ``examples/gi_pion_pdf_manifest.json`` for the HISQa060_X GI ``hyp`` correlators under ``data_gi_pdf/``, running P0+P4 through correlator analysis, hybrid-ratio renormalization, Fourier transform, and perturbative matching at ``pz_gev=1.72``; the P4 2pt uses ``PX4PY0PZ0`` from the shared ``CG52bxp30`` HDF5 file.
- Added ``runs/ds_gi_pdf/run.sh`` and ``plot_matched_pdf_compare.py`` mirroring ``runs/ds_pdf_complete`` for the GI PX4 reference tables.

## 2026-06-26 (Hybrid-ratio manifest parameters)

- Declared explicit ``m0``, ``delta_m``, and ``z0`` in ``examples/cg_pion_pdf_manifest.json``, ``examples/gi_pion_pdf_manifest.json``, and ``examples/sample_manifest.jsonc`` renormalization defaults; CG uses zeros, GI uses ``m0=0.1232`` GeV and ``delta_m=0.545227463`` GeV (``0.1586 * GEV_FM / a_fm`` at ``a_fm=0.0574``).
- Extended ``test_prepare_renormalization_args_bind_roles_and_scheme`` to assert manifest passthrough of the new top-level hybrid-ratio fields.

## 2026-06-26 (Renormalization parameter cleanup and unit fix)

- Removed configurable ``z0``; hybrid-ratio normalization is fixed at lattice ``z=0``.
- Moved ``m0_gev`` and ``delta_m_gev`` into ``scheme_parameters`` (GeV); updated GI/CG/sample manifests accordingly.
- Fixed long-range exponent to use physical distance: ``exp((m0_gev + delta_m_gev) * (|z|_fm - z_s) / GEV_FM)``.
- Updated renormalization reporting formula and unit tests for the corrected exponent scaling.

## 2026-06-29 (Correlator fit-function model averaging)

- Reworked correlator ``model_average`` semantics so data-window choices are fixed from sample-average tuning and model averaging varies fit-function choices only.
- Added correlator ``prior_width`` scans with default factors ``[0.5, 1.0, 2.0]`` and documented the revised systematic-error meaning as fit-model spread.

## 2026-06-29 (Correlator data-window selection)

- Added explicit ``pt3_windows`` guidance to the sample manifest so tau-cut scans can use all selected tseps by default or opt into tsep subsets.
- Split correlator data-window selection from fit-model selection: data windows now gate on ``Q`` and ``n_data > n_params``, then prefer low ``chi2/dof`` with a bias toward more data when fits are comparable.
- Exposed data-window size metadata in tuning candidates and updated correlator prompts so the agent chooses windows from ``Q``/``n_data``-passing candidates without ranking different data windows by raw ``logGBF``.
- Hardened correlator terminal-tool argument preparation so ``model_average=true`` preserves manifest ``nstate``/``prior_width`` scan lists even if the LLM proposes a single fit model, and normalized bare ``tmin``/``tmax``/``tau_cut`` shorthand into explicit window arguments.

## 2026-06-29 (Report language selection)

- Added ``--report_language en|ch`` to ``lamet-agent run`` and threaded it through ``run_agent()``.
- Changed stage and per-job report writers to emit only the selected single-language Markdown report instead of both English and Chinese files by default.

## 2026-06-29 (Correlator component-specific output)

- Made correlator bare-matrix output honor ``component``/``part`` when exporting samples and summary plots, setting the excluded component to zero instead of propagating unconstrained prior means.
- Added a unit test covering the ``re``-only path that should force the imaginary bare matrix element to zero downstream.

## 2026-07-02 (Codex LLM session backend)

- Added ``model=codex`` to ``core.llm.make_llm_session()` using the new ``codex_decide`` helper so the main agent loop can use the Codex Python SDK instead of an OpenAI-compatible HTTP API provider.
- Kept ``openai-codex`` as an optional ``[codex]`` extra and delayed importing the SDK until the codex backend is used, so existing ``mock``/``external``/``deepseek``/``openai`` workflows remain importable without the SDK.
- Updated CLI/README backend lists and added unit coverage for routing stage prompts and tool observations through ``codex_decide``.
- Removed strict ``output_schema`` from the Codex SDK turn call after diagnosing the SDK failure as an ``invalid_json_schema`` rejection for flexible tool ``args``; Codex responses are now parsed with the same JSON repair helper used by API providers.

## 2026-07-01 (Global resampling metadata: random_seed, bs_samples, bin_size)

- Moved correlator-stage resampling configuration out of ``correlator_analysis.defaults.seed`` and into required/optional top-level ``metadata`` fields: ``random_seed`` (required, seeds every jackknife/bootstrap call), ``bs_samples`` (required only when ``resample_mode`` is ``"bs"``; ignored for ``"jk"``, replaces the hardcoded ``n_boot=200``), and ``bin_size`` (optional, no default requirement).
- Added a ``RunMetadata`` model validator that rejects manifests with ``resample_mode: "bs"`` and no ``bs_samples``; documented required vs optional fields directly in the ``RunMetadata`` docstring.
- Added ``bin_data()`` plus ``bin_size`` support to ``jackknife``/``bootstrap``/``resample_config_samples`` in ``core/resampling.py``, and threaded ``bin_size`` through ``_resample_pt2``, ``tune_ground_state``, ``tune_bare_matrix``, and ``fit_bare_matrix_grid`` in the correlator stage.
- ``prepare_tool_args`` now injects ``seed``/``n_boot``/``bin_size`` from ``metadata.random_seed``/``metadata.bs_samples``/``metadata.bin_size`` for every correlator tool call, the same way ``resample_mode`` is already injected; a job/stage no longer needs its own ``seed``.
- Updated all tracked example manifests and ``sample_manifest.jsonc`` (with inline comments for the new fields) plus inline test manifests to include ``metadata.random_seed``.

## 2026-07-01 (CLI backend/model flag refactor)

- Replaced overloaded CLI ``--model`` backend selector with required ``--backend mock|external|api|codex`` and ``--model provider/model_id`` for the ``api`` backend only; removed ``--llm-model``.
- Added ``parse_api_model()`` and ``format_api_model_spec()`` in ``core/llm.py``; refactored ``make_llm_session()`` / ``_request_llm_action()`` to take ``backend`` plus optional ``provider``/``model_name``; unknown backends now raise ``ValueError`` instead of silently falling back to mock.
- Updated ``run_agent()`` return summary to emit ``backend`` and, for ``api``, ``model`` as ``provider/model_id``; trace output uses ``backend`` + optional ``model_spec``.
- Updated unit tests, README, and AGENTS.md. Local run scripts under ``runs/`` must be updated manually (e.g. ``--backend api --model deepseek/deepseek-chat`` instead of ``--model deepseek``).

## 2026-07-01 (Quiet CLI startup banner and job headers)

- Added ``core/banner.py`` with a GRID-style LaMET Agent ASCII banner and ``format_job_header()``.
- Extended ``AgentTrace`` with ``quiet_ui`` mode: non-verbose runs print the banner, run summary, and one ``Stage: … | Job: …`` line before each job; ``--verbose`` behavior is unchanged.
- Wired ``run_agent()`` to use ``run_banner()``/``job_begin()`` when ``verbose=False`` and added unit tests in ``tests/unit/test_banner.py``.

## 2026-07-02 (Fit-log ylim: data band at 3/12–7/12)

- Extended ``_ylim_middle_third()`` with optional asymmetric margin factors; default remains symmetric middle third.
- Fit-log pt3 ratio and FH panels now place data±error at axis height ``3/12``–``7/12`` via ``FIT_LOG_YLIM_*`` constants (``bottom=0.75*span``, ``top=1.25*span``).
- Added unit test ``test_ylim_middle_third_fit_log_factors_place_data_at_three_to_seven_twelfths``.

## 2026-07-02 (Central sample-error mode)

- Added top-level ``metadata.sample_error_mode`` with ``mean``/``median``/``covariance`` options and rejected the invalid jackknife-plus-median combination during manifest validation.
- Centralized bootstrap/jackknife sample averages, mean/sdev extraction, and sample-by-sample error attachment in ``core/resampling.py``.
- Threaded ``sample_error_mode`` through correlator, renormalization, and Fourier tools; Fourier no longer reads per-stage ``fit_error_mode``.
- Updated tracked example manifests and README metadata guidance for the new sample-error contract.
- Kept ``sample_error_mode`` strict to the three public values only: ``mean``, ``median``, and ``covariance``.

### 2026-07-02 — Example manifest cleanup

- Normalized 2-space indentation across all ``examples/*manifest*`` files; removed tab characters from Fourier defaults.
- Replaced obsolete ``unpolarized_gT`` kernel ids with ``CG_gt_PDF_hybrid`` and aligned ``zs_fm`` to ``0.1722`` in sample/partial_sample manifests.
- Simplified Fourier ``scheme_scan`` to auto-fill style (``model_average`` only) in all example manifests; ``sample_manifest.jsonc`` documents optional override keys in comments.
- Updated ``sample_manifest.jsonc`` correlator defaults to ``pt3_tau_cuts`` and ``HISQa060_X`` ensemble metadata.
- Added ``test_example_manifests_validate`` to guard example manifest schema and stage-input validation.

## 2026-07-03 (Renormalization stage normalization switch)

- Added ``renormalization.defaults.normalization`` (default ``true``) to control z=0 division of bare matrix elements at renormalization job entry.
- Extracted ``normalize_bare_matrix_element_at_z0`` from hybrid-ratio scheme logic; ``apply_ratio_scheme_renormalization`` now applies the pure ratio/hybrid map only.
- Removed ``normalize_z0`` from ``fit_self_renormalization_factor``; pre-normalized inputs are detected via ``normalized_at_z0`` attrs.
- Updated example manifests, renorm prompts/skills, README semantics, and unit tests for the new contract.

## 2026-07-04 — Multi-z correlator window tuning

- Extended ``tune_bare_matrix`` to require LLM-supplied ``tune_z_values`` and scan each configured window at every tune z using the same ``_scan_average`` / ``_fit_usable`` gates as ``fit_bare_matrix_grid``.
- Added cross-z candidate summaries (`feasible_at_all_tune_z`, `tune_z_diagnostics`, `min_Q`, `worst_chi2_dof`, `failure_reasons`) plus ``recommended_robust_index`` / ``recommended_robust_window``.
- Updated correlator prompts/skills so the agent picks representative tune z values from the job ``bz`` list and selects only cross-z-feasible shared windows before calling ``fit_bare_matrix_grid``.
- Added unit tests for validation, helper aggregation, and tool-arg wiring; updated README correlator tuning notes.

## 2026-07-07 (Interactive manifest planning)

- Added ``lamet-agent plan`` as an interactive draft-manifest review mode using the existing ``api``/``codex`` LLM configuration, with ``mock`` retained for tests.
- Added relaxed JSONC loading, deterministic manifest issue checks, scheme-alignment proposals, and quick/full manifest generation while keeping ``validate``/``run`` strict.
- Added correlator-only HDF5 inspection and conversion into the standard reader layout under ``<artifacts_directory>/plan_data/``.
- Refined the terminal flow to print the LaMET Agent banner, ask deterministic questions one at a time before acceptance, and render a concise categorized summary instead of model-generated unresolved-question lists.
- Added deterministic handling for revision requests that broaden correlator fit-window searches, so revised summaries and generated manifests reflect the user's request.
- Added path-aware revision rollback so later user instructions such as reverting ``pt3_tau_cuts`` remove stale deterministic edits instead of accumulating contradictory changes.
- Moved generated quick/full manifests under ``<artifacts_directory>/plan_manifests/`` and print separate post-write summaries of the quick/full changes.
- Documented plan mode and added unit coverage for relaxed loading, issue detection, HDF5 conversion, and the mock CLI accept path.

## 2026-07-07 (LLM-controlled planning loop)

- Reworked ``lamet-agent plan`` so ``api``/``codex`` backends drive an iterative planning action loop instead of only generating a summary after deterministic checks.
- Added guarded planning tools for manifest checks, HDF5 inspection/conversion planning, JSON Patch candidate edits, candidate validation, and quick/full manifest generation.
- Kept final file writes behind explicit user acceptance; revision text now routes back through the planning agent and validated patches rather than fixed phrase matching.
- Added unit coverage for patch application/rejection, invalid candidate validation, and Chinese natural-language renormalization-stage revision through the mock planning action path.

## 2026-07-07 (Planning user-answer guardrails)

- Rejected malformed planning-agent ``request_user_input`` actions that omit a concrete prompt instead of showing an empty terminal question.
- Applied answers to manifest-path questions such as ``metadata.random_seed`` directly through the guarded JSON Patch tool path, so the LLM does not need to re-patch required scalar fields after the user answers.
- Added API-style regression tests for malformed input actions and direct random-seed answer application.

## 2026-07-08 (Self-renormalization scheme)

- Added coordinate-space ``ZMSbar_pdf`` / ``ZMSbar_da`` kernels in ``kernels.py`` for the renormalization stage.
- Wired ``self_renormalization`` beside ``hybrid_ratio``: fit ``zR`` from a multi-``a`` reference, then apply ``H/(zR*ZMSbar)`` via ``apply_self_renormalization``.
- Extended renorm skills/prompts/tool-arg binding, artifact hydration, and reporting for scheme branching on roles ``target``+``reference``.
- Added ``examples/temp_self_renorm_manifest.json`` and ``runs/ds_self_renorm/`` prepare/run helpers that convert ``temp/lamet_da_self_renorm`` dumps into NetCDF smoke inputs.

## 2026-07-08 (Self-renormalization diagnostic plots)

- Extended ``fit_self_renormalization_factor`` to stash ``store['self_renorm_fit']`` arrays for diagnostic plotting.
- Added ``plot_self_renormalization_diagnostics`` covering zR-fit checks, ``H/zR`` vs ``ZMSbar``, and multi-a discrete-effect overlays (no continuum band).
- Expanded the self-renorm smoke manifest/actions to a06/a09/a12 jobs and wired diagnostics into prompts, tool-arg binding, and renorm reports.

## 2026-07-08 (Self-renorm svdcut and plot labels)

- Made ``fit_self_renormalization_factor`` accept ``svdcut`` (default ``1e-12``) instead of hard-coded ``1e-100``, and bind it from ``scheme_parameters.svdcut``.
- Fixed ``plot_renormalized_matrix_element`` default title/x-axis so self-renorm plots are not labeled as ratio-scheme ``z/a``.

## 2026-07-09 (Self-renorm fidelity and fit/apply split)

- Split self-renormalization into one ``{reference}`` fit job and three ``{target, zR}`` apply jobs; zR is fit once on sample-averaged MILC reference and stored as a one-sample mean EnsembleData.
- Separated ``d_fit`` (PDF gz fit) from ``d`` (DA zR construction); m0 fitting uses ``ZMSbar_pdf`` while apply uses declared ``ZMSbar_da``.
- Regenerated MILC-only bootstrap reference on the full DA z grid; dropped ``fit_vs_data``; emit fit diagnostics once and ``discrete_effect`` once on the last apply job.

## 2026-07-09 (Simplify self-renorm fixed m0/d)

- Required fixed ``m0_gev`` (no m0 fit / no ``fit_m0`` panel); removed ``n_m0`` and ``d_fit`` so a single ``d`` enters both gz fit and zR construction.
- Write multi-a discrete-effect overlays as stage-level ``discrete_effect_re`` / ``discrete_effect_im`` (no job-id prefix).
- Simplified ``examples/temp_self_renorm_manifest.json`` ``scheme_parameters`` to ``m0_gev``, ``d``, ``mu``, ``svdcut``.

## 2026-07-09 (Optional m0_gev for self-renorm fit)

- ``scheme_parameters.d`` is required on the self-renormalization fit job (fixed; never fitted).
- ``scheme_parameters.m0_gev`` is optional: omit to fit ``m0`` from the first three ``g(z)`` points vs ``ZMSbar_pdf``; set it to freeze ``m0`` (e.g. PDF reference applied to DA).
- Record ``m0_source`` (``fixed``|``fit``) and ``d`` on ``zR`` attrs / ``self_renorm_fit`` / tool return.
- Moved fit-job ``d``/``m0_gev`` onto ``rn_zR_fit`` params in ``examples/temp_self_renorm_manifest.json``.

## 2026-07-09 (Flat job params + apply-job d/m0 remap)

- Self-renorm ``d`` / ``m0_gev`` / ``mu`` / ``svdcut`` are flat job ``params`` (not nested ``scheme_parameters``).
- Fit job requires ``params.d``; ``params.m0_gev`` optional (omit → fit).
- Apply jobs may set ``params.d`` / ``params.m0_gev`` to remap upstream zR (PDF→DA); ``apply_self_renormalization`` rewrites store ``zR`` for diagnostics.
- ``examples/temp_self_renorm_manifest.json``: fit uses ``d=-0.08183``; apply jobs use ``d=0.19``, ``m0_gev=-0.094``.

## 2026-07-09 (README self-renormalization section)

- Added a dedicated README section covering self-renorm workflow, manifest shape, parameter table (required vs optional), and outputs.

## 2026-07-09 (Kernel stage id: perturbative_matching)

- Renamed ``inputs.kernels[].stage`` from shorthand ``matching`` to full stage id ``perturbative_matching`` in all example manifests and planning tests.
- Updated ``effective_matching_params`` to filter kernels by ``stage == "perturbative_matching"``.
- Tightened ``KernelInput.stage`` to ``StageId`` so invalid shorthand fails schema validation.

## 2026-07-14 (Sample-fit process parallelism)

- Added optional positive ``metadata.workers`` (default ``1``) and injected it into correlator-grid and Fourier terminal tools.
- Parallelized independent correlator and Fourier sample fits with reusable ``ProcessPoolExecutor`` pools while keeping tuning, logging, plotting, extrapolation, and Fourier summation in the main process.
- Used ``gvar.dumps`` / ``gvar.loads`` for multiprocessing payloads so correlated priors and covariance templates retain their correlations.
- Added serial/parallel equivalence tests and documented BLAS thread limits for avoiding process/thread oversubscription.

## 2026-07-14 (Canonical matching kernel ids)

- Replaced stale matching example id ``CG_gt_PDF_hybrid`` with the exact ``kernels.py`` function names: ``CG_gt_qPDF_hybrid_NLO`` for CG workflows and ``GI_gt_qPDF_hybrid_NLO`` for the GI workflow.
- Updated matching/tool/planning tests and added a registry invariant requiring every ``KERNEL_REGISTRY`` key to equal its kernel builder's public function name.

## 2026-07-14 (Per-job hybrid switch distance)

- Moved hybrid ``zs_fm`` out of matching ``kernel_parameters`` and renormalization ``scheme_parameters`` into flat stage defaults or job params, with job-level overrides.
- Rejected both legacy manifest locations and updated stage validation, tool argument preparation, planning guidance, and workflow examples to use the new canonical paths.
- Added a non-blocking review check that follows matching → Fourier → renormalization DAG chains and reports consistent, mismatched, non-applicable, or externally unverifiable ``zs_fm`` settings.

## 2026-07-14 (Correlator readability cleanup)

- Consolidated Breit, NonBreit, ratio, FH, and ratio+FH nonlinear fits behind one parameterized ``fit_matrix_element`` core while preserving the four registered correlator tool contracts.
- Inlined single-use tuning, logging, progress, and output orchestration; unified serial/parallel sample fitting through one batch path; narrowed numerical soft-fail handling; and reduced the terminal NetCDF write to one final write.
- Removed production-dead correlator helpers, reconciled the correlator tool catalog with ``STAGE_TOOLS``, and added focused fit/catalog coverage.
- Moved shared report formatting, language-target, and Markdown artifact-path handling into ``core/reporting.py`` for correlator, renormalization, Fourier, and matching reports.

## 2026-07-15 (Manifest and standard correlator HDF5 v2)

- Replaced correlator gamma/source-sink selectors with free-form source, sink, and current operator labels; added canonical volume labels, list-valued momentum/``tsep`` settings, and ``bT`` naming.
- Standardized 2pt and 3pt HDF5 paths, including explicit ``tsep`` groups, and updated readers, planner conversion, HDF5 inspection, and fake-data generation to the v2 layout.
- Made discrete momentum, volume, and lattice spacing the manifest-authoritative kinematics and derived physical momentum consistently across correlator, Fourier, matching, reports, and artifact attributes.
- Consolidated the tracked CG/GI/sample manifests around shared multi-setting correlator files and updated partial-run artifacts to declare discrete kinematics.
- Migrated the ignored CG/GI data catalogs into per-ensemble 2pt and per-ensemble/channel 3pt files, verified every dataset byte-for-byte at the array level, and rewrote both catalogs as version-2 metadata.
- Documented the standard correlator HDF5 contract in ``README.md`` and expanded schema, reader, planner, tool-preparation, and momentum-derivation tests.

## 2026-07-15 (Annotated sample manifest reference)

- Expanded ``examples/sample_manifest.jsonc`` as a commented reference template for optional metadata, correlator, stage, plotting, reporting, and partial-run fields while retaining shared multi-setting HDF5 entries.
- Documented mutually exclusive branches in place, including Breit ``momentum`` versus NonBreit ``initial_momentum``/``final_momentum``, hybrid-ratio versus self-renormalization, and Fourier ``sector`` versus low-level ``part`` selection.

## 2026-07-15 (Temporary manifests and local data migration)

- Migrated all four ignored ``examples/temp*manifest.json`` workflows to the v2 manifest contract, including shared multi-momentum/multi-``tsep`` correlator inputs and discrete partial-run kinematics.
- Consolidated the local C-CLQCD gluon catalog from 51 legacy HDF5 files into one 2pt and one 3pt file, preserving the real/imaginary current channels as distinct nonlocal operators and verifying all 483 mapped datasets exactly.
- Updated the associated C-CLQCD data builder to emit the shared v2 files, metadata catalog, and manifest entries directly.
- Updated local GI DA and self-renormalization NetCDF provenance to ``volume``, ``lattice_spacing_fm``, and formula-derived ``momentum_gev`` while verifying that variable values and dimensions were unchanged.

## 2026-07-15 (Correlator separation-direction provenance)

- Added required 3pt ``bz_direction`` provenance with canonical axis-set labels ``X``, ``Y``, ``Z``, ``XY``, ``XZ``, ``YZ``, and ``XYZ`` while keeping the standard HDF5 dataset path unchanged.
- Propagated ``bz_direction`` through correlator tool preparation and bare matrix-element attrs, taught the planner to request and inspect it, and documented ``bz`` as longitudinal/nonlocal separation and ``bT`` as transverse/nonlocal separation.
- Removed the unused correlator ``variant`` parameter from manifests, tool signatures, log names, and new artifacts; migrated existing local HDF5 catalogs/root attrs and removed historical NetCDF ``variant`` attrs with exact data-equivalence checks.

## 2026-07-16 (Fourier decay-offset units)

- Renamed the Fourier-stage ``Lambda0`` parameter to ``Lambda0_gev`` across manifests, Python APIs, results, NetCDF attributes, and reports, and changed its default from ``0.1`` to ``0.0``.
- Rejected the legacy manifest key with path-specific validation errors and migrated all tracked example manifests to the new name.
- Added schema, argument-preparation, numerical, artifact, and report coverage for the renamed parameter and its new default.
- Decoupled tool-preparation tests from mutable example-manifest parameter values by deriving expectations from the loaded manifest or using test-local sentinels.

## 2026-07-16 (Upstream matching grid integration)

- Integrated the upstream matching-grid update with the local Fourier decay-offset work, adopting the ``*_quark_PDF_*`` kernel ids and the ``quasi_y_ls``, ``lc_x_ls``, and ``endpoint_cut`` tool parameters.
- Preserved manifest-driven tool tests across the kernel rename so editable example parameter values are not hard-coded in assertions.

## 2026-07-16 (Strict stage manifest parameter contracts)

- Added lightweight per-stage ``params.py`` contracts and recursive validation for unknown keys in stage ``defaults`` and job ``params``, including nested grids, scans, plot settings, and correlator windows.
- Added path-specific typo, legacy-field, derived-kinematics, and run-metadata guidance; ``validate``, ``run``, and interactive planning now reject ignored stage parameters consistently.
- Removed unused extrapolation momentum placeholders and fixed the planning quick variant so correlator-only ``model_average`` is no longer written into unrelated stages.

## 2026-07-16 (Centralized stage parameter contracts)

- Consolidated the stage parameter schemas and removed-field guidance into the central ``STAGE_PARAM_CONTRACTS`` registry in ``manifest_params.py``.
- Removed per-stage ``params.py`` modules and the temporary top-level stage registry while preserving strict validation behavior and lightweight manifest imports.

## 2026-07-16 (Self-describing partial-run artifacts)

- Made standard ``EnsembleData`` NetCDF sources self-describing for partial workflows by reading data-variable attrs before stage validation without loading array values.
- Kept the complete manifest kinematic triple as a legacy fallback, derived physical momentum from resolved discrete kinematics, and rejected conflicting manifest/NetCDF metadata before execution.
- Simplified the tracked partial PDF manifest to ``id``/``stage``/``path`` and added attrs-only, fallback, conflict, missing-metadata, and no-write coverage.

## 2026-07-16 (Pointwise ratio renormalization)

- Added ``scheme: "ratio"`` beside hybrid-ratio and self-renormalization, applying sample-preserving ``target(z) / denominator(z)`` on the complete coordinate grid without hybrid switch or mass parameters.
- Extended renormalization validation, tool preparation, prompts, planning guidance, and bilingual reports while preserving the existing optional z=0 normalization preprocessing.
- Added numerical, metadata, planning, argument-binding, validation, and report coverage for the new scheme.

## 2026-07-17 (Pion PDF data layout)

- Renamed the local CG/GI pion PDF data roots to ``data_pion_pdf_cg`` and ``data_pion_pdf_gi`` and updated active manifests, conversion utilities, and comparison scripts.
- Consolidated 191 per-z CG bare-matrix bootstrap text files into seven HDF5 sample grids with ``z`` and complex ``samples`` datasets while retaining mean/sdev x-dependence tables as text.
- Updated CG/GI comparison readers to consume the HDF5 sample-grid contract and preserve bootstrap mean/error calculations.

## 2026-07-17 (Bare-matrix ensemble containers)

- Consolidated the seven CG bare-matrix sample grids into three ensemble-named HDF5 files, using ``direction/momentum`` groups to distinguish operator directions and kinematics without encoding implementation details in filenames.
- Updated CG/GI comparison readers to select the required HDF5 group from the simplified per-ensemble container convention.

## 2026-07-17 (Particle-first manifest names)

- Renamed the formal pion PDF and pion/kaon DA manifests to particle-first names, synchronized their run ids, and updated active documentation, tests, and run-script references.
- Added dedicated ``ds_pion_da_gi`` and ``ds_kaon_da_gi`` run entry points for the renormalization-only DA workflows while retaining the unrelated J/psi DA temp workflow.

## 2026-07-17 (Renormalization job tool routing)

- Restricted model-visible renormalization tools by scheme and job input roles after external-artifact hydration, preventing self-renormalization apply jobs from invoking the reference-only fit tool.
- Added the job-specific allowed tool list to static prompts and made disallowed stage-tool requests recoverable observations instead of executing them against incompatible stores.
- Added routing and agent-loop regression coverage for ratio, self-renormalization fit, and self-renormalization apply jobs.
- Made the pion/kaon DA run scripts invoke the repository ``.venv`` CLI explicitly and verified the pion API workflow through one fit plus nine apply jobs.

## 2026-07-17 (Hybrid self-renormalization parity)

- Renamed the public ``self_renormalization`` scheme to ``hybrid_self_renormalization`` with an explicit migration error for the removed name, while retaining the internal fit/apply tool names.
- Restored the legacy MILC+RBC joint fit with shared ``g(z)``, discretization-group-specific ``f_i(z)``, quadratic long-distance extension through 1.50 fm, and strict target-grid/lattice-spacing checks.
- Added explicit ``alpha_s`` support to the MSbar kernels and propagated the legacy coupling, SVD cut, PDF fit, and DA conversion parameters through manifests, diagnostics, reports, and artifacts.
- Regenerated the grouped pion/kaon zero-momentum references, reran both DA workflows, and verified the 25-point ``Z_R`` and all 18 renormalized 600-sample outputs against the legacy formulas to machine precision.

## 2026-07-17 (Momentum-resolved discretization diagnostics)

- Split hybrid-self-renormalization discrete-effect overlays by momentum so each figure compares only lattice spacings for the same matrix-element quantity.
- Added momentum-specific stage artifact names and documented that the legacy explicit coupling is derived from one-loop running with a manually fixed ``Lambda_MSbar``, distinct from the self-renormalization ``lqcd`` ansatz parameter.

## 2026-07-17 (Generalized hybrid self-renormalization)

- Removed the legacy-only numerical ``alpha_s`` override, multi-discretization grouping, long-distance ``z_R`` extension, kernel aliases, and user-overridable ansatz constants; rejected every removed manifest field with migration guidance.
- Routed PDF matching, DA conversion, and diagnostics through ``alphas_nloop(mu, order, Nf)`` and recorded the derived coupling and running-helper provenance in NetCDF artifacts and reports.
- Added strict/intersection target coverage policies with explicit range/drop provenance, kept exact lattice-spacing matching, and constrained fitted ``z_R`` to the single-family reference grid.
- Reduced the pion/kaon references to five MILC spacings and twenty points through 1.20 fm, reran both workflows, and verified all 18 outputs (600 samples by 20 points) exactly against the direct hybrid-self-renormalization formula.

## 2026-07-17 (Automatic apply-time zR extension)

- Added default apply-time long-distance extension for hybrid self-renormalization: when the target exceeds the fitted ``z_R`` grid, infer the single-family ``f1(z)``, fit its derived long-distance tail quadratically, and rebuild only the missing upper-end ``z_R`` points.
- Kept ``strict`` and ``intersection`` as explicit alternatives while requiring no user-supplied extension length or fit boundary; artifacts and reports record the source range, extrapolated-point count, tail boundary, and method.
- Clarified that the fit job determines the reference-operator ``m0``, while apply jobs continue to accept ``m0_gev`` and ``d`` overrides for the target operator.
- Restored all 18 pion/kaon DA outputs to 600 samples by 25 points, verified them exactly against the direct extrapolated formula, and removed tests and documentation for the retired partial pion-PDF manifest.
- Expanded the annotated sample manifest with every supported hybrid-self optional parameter, fit/apply scope, defaults, target-``m0`` override semantics, and coverage-policy choices.

## 2026-07-17 (Deterministic renormalization job completion)

- Removed ``order`` and ``Nf`` from the self-renormalization manifest, tools, MSbar conversion interfaces, provenance, reports, and examples; self-renormalization now derives its coupling through ``alphas_nloop(mu)`` while the general running helper remains available to matching.
- Rebuilt renormalization tool arguments exclusively from runner-owned manifest and store state so model-supplied role names, paths, and explicit null values cannot override job contracts.
- Enforced the scheme-specific renormalization tool sequence, rejected premature finish/input requests as recoverable observations, and made exhausted incomplete jobs fail instead of reporting a partial stage as completed.

## 2026-07-17 (Configurable hybrid-self LambdaQCD)

- Added ``scheme_parameters.LambdaQCD`` (GeV, default ``0.1``) to the hybrid-self-renormalization fit, apply, extension, remap, diagnostic, reporting, and artifact-provenance paths.
- Grouped the other hybrid-self-only manifest settings under ``scheme_parameters`` (``d``, ``svdcut``, and ``z_coverage_policy``), while keeping the scheme-shared ``m0_gev`` nested and leaving ``mu``/``kernel_id`` at the kernel-selection layer.
- Reject flat legacy placements and hybrid-self-only keys on ratio schemes, require apply jobs to agree with upstream ``zR`` LambdaQCD provenance, and migrated the pion/kaon examples plus schema, runner, planning, and numerical regression tests.

## 2026-07-17 (Required unit-explicit LambdaQCD)

- Renamed the hybrid-self parameter and artifact provenance field to ``scheme_parameters.LambdaQCD_gev`` so the GeV unit is explicit.
- Removed the internal ``0.1`` fallback: every hybrid-self fit and apply job must now declare ``LambdaQCD_gev``, and missing values fail stage validation before tool execution.
- Kept ``0.1`` explicitly in every pion/kaon DA fit and apply job, and added migration errors for the old ``LambdaQCD`` and lowercase ``lqcd`` names.

## 2026-07-17 (Recursive stage parameter inheritance)

- Replaced shallow defaults/job parameter composition with one shared recursive merge used by the runner, stage validation, planning, kinematics derivation, and review checks.
- Nested mappings now inherit and override by key, while job-level lists, scalars, and type changes still replace the complete default value.
- Moved the required pion/kaon DA ``scheme_parameters.LambdaQCD_gev: 0.1`` to renormalization defaults; fit/apply jobs now declare only their operator-specific ``d``, ``m0_gev``, ``svdcut``, or coverage overrides.

## 2026-07-18 (Unified qDA-ratio correlator fits)

- Replaced the standalone empirical DA 2pt-ratio tool with ``fit_scope: qda_ratio`` in the shared correlator tuning and grid-fit tools.
- Added a multi-state qDA numerator divided by the existing 2pt spectral function, with joint/chained strategies and ``O00/z0`` bare-matrix output.
- Replaced the old correlator scopes with ``3pt_ratio``, ``FH``, ``3pt_ratio+FH``, and ``qda_ratio`` and removed ``analysis_mode`` from the manifest and reporting interfaces.
- Fixed qDA normalization to the local ``bz=0`` correlator and made the ``bz`` grid runner-derived, removing the redundant ``reference_z`` and ``z_values`` stage parameters.
- Corrected the qDA denominator contract to use a separate ordinary local-source/local-sink 2pt correlator; ``bT`` and ``bz`` are now independent qDA HDF5 selectors rather than operator-name placeholders, with no special ``bz=0`` normalization branch.
- Added sample-0 qDA-ratio fit-on-data PDF/SVG diagnostics for every ``bz``, with posterior bands, component filtering, fit-log artifact collection, and report embedding aligned with 3pt fits.

## 2026-07-18 (qDA bz=0 denominator fallback)

- Made the ordinary local-local 2pt input optional for ``qda_ratio`` jobs; qDA-only jobs now use the same nonlocal operator at ``bz=0`` as the resampled denominator.
- Added the mixed source/sink overlap model ``z_n*zprime_n`` for chained and joint qDA fits, with ``O00/zprime0`` extraction while retaining the existing local-local path.
- Added compatibility for legacy qDA HDF5 paths whose nonlocal operator label carries ``bT`` and ``bz`` suffixes.
- Moved lattice-to-fm target conversion into hybrid self-renormalization and excluded ``z=0`` before coverage, ``zR``, and MSbar-factor evaluation.
- Converted ``examples/pion_da_gi_manifest.json`` from precomputed-bare renormalization to a two-stage qDA correlator-analysis and hybrid-self-renormalization workflow.

## 2026-07-18 (Hybrid-self z=0 output preservation)

- Kept normalized ``z=0`` target samples out of ``zR`` and MSbar-factor evaluation while passing them through unchanged into the renormalized output.
- Restored the complete physical-coordinate output grid with ``H^R(0)=1`` and separated zero-point passthrough provenance from genuine coverage drops.

## 2026-07-18 (Fast qDA unit coverage)

- Removed two qDA grid-fit tests that repeated full nonlinear fits across multiple coordinates and resamples, including a duplicate serial/parallel run, from the regular unit-test suite.
- Retained fast coverage for the qDA spectral formulas, mixed overlaps, ``bz=0`` fallback, legacy HDF5 layout, and fit plotting.

## 2026-07-19 (Short-distance self-renormalization reference)

- Generated local ignored sample-based ``(a,z)`` self-renormalization references from the legacy five-spacing pion PDF mean/error table.
- Replaced the inherited 0.06 fm grid with the positive pion-DA target-coordinate union inside common source coverage: 44 points from 0.0574 to 1.213 fm.
- Restored a06 ``bz=1`` in the pion DA correlator grids and regenerated identical pion/kaon reference artifacts while retaining apply-time long-distance ``zR`` extension.
- Corrected the shared pion/kaon DA reference provenance to ``gfix=GI`` and ``target_observable=pdf`` in the generated artifacts, with matching ``gfix`` declarations in the GI workflow manifests.

## 2026-07-19 (Full pion/kaon GI-DA workflows)

- Rebuilt the local ignored pion/kaon ``gZ5_nonlocal`` correlator files from the legacy ``DA_new.hdf5`` arrays with the physical 192/96/64 temporal extents.
- Expanded the kaon manifest from precomputed bare-matrix inputs to the same correlator-analysis, self-renormalization, Fourier, matching, and review chain used by pion.
- Kept the shared DA self-renormalization conversion parameters ``d=0.19`` and ``m0=-0.094 GeV`` and the light/light Fourier tail constraint for kaon.

## 2026-07-19 (Unified qDA correlator implementation and sample logs)

- Merged the qDA-ratio tuning, data-loading, worker, plotting, and grid-fit implementation into the correlator stage's `functions.py`, removing the circular implementation split through `qda.py`.
- Aligned qDA fit logging with the other correlator scopes by writing separate `_tuning.log` and `_samples.log` artifacts.
- Added per-z and per-sample qDA fit-quality, rejected-model, failed-sample, and summary records while preserving the existing fit and output contracts.
- Added stage-package hygiene guidance requiring tightly coupled stage helpers to remain in `functions.py` and dependencies between any additional modules to stay one-way.

## 2026-07-19 (Automatic correlator fit-window scans)

- Added bounded automatic 2pt windows from first-half resampled signal-to-noise diagnostics, with conservative NonBreit channel handling and explicit fallback metadata.
- Added automatic contiguous-`tsep`/`tau_cut` candidates that allow a single central insertion point, while preserving explicit `pt2_windows`, `pt3_windows`, and `pt3_tau_cuts` as exact overrides.
- Propagated `auto_window_scan` diagnostics through tuning, grid-fit logs, and bilingual reports, and stopped the full planner variant from synthesizing correlator windows.

## 2026-07-19 (qDA tune soft-fail and agent stop-on-input)

- Changed `tune_qda_ratio` so an empty cross-z window intersection returns
  `status="no_common_feasible_candidate"` with `succeeded_counts_by_z` and
  `retry_hint` instead of raising, and updated correlator prompts/skills to
  narrow `tune_z_values` at least once before `request_user_input`.
- Record LLM `request_user_input` into `pending_user_input` and stop the run
  instead of silently finishing the job; raise immediately when a job ends
  without `store["output"]` so downstream stages cannot start on holes.

## 2026-07-19 (Skip z=0 fit for nonlocal_bz0 qDA)

- For `qda_ratio` with `qda_denominator_mode="nonlocal_bz0"` only, skip fitting
  z=0 in tune/grid paths and reinject bare ME samples `1+0j` into the output
  NetCDF; `local` denominator mode still fits z=0.
- Renamed the denominator mode token `local_local` to `local`.
- Updated correlator prompts/skills and README to describe the gate and
  assigned-unity output.

## 2026-07-19 (Cap auto 2pt windows at the last valid data point)

- Fixed `_pt2_snr_endpoint` so the tail-point extension past the last
  SNR-passing timeslice can no longer reach zero-padded (zero-sdev) tail
  points; the endpoint is now additionally capped at `last_valid_t + 1`.
- Root cause: the pion GI-DA correlator files are zero-padded beyond the
  measured range (t<=23 on a06m130, t<=15 on a09m130/a12m130), so every
  automatic window included exactly-zero points and every tune fit failed
  with "Residuals are not finite in the initial point" at all tune z values.
- Added `last_valid_t` to the auto-window diagnostics and a unit test that a
  zero-padded tail bounds all generated window `tmax` values.

## 2026-07-19 (Widen automatic 2pt window scan)

- Expanded `_auto_pt2_windows` so `tmax` is evenly sampled (up to 4 values)
  from the shortest overdetermined length through the SNR `stable_tmax`, and
  `tmin` uses up to 4 evenly spaced candidates per `tmax`.
- Raised the auto-only candidate cap to 16 with even subsampling when the
  cartesian product exceeds that limit, avoiding the old `[:6]` short-`tmax`
  bias. Legacy `_normalise_pt2_windows(None, ...)` still uses the cap of 6.

## 2026-07-19 (Unique sample-0 fit-log plot names)

- Prefixed correlator sample-0 diagnostic plot stems with `{ensemble}_{tag}_`
  for qDA, 3pt ratio, FH, and chained 2pt plots so multi-ensemble jobs sharing
  one `fit_logs/` directory no longer overwrite identical momentum/bT/bz files.

## 2026-07-19 (Independent correlator fit strategy)

- Added `fit_strategy: independent` for ratio/FH/`qda_ratio` fits that skip any
  2pt channel and any prior 2pt fit (alongside `joint` and `chained`).
- Updated correlator prompts, skills, reporting, and README to document the
  three strategies.
- Set `examples/pion_da_gi_manifest.json` to one-state `independent` qDA fits
  with per-job `pt2_windows` of length 3/4/5 from paper physical \(t_{\min}\).

## 2026-07-19 (Agent vs reference DA comparison plots)

- Renamed `runs/ds_{pion,kaon}_da_gi/plot_renormalized_matrix_compare.py` to
  `plot_agent_data_compare.py` and expanded it to overlay agent artifacts against
  `data_*_da_gi` bare ME, renormalized ME, quasi-x, and lightcone DA.
- Exported large-\(P_z\) lightcone references from
  `temp/lamet_da_self_renorm/final_plots/dump/*_final_plot_mom_space.pkl` to
  `data_{pion,kaon}_da_gi/x_dependence/{pion,kaon}_DA_lightcone_x.txt` (TXT only).
- Comparison outputs go under `artifacts/comparison/` as SVG only (no PDF/TSV);
  missing agent stages are skipped with a warning so partial kaon runs still
  produce renorm overlays.
  
## 2026-07-19 (Codex backend model selection)

- Allowed `--model <model_id>` with the `codex` backend for both `run` and
  `plan`, while preserving the Codex SDK default when the option is omitted.
- Routed the selected model through structured agent decisions and free-form
  planning/report requests, and included explicit Codex models in run traces and
  summaries.
- Added unit coverage for CLI model resolution and Codex SDK `thread_start`
  model forwarding.

## 2026-07-23 (Review-stage literature context)

- Added a minimal review-stage SQLite literature lookup in `src/lamet_agent/stages/review/functions.py`.
- The review stage now reads background-only LaMET papers from `lamet-papers/data/lamet_arxiv.sqlite3` using a repository-relative path.
- Retrieved papers are selected from lightweight keyword matches against the manifest, stage report text, and stage SVG subpaths, then injected into the review prompt as qualitative literature context.
- Prompt rules now explicitly forbid using literature abstracts as numerical evidence for the current run; run-specific numbers must still come only from the manifest, stage reports, NetCDF summaries, and deterministic checks.
- Added `stages.review.defaults.literature` as a manifest-controlled boolean toggle.
- `literature=false` restores the pre-literature review prompt path, while `literature=true` enables background-only SQLite paper retrieval from `lamet-papers/data/lamet_arxiv.sqlite3`.
- Added schema and review-stage tests to verify that literature context is omitted when disabled and injected only when explicitly enabled.
- Refined the literature-enabled review prompt so that each stage may append one short literature-based context paragraph inside the Diagnostics subsection, rather than creating a separate literature section.
- Refined the literature-enabled review prompt so figure embedding keeps the original review-relative `markdown_path` convention (for example `../correlator_analysis/...`) and literature diagnostics may now cite the most relevant retrieved papers for qualitative signal/noise/systematics reasonableness checks without treating literature as run-specific numerical evidence.

## 2026-07-24 (Review literature anchor ranking)

- Replaced the review-stage literature retrieval heuristic with manifest-anchor weighting in `src/lamet_agent/stages/review/functions.py`.
- Literature ranking now prioritizes overlap with `target_observable`, `parton`, hadron channel, gauge fixing, renormalization scheme, matching order/method, boosted-momentum context, and lattice ensemble signals rather than report/SVG diagnostic keywords.
- The injected literature context now records stronger `matched_topics` and instructs the review prompt to prefer the papers with the closest manifest-physics overlap, while keeping literature strictly background-only.
- Added review-stage tests with a temporary SQLite paper table to verify that exact physics-channel matches outrank generic LaMET background papers.

## 2026-07-24 (Unified renormalization output coordinates)

- Converted ratio and hybrid-ratio terminal `z` coordinates from lattice units
  to signed physical fm using the target `lattice_spacing_fm`.
- Standardized their output provenance with `coord_unit="fm"` and
  `input_coord_unit="lattice"` to match hybrid self-renormalization artifacts.
- Added validation for missing or invalid lattice spacing and unit coverage for
  in-memory, legacy-array, and NetCDF outputs while preserving hybrid-ratio
  physics calculations.

## 2026-07-24 (Papers path and broader LaMET KB rules)

- Documented the in-repo literature knowledge base under `papers/` in the root `README.md`, including first bootstrap, update, and review-stage usage.
- Updated `papers/README.md` to use the renamed repository path and to describe the broader LaMET-related scope rather than only a narrow core set.
- Expanded `papers/config/relevance_config.json` so harvesting and scoring cover broader LaMET-related theory, lattice-analysis, and perturbative papers, including short-distance factorization and systematics-oriented queries.

## 2026-08-06 (Repair broken local .venv)

- Recreated the repository-root `.venv` after the previous environment pointed at a missing Miniconda interpreter (`/home/jinchen/miniconda3/bin/python`).
- Bootstrapped with system CPython 3.12.3 via `uv venv --seed` because `python3.12-venv` is not installed on this host.
- Reinstalled the editable package with `pip install -e ".[dev,analysis]"` and verified core imports plus the `lamet-agent` CLI entrypoint.
- Added missing `mpmath>=1.3` to the `analysis` optional dependency set so LRR matching kernels can import `mpmath.expint` after a fresh editable install.

## 2026-08-06 (Update stale unit tests)

- Updated `test_parse_api_model_provider_shorthand_uses_default_model` to expect DeepSeek default `deepseek-v4-flash`.
- Added explicit `momentum_gev=2.0` to Fourier unit tests that called `run_fourier_transform` under default `coord_unit=\"fm\"` without kinematics.

## 2026-08-06 (Drop unused zspz from ratio kernels)

- Removed the unused `zspz` parameter from all `*_ratio_NLO` matching kernels in `kernels.py` (CG/GI PDF and GI DA).
- Stopped forwarding `zspz` from ratio delegates and from CG transversity msbar/hybrid wrappers that call the ratio builder.
- Updated the matching-stage registry comment so it no longer claims a uniform `zspz=None` signature; hybrid still receives `zspz` via `is_hybrid_kernel`.

## 2026-08-06 (Split renormalization scheme and strategy)

- Made stage parameters authoritative for `scheme`; removed the duplicated field
  from kernel declarations and required matching scheme values to agree with the
  exact kernel-id token.
- Split renormalization into physical `ratio`/`hybrid`/`msbar` schemes and
  `ratio`/`self_renormalization` execution strategies.
- Added `msbar + self_renormalization` and a continuous
  `hybrid + self_renormalization` path with per-resample constant `Z_T` fixed at
  the nearest `zs_fm` grid point.
- Migrated tracked examples, planning/review/reporting behavior, documentation,
  and unit-test fixtures to the new contracts.

## 2026-08-07 (Fix pion PDF scheme_scan units)

- Converted `fourier_transform.scheme_scan` `zmin_values`/`zmax_values` in `examples/pion_pdf_cg_manifest.json` and `examples/pion_pdf_gi_manifest.json` from lattice indices to fm (`a=0.06`), matching renormalized NetCDF coordinates.

## 2026-08-07 (Document Fourier scheme_scan coord units)

- Audited `examples/` manifests: production DA/sys manifests already omit explicit zmin/zmax (auto-fill); only the annotated sample manifests still used lattice-like ranges under `coord_unit: "fm"`.
- Converted `scheme_scan` ranges in `examples/sample_manifest.jsonc` and `examples/partial_sample_manifest.jsonc` to fm using `a_fm=0.0574`, and clarified comments.
- Documented the `coord_unit` requirement for Fourier `scheme_scan` in `README.md`.

## 2026-08-07 (Rename renormalization strategy ratio → external_denominator)

- Renamed renormalization `strategy` token `ratio` to `external_denominator` across stage validation, tool routing, planning, examples, docs, and tests.
- Kept physical `scheme` `ratio` and tool name `apply_ratio_scheme_renormalization`; old `strategy: "ratio"` now returns an explicit migration error.

## 2026-08-09 (Distribution-aware PDF/GPD Fourier observables)

- Added the 3pt `distribution_type` contract (`unpolarized`, `helicity`, or
  `transversity`) and propagated it with `current_operator` through correlator,
  renormalization, partial-artifact, and Fourier metadata.
- Replaced the implicit transversity Fourier fallback with explicit/upstream/
  manifest observable resolution and canonical quark PDF/GPD observable labels.
- Implemented distribution-aware quark `sea`, `valence`, `singlet`, and `full`
  projections, including the helicity negative-x convention, while keeping
  gluon jobs full-only and limiting automatic gluon inference to the existing
  unpolarized-PDF backend.
- Updated Fourier reports with operator provenance, GPD-family and ERBL scope,
  retained ordinary-stage LLM translation for Chinese output, and preserved DA
  and review-stage language behavior.
- Synchronized plan-mode defaults, provenance gaps, prompts, examples, and unit
  coverage without adding a second observable-inference path.

## 2026-08-11 (Omit temperature for OpenAI GPT-5+/o-series)

- Chat Completions requests no longer send `temperature` for GPT-5+ and o-series
  model ids, which reject custom sampling params with HTTP 400.
- Kept `temperature: 0.0` for GPT-4o-class and DeepSeek models; covered both
  action and text completion request paths in unit tests.

## 2026-08-11 (Move stage prompts to Markdown resources)

- Consolidated each stage's instruction, strategy guidance, and tool catalog in
  a stage-local `prompts.md`; `skills.py` now retains validation and parameter
  helpers only.
- Switched prompt assembly to packaged Markdown resources, included those files
  in distributions, and added coverage for every registered stage.
- Restored the review-stage instruction to the assembled prompt; its former
  `STAGE_INSTRUCTION` name was not consumed by the Python prompt loader.

## 2026-08-11 (Structure stage prompt Markdown)

- Added consistent stage titles and `Basic Procedure`, `Stage Skill`, and
  `Available Tools` sections to every stage prompt for easier human review and
  editing.
- Added the previously implicit extrapolation tool catalog and covered the
  shared Markdown structure across all registered stages.

## 2026-08-11 (Mark prompt tool names as code)

- Wrapped stage tool function names in Markdown inline code spans across prompt
  procedures and tool catalogs for clearer human editing.
