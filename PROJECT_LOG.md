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
