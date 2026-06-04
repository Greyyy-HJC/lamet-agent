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
