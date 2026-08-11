# Continuum Extrapolation

## Basic Procedure

For ordinary extrapolation jobs call `run_extrapolation` once. For jobs with
operation=systematics_budget call `run_systematics_budget` once. The runner binds
perturbative-matching inputs from the lightcone role.

## Stage Skill

Fit matched light-cone distributions to IMF and/or continuum limits.

## Available Tools

- `run_extrapolation`: Fit matched light-cone data to the IMF and/or continuum limit.
- `run_systematics_budget`: Build a systematic-error budget from extrapolated outputs.
