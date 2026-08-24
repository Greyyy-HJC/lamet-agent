# LaMET literature tagging

This directory contains the local literature corpus and the scripts used to
build review tags from it. `inspirehep.json` is the selected paper list,
`arxiv/` contains downloaded HTML, and `arxiv.json` is the review index.

Install the optional dependencies and run the classifier with the local
OpenAI-compatible server already listening on `127.0.0.1:8080`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[literature]"
.\.venv\Scripts\python.exe .\lamet_literature\classify_arxiv.py
```

The run is resumable. It skips papers already stored with the current schema.
Use `--force` only to replace selected classifications. A focused prompt check
can reclassify one or more records without dropping the rest of the index:

```powershell
.\.venv\Scripts\python.exe .\lamet_literature\classify_arxiv.py `
  --arxiv-id 1810.05043 `
  --arxiv-id 2404.14525 `
  --arxiv-id 2412.20461 `
  --force
```

The classifier sends an evidence packet rather than blindly truncating the
start of a long paper. The packet keeps the abstract, section outline, and
sections likely to contain the formalism, lattice setup, matching,
renormalization, results, and systematics. Structured output uses closed enums,
and a deterministic pass removes duplicates and physically invalid setup
values.

## Tag semantics

Standard physics abbreviations are intentional. The important field boundaries
are:

| Field | Meaning |
| --- | --- |
| `observables` | Physical objects such as `pdf`, `da`, `gpd`, `collins_soper_kernel`, `soft_function`, and `wave_function`. |
| `kinematic_dependence` | Modifiers such as `collinear`, `tmd`, `off_forward`, and `impact_parameter`. A TMDPDF is `pdf` plus `tmd`; `tmd` is not an observable. |
| `partons` | `quark` or `gluon`. The symbol `g` is never a quark flavor. |
| `flavors` | Only quark species: `u`, `d`, `s`, `c`, `b`, `light`, or `heavy`. |
| `quark_sectors` | `valence`, `sea`, `singlet`, or `nonsinglet`. |
| `polarizations` | `unpolarized`, `helicity`, or `transversity`. |
| `twist` | `twist_2`, `twist_3`, or `higher_twist`. |
| `correlator_types` | `two_point`, `three_point`, or `current_current`. |
| `currents` | Current insertions used in explicit three-point correlators. Each entry records `type`, `flavor_structure`, and the stated Dirac structure. `isovector` and `isoscalar` belong here, not in `flavors`. |
| `sea_quark_content` | Ensemble content such as `Nf=2+1+1`; it is not a flavor tag. |

`relevance` may be `core`, `secondary`, or `unrelated`. An unrelated record is
kept in the JSON so the broad initial search can be audited later.

Schema version 2 introduced these boundaries. The existing 128-record index was
conservatively migrated from version 1; a future full `--force` run will replace
those migrated tags with direct version-2 model classifications.

## Startup task for a Codex-style agent

Copy the following task into a fresh coding-agent session when the corpus needs
to be classified again or the taxonomy needs to be audited:

```text
Work in the lamet-agent repository and read AGENTS.md before making changes.

Goal: maintain lamet_literature/arxiv.json as a review-oriented classification
of the papers already selected in lamet_literature/inspirehep.json and already
downloaded under lamet_literature/arxiv/. Do not redownload papers and do not
query arXiv metadata. Use the local OpenAI-compatible model at
http://127.0.0.1:8080/v1 through lamet_literature/classify_arxiv.py.

Keep standard physics abbreviations such as PDF, DA, GPD, TMD, LaMET, RI/MOM,
MSbar, LO, NLO, and NNLO. Enforce these semantic boundaries:
- observables are physical objects; TMD is only kinematic_dependence;
- flavors contains only u, d, s, c, b, light, or heavy;
- g means the gluon parton, never a flavor;
- unpolarized, helicity, and transversity are polarizations;
- valence, sea, singlet, and nonsinglet are quark_sectors;
- isovector and isoscalar are current flavor structures;
- currents is populated only for an explicit three-point correlator;
- Nf values belong to an ensemble's sea_quark_content.

Treat these as regression cases: arXiv:1810.05043 must not store isovector in
flavors; arXiv:2404.14525 must not store unpolarized in flavors; and
arXiv:2412.20461 must represent g as gluon rather than a quark flavor.

First inspect the current schema, prompt, JSON vocabulary, and the three
regression records. Prefer targeted --arxiv-id --force calls before any full
rerun. Preserve resumability and atomic output. After changes, compile the
script, run the focused literature tests, verify that every INSPIRE arXiv id is
represented exactly once, check all controlled fields for out-of-vocabulary
values, and report any classification that remains uncertain. Update this
README and append PROJECT_LOG.md only when behavior or the schema changes.
```
