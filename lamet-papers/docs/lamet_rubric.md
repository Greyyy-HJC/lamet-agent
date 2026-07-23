# LaMET Relevance Rubric

## Goal

Build a topic-focused arXiv knowledge base for LaMET and closely adjacent
methods without downloading the entire arXiv full-text corpus.

## Labels

### `core`

Use for papers that directly study one or more of the following:

- LaMET itself
- quasi-PDF or quasi-distribution methods
- pseudo-PDF or pseudo-distribution methods
- Ioffe-time distribution approaches
- lattice cross section or hadronic tensor approaches used for x-dependent
  hadron structure
- direct lattice extraction of x-dependent PDFs, GPDs, TMDs, or distribution
  amplitudes
- matching, factorization, and renormalization developed specifically for the
  above methods
- perturbative LaMET work such as matching coefficients, factorization theorems,
  evolution kernels, anomalous dimensions, Collins-Soper kernels, and NLO/NNLO
  corrections when the paper is explicitly framed in LaMET, quasi-, pseudo-, or
  Ioffe-time language

### `secondary`

Use for papers that are not centered on LaMET, but directly support a LaMET
workflow:

- RI/MOM or related renormalization for nonlocal Wilson-line operators
- boosted-hadron techniques such as momentum smearing
- finite-momentum and higher-twist systematics
- matrix-element methodology tightly connected to x-dependent structure
- GPD, TMD, DA, gluon, helicity, or transversity studies when the paper is
  clearly framed in a quasi-, pseudo-, or LaMET-adjacent setup
- perturbative support work in `hep-ph` or `hep-th` that is not primarily a
  lattice calculation but directly develops the perturbative ingredients needed
  by a LaMET workflow

### `irrelevant`

Use for papers that match a broad query but do not actually belong in the
knowledge base:

- generic PDF phenomenology with no lattice x-dependent extraction link
- unrelated uses of `PDF`
- broad lattice structure papers that never touch LaMET-adjacent methods

## Scoring logic

- anchor terms carry the largest weights and usually determine `core`
- secondary terms add support but should not dominate without at least one
  topic anchor
- category bonuses favor `hep-lat`, then `hep-ph`, then `nucl-th`
- `hep-th` is also included because some perturbative LaMET development appears
  there rather than in `hep-ph`
- manual seed IDs can be listed in `config/manual_seeds.json`
- borderline records are retained in the database only if they reach the
  configured acceptance thresholds

## Query strategy

The crawler intentionally does not mirror full arXiv categories. Instead it:

1. searches a small set of topic-focused query groups
2. deduplicates records across queries
3. scores each paper locally
4. stores only `core` and `secondary` results

This keeps the project lightweight while remaining easy to re-run and update.
