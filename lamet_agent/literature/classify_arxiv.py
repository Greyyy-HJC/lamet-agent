"""Classify local arXiv HTML with a local OpenAI-compatible model.

Inputs are ``inspirehep.json`` and the files under ``arxiv/``. The output is
``arxiv.json``, containing controlled physics and lattice tags for review.

Example usage::

    .venv\\Scripts\\python.exe lamet_literature\\classify_arxiv.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import requests

from lamet_agent.ui import log


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
PROMPT_VERSION = "2.3"

SYSTEM_PROMPT = """Classify this paper for a LaMET lattice-QCD literature database.
Use only facts supported by the supplied paper. Never infer numerical lattice setup
from typical values. Use empty arrays or null when information is absent. Tag what
this paper itself calculates or derives; a review may receive broader topical tags,
but its cited ensembles must not be reported as its own lattice setup.

Tag primary work, not vocabulary. A method mentioned only in the introduction,
related-work discussion, citation, or comparison is not a method used by this paper.
Likewise, do not tag an operator, workflow stage, or lattice ensemble merely because
the paper cites another calculation that used it.

Relevance:
- core: directly studies LaMET, quasi/pseudo distributions, Ioffe-time methods,
  lattice x-dependent structure, or matching/renormalization specific to them.
- secondary: directly supports those workflows but is not centered on them.
- unrelated: matched the broad initial search but is not useful for a LaMET review.

Paper type:
- lattice_calculation: the original result is obtained from lattice data. Routine
  perturbative matching of that result does not make the paper mixed.
- perturbative_theory: the paper derives original matching kernels, factorization,
  renormalization, or other perturbative results. A one-loop matching calculation is
  perturbative_theory, not a review.
- methodology: the original result is primarily a new analysis or reconstruction
  method rather than a lattice measurement or perturbative derivation.
- mixed: the paper has multiple original-result components of comparable importance.
- review: the paper synthesizes prior results and has no original lattice or theory
  result. Do not use review just because the introduction surveys earlier work.

Field boundaries are strict:
- observables identifies the physical object: pdf, da, gpd, collins_soper_kernel,
  soft_function, or wave_function. Never put tmd in observables.
  pdf means a parton distribution in a hadron and includes unpolarized, helicity,
  and transversity PDFs. da means a distribution amplitude from a vacuum-to-hadron
  matrix element, not any quantity whose name merely contains "distribution". gpd
  requires off-forward kinematics.
- kinematic_dependence contains collinear, tmd, off_forward, or impact_parameter.
  A TMDPDF is observables=[pdf] plus kinematic_dependence=[tmd]. A TMD wave function
  is observables=[wave_function] plus kinematic_dependence=[tmd].
- partons contains quark and/or gluon. The symbol g means gluon, never a flavor.
- flavors contains only actual quark species: u, d, s, c, b, light, or heavy.
  Do not put isovector, isoscalar, unpolarized, helicity, transversity, valence,
  sea, singlet, Nf=2+1, an action name, or g in flavors.
  If an explicitly isovector u-d current is studied, flavors may contain u and d,
  while isovector still goes in current.flavor_structure.
- currents describes currents inserted in three-point functions. Current type is
  vector, axial_vector, tensor, scalar, pseudoscalar, or gluon.
  isovector/isoscalar belongs in current.flavor_structure, not flavors.
  Leave currents empty unless a three-point current is explicit in this paper.
- polarizations contains unpolarized, helicity, or transversity. Twist is not a
  polarization; put twist_2, twist_3, or higher_twist in twist.
- quark_sectors contains valence, sea, singlet, or nonsinglet.
  Use valence or sea only when that sector is an explicit target; a valence/sea
  fermion action or the negative-x region is not sufficient evidence.
- Nf=2, Nf=2+1, and Nf=2+1+1 describe sea_quark_content in an ensemble, not flavors.

Correlator boundaries are also strict:
- two_point is a source-sink or vacuum-to-hadron two-point correlator.
- three_point is a hadron source, one local or nonlocal operator insertion, and a
  hadron sink. A straight Wilson-line quark bilinear inserted between hadron states
  is ordinarily a three-point correlator, not current_current.
- current_current requires a product of two separately inserted currents and must
  also have operators=[..., current_current, ...]. Do not infer it from the generic
  word current or from an introductory list of alternative PDF methods.
- current_current_correlator, hadronic_tensor, and lattice_cross_section are methods
  only when this paper directly computes or develops them. Ordinary quasi-PDF
  matrix elements do not receive those tags.

Concrete corrections:
- arXiv:1810.05043 studies a transversity PDF, not a DA; isovector is the tensor
  current's flavor structure, while the actual quark flavors are u and d.
- arXiv:2404.14525 studies an unpolarized valence PDF; unpolarized is a
  polarization, not a quark flavor. Its Wilson-line matrix element is not the
  current-current method merely mentioned as background in the introduction.
- arXiv:2412.20461 derives perturbative matching for singlet quasi-PDFs; g denotes
  the gluon parton, not a quark flavor. It does not analyze lattice correlator data,
  so do not tag a two-point/three-point correlator, current insertion, or the
  lattice_cross_section method.

Other controlled tags:
- methods: lamet, quasi_distribution, pseudo_distribution, ioffe_time,
  lattice_cross_section, hadronic_tensor, current_current_correlator, gradient_flow,
  short_distance_factorization, operator_product_expansion, gaussian_process.
- workflow_stages: correlator_analysis, renormalization, fourier_transform,
  perturbative_matching, extrapolation, review.
- renormalization_schemes: ratio, hybrid, ri_mom, modified_ri_mom, msbar,
  self_renormalization, gradient_flow, reduced_itd, nonperturbative.
- matching_orders: LO, NLO, NNLO, N3LO, resummed, nonperturbative, not_stated.
- gauge_choices: gauge_invariant, coulomb, landau, axial, general_covariant,
  not_stated.
- operators: straight_wilson_line, staple_wilson_line, coulomb_gauge_equal_time,
  local_current, current_current, three_quark_operator, gluon_operator.

Use paper_type=review only when the paper synthesizes prior literature without an
original lattice or theory result. Set uses_lattice_data=true only when this paper
itself analyzes lattice data. physical_pion_mass=true requires an explicitly stated
pion mass near 135--140 MeV. Source-sink separation means a three-point-correlator
separation, not Wilson-line length or flow time. Keep review_summary to at most two
factual sentences. Evidence must contain at most four short phrases supporting the
most important classifications or setup values. configuration_counts contains only
an explicitly stated number of independent gauge configurations. Never put the
number of measurements, sources, inversions, samples, or an O(10^N) statistics count
there. Keep it empty when the configuration count is not stated. Distinguish fermion
and gauge actions; do not guess either action from the collaboration name."""

ARRAY_ENUMS = {
    "hadrons": [
        "proton",
        "neutron",
        "nucleon",
        "pion",
        "kaon",
        "rho",
        "j_psi",
        "d_meson",
        "b_meson",
        "delta_baryon",
        "deuteron",
        "dibaryon",
        "baryon",
        "meson",
    ],
    "observables": [
        "pdf",
        "da",
        "gpd",
        "collins_soper_kernel",
        "soft_function",
        "wave_function",
    ],
    "kinematic_dependence": ["collinear", "tmd", "off_forward", "impact_parameter"],
    "partons": ["quark", "gluon"],
    "flavors": ["u", "d", "s", "c", "b", "light", "heavy"],
    "quark_sectors": ["valence", "sea", "singlet", "nonsinglet"],
    "polarizations": ["unpolarized", "helicity", "transversity"],
    "twist": ["twist_2", "twist_3", "higher_twist"],
    "correlator_types": ["two_point", "three_point", "current_current"],
    "methods": [
        "lamet",
        "quasi_distribution",
        "pseudo_distribution",
        "ioffe_time",
        "lattice_cross_section",
        "hadronic_tensor",
        "current_current_correlator",
        "gradient_flow",
        "short_distance_factorization",
        "operator_product_expansion",
        "gaussian_process",
    ],
    "workflow_stages": [
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
        "review",
    ],
    "renormalization_schemes": [
        "ratio",
        "hybrid",
        "ri_mom",
        "modified_ri_mom",
        "msbar",
        "self_renormalization",
        "gradient_flow",
        "reduced_itd",
        "nonperturbative",
    ],
    "matching_orders": ["LO", "NLO", "NNLO", "N3LO", "resummed", "nonperturbative", "not_stated"],
    "gauge_choices": ["gauge_invariant", "coulomb", "landau", "axial", "general_covariant", "not_stated"],
    "operators": [
        "straight_wilson_line",
        "staple_wilson_line",
        "coulomb_gauge_equal_time",
        "local_current",
        "current_current",
        "three_quark_operator",
        "gluon_operator",
    ],
    "systematics": [
        "excited_state_contamination",
        "finite_volume",
        "discretization",
        "continuum_extrapolation",
        "finite_momentum",
        "higher_twist",
        "target_mass",
        "long_distance_truncation",
        "fourier_reconstruction",
        "statistical_noise",
        "renormalization_uncertainty",
        "matching_truncation",
        "operator_mixing",
        "power_divergence",
        "linear_divergence",
        "gribov_copies",
        "gauge_fixing",
        "factorization_power_corrections",
        "inverse_problem",
        "model_dependence",
        "finite_flow_time",
    ],
}

TAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevance": {"type": "string", "enum": ["core", "secondary", "unrelated"]},
        "paper_type": {
            "type": "string",
            "enum": [
                "lattice_calculation",
                "perturbative_theory",
                "methodology",
                "review",
                "phenomenology",
                "mixed",
                "other",
            ],
        },
        "review_summary": {"type": "string"},
        "tags": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": "array", "items": {"type": "string", "enum": values}}
                    for name, values in ARRAY_ENUMS.items()
                },
                "currents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "vector",
                                    "axial_vector",
                                    "tensor",
                                    "scalar",
                                    "pseudoscalar",
                                    "gluon",
                                    "not_stated",
                                ],
                            },
                            "flavor_structure": {
                                "type": "string",
                                "enum": [
                                    "isovector",
                                    "isoscalar",
                                    "singlet",
                                    "nonsinglet",
                                    "flavor_diagonal",
                                    "not_stated",
                                ],
                            },
                            "dirac_structure": {"type": ["string", "null"]},
                        },
                        "required": ["type", "flavor_structure", "dirac_structure"],
                    },
                },
            },
            "required": [*ARRAY_ENUMS, "currents"],
        },
        "lattice_setup": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "uses_lattice_data": {"type": "boolean"},
                "ensembles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "fermion_action": {"type": ["string", "null"]},
                            "gauge_action": {"type": ["string", "null"]},
                            "sea_quark_content": {
                                "type": ["string", "null"],
                                "enum": ["Nf=2", "Nf=2+1", "Nf=2+1+1", "quenched", "not_stated", None],
                            },
                            "lattice_spacings_fm": {"type": "array", "items": {"type": "number"}},
                            "volumes": {"type": "array", "items": {"type": "string"}},
                            "pion_masses_mev": {"type": "array", "items": {"type": "number"}},
                            "configuration_counts": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": [
                            "name",
                            "fermion_action",
                            "gauge_action",
                            "sea_quark_content",
                            "lattice_spacings_fm",
                            "volumes",
                            "pion_masses_mev",
                            "configuration_counts",
                        ],
                    },
                },
                "momenta_gev": {"type": "array", "items": {"type": "number"}},
                "source_sink_separations_fm": {"type": "array", "items": {"type": "number"}},
                "continuum_extrapolation": {"type": ["boolean", "null"]},
                "physical_pion_mass": {"type": ["boolean", "null"]},
            },
            "required": [
                "uses_lattice_data",
                "ensembles",
                "momenta_gev",
                "source_sink_separations_fm",
                "continuum_extrapolation",
                "physical_pion_mass",
            ],
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "relevance",
        "paper_type",
        "review_summary",
        "tags",
        "lattice_setup",
        "evidence",
        "confidence",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tag local arXiv HTML with a local LLM.")
    parser.add_argument("--input", type=Path, default=SCRIPT_DIRECTORY / "inspirehep.json")
    parser.add_argument("--html-dir", type=Path, default=SCRIPT_DIRECTORY / "arxiv")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIRECTORY / "arxiv.json")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--arxiv-id",
        action="append",
        dest="arxiv_ids",
        help="Classify only this arXiv id; repeat for multiple papers.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-chars", type=int, default=80_000)
    args = parser.parse_args(argv)

    inspire_records = json.loads(args.input.read_text(encoding="utf-8"))
    papers = []
    seen = set()
    for record in inspire_records:
        metadata = record["metadata"]
        arxiv_id = metadata.get("arxiv_eprints", [{}])[0].get("value")
        if not arxiv_id or arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join(metadata["titles"][0]["title"].split()),
                "abstract": " ".join(metadata.get("abstracts", [{}])[0].get("value", "").split()),
            }
        )
    catalog_papers = papers
    if args.arxiv_ids:
        requested_ids = set(args.arxiv_ids)
        papers = [paper for paper in papers if paper["arxiv_id"] in requested_ids]
        missing_ids = requested_ids - {paper["arxiv_id"] for paper in papers}
        if missing_ids:
            parser.error(f"arXiv ids not found in input: {', '.join(sorted(missing_ids))}")
    if args.limit is not None:
        papers = papers[: args.limit]

    classified = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("schema_version") == SCHEMA_VERSION:
            classified = {paper["arxiv_id"]: paper for paper in existing.get("papers", [])}
        elif not args.force:
            parser.error(
                f"{args.output} uses schema_version={existing.get('schema_version')}; "
                f"use --force to rebuild schema_version={SCHEMA_VERSION}"
            )

    pending_papers = [paper for paper in papers if args.force or paper["arxiv_id"] not in classified]
    if not pending_papers:
        log(f"All {len(papers)} selected papers are already classified.")
        return 0

    model_response = requests.get(f"{args.api_base}/models", timeout=30)
    model_response.raise_for_status()
    model = args.model or model_response.json()["data"][0]["id"]

    with requests.Session() as session:
        for index, paper in enumerate(papers, start=1):
            arxiv_id = paper["arxiv_id"]
            if arxiv_id in classified and not args.force:
                log(f"[{index}/{len(papers)}] {arxiv_id}: skipped")
                continue

            html_path = args.html_dir / f"{arxiv_id.replace('/', '_')}.html"
            soup = BeautifulSoup(html_path.read_bytes(), "html.parser")
            document = soup.select_one(".ltx_document")
            source_scope = "full_text" if document is not None else "abstract_page"
            document = document or soup
            for tag in document.select("script, style, nav, header, footer, .ltx_bibliography, .ltx_page_footer"):
                tag.decompose()

            full_text = "\n".join(line.strip() for line in document.get_text("\n").splitlines() if line.strip())
            sections = document.select("section.ltx_section")
            if source_scope == "full_text" and sections:
                headings = [
                    heading.get_text(" ", strip=True)
                    for section in sections
                    if (heading := section.find(["h1", "h2"], recursive=False))
                ]
                excerpts = ["SECTION OUTLINE:\n" + "\n".join(headings)]
                abstract = document.select_one(".ltx_abstract")
                if abstract is not None:
                    excerpts.insert(0, abstract.get_text("\n", strip=True))
                section_keywords = (
                    "formalism",
                    "method",
                    "operator",
                    "correlator",
                    "lattice",
                    "simulation",
                    "ensemble",
                    "setup",
                    "renormal",
                    "matching",
                    "factorization",
                    "result",
                    "discussion",
                    "systematic",
                    "extrapol",
                    "continuum",
                    "conclusion",
                    "summary",
                )
                for section in sections:
                    heading = section.find(["h1", "h2"], recursive=False)
                    if heading is None or not any(
                        keyword in heading.get_text(" ", strip=True).lower() for keyword in section_keywords
                    ):
                        continue
                    section_text = "\n".join(
                        line.strip() for line in section.get_text("\n").splitlines() if line.strip()
                    )
                    if len(section_text) > 12_000:
                        section_text = section_text[:9_000] + "\n[...]\n" + section_text[-3_000:]
                    excerpts.append(section_text)
                paper_text = "\n\n".join(excerpts)
                if len(paper_text) > args.max_chars:
                    paper_text = paper_text[: args.max_chars - 15_000] + "\n[...]\n" + paper_text[-15_000:]
            else:
                paper_text = full_text[: args.max_chars]

            metadata_text = f"{paper['title']} {paper['abstract']}".lower()

            response = session.post(
                f"{args.api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"ARXIV ID: {arxiv_id}\n"
                                f"INSPIRE TITLE: {paper['title']}\n"
                                f"INSPIRE ABSTRACT: {paper['abstract']}\n"
                                f"SOURCE SCOPE: {source_scope}\n\n"
                                f"PAPER EVIDENCE PACKET:\n{paper_text}"
                            ),
                        },
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "lamet_paper_tags_v2",
                            "strict": True,
                            "schema": TAG_SCHEMA,
                        },
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=600,
            )
            response.raise_for_status()
            result = json.loads(response.json()["choices"][0]["message"]["content"])

            if not result["lattice_setup"]["uses_lattice_data"]:
                result["lattice_setup"] = {
                    "uses_lattice_data": False,
                    "ensembles": [],
                    "momenta_gev": [],
                    "source_sink_separations_fm": [],
                    "continuum_extrapolation": None,
                    "physical_pion_mass": None,
                }
            else:
                if result["paper_type"] == "review":
                    result["paper_type"] = "lattice_calculation"
                elif result["paper_type"] == "other":
                    result["paper_type"] = "lattice_calculation"
                for ensemble in result["lattice_setup"]["ensembles"]:
                    for field in ["name", "fermion_action", "gauge_action", "sea_quark_content"]:
                        if ensemble[field] in {"", "unknown", "not_applicable", "not_stated"}:
                            ensemble[field] = None
                    ensemble["lattice_spacings_fm"] = [
                        value for value in ensemble["lattice_spacings_fm"] if 0.02 <= value <= 0.3
                    ]
                    ensemble["volumes"] = [
                        value for value in ensemble["volumes"] if value not in {"", "unknown", "not_stated"}
                    ]
                    ensemble["pion_masses_mev"] = [
                        value for value in ensemble["pion_masses_mev"] if 100 <= value <= 1000
                    ]
                    ensemble["configuration_counts"] = [
                        value
                        for value in ensemble["configuration_counts"]
                        if value > 0
                        and (
                            re.search(
                                rf"configuration(?:s)?[^.\n]{{0,120}}\b{value}\b",
                                full_text.replace(",", ""),
                                re.IGNORECASE,
                            )
                            or re.search(
                                rf"\b{value}\b[^.\n]{{0,120}}(?:gauge\s+)?configuration(?:s)?",
                                full_text.replace(",", ""),
                                re.IGNORECASE,
                            )
                        )
                    ]
                result["lattice_setup"]["momenta_gev"] = [
                    value for value in result["lattice_setup"]["momenta_gev"] if 0 <= value <= 20
                ]
                result["lattice_setup"]["source_sink_separations_fm"] = [
                    value for value in result["lattice_setup"]["source_sink_separations_fm"] if 0 < value <= 5
                ]
                pion_masses = [
                    mass for ensemble in result["lattice_setup"]["ensembles"] for mass in ensemble["pion_masses_mev"]
                ]
                if pion_masses:
                    result["lattice_setup"]["physical_pion_mass"] = any(125 <= mass <= 145 for mass in pion_masses)

            if (
                result["paper_type"] == "review"
                and not result["lattice_setup"]["uses_lattice_data"]
                and re.search(
                    r"\b(?:one|two|three)[ -]loop\b|matching (?:factor|kernel)",
                    paper["title"],
                    re.IGNORECASE,
                )
            ):
                result["paper_type"] = "perturbative_theory"

            if (
                "da" in result["tags"]["observables"]
                and "distribution amplitude" not in metadata_text
                and re.search(
                    r"\bpdfs?\b|parton distribution|transversity distribution|helicity distribution",
                    metadata_text,
                )
            ):
                result["tags"]["observables"] = [value for value in result["tags"]["observables"] if value != "da"]
                result["tags"]["observables"].append("pdf")

            if not result["lattice_setup"]["uses_lattice_data"]:
                result["tags"]["correlator_types"] = []
                result["tags"]["currents"] = []

            if "current_current" not in result["tags"]["operators"]:
                result["tags"]["correlator_types"] = [
                    value for value in result["tags"]["correlator_types"] if value != "current_current"
                ]
                result["tags"]["methods"] = [
                    value
                    for value in result["tags"]["methods"]
                    if value not in {"current_current_correlator", "hadronic_tensor"}
                ]
            if "lattice_cross_section" in result["tags"]["methods"] and not re.search(
                r"lattice.{0,2}cross.{0,2}section(?:s)?",
                full_text,
                re.IGNORECASE,
            ):
                result["tags"]["methods"].remove("lattice_cross_section")
            if (
                result["lattice_setup"]["uses_lattice_data"]
                and {"pdf", "gpd"} & set(result["tags"]["observables"])
                and {"straight_wilson_line", "local_current", "gluon_operator"} & set(result["tags"]["operators"])
                and "current_current" not in result["tags"]["correlator_types"]
                and "three_point" not in result["tags"]["correlator_types"]
            ):
                result["tags"]["correlator_types"].append("three_point")
                if not result["tags"]["currents"]:
                    flavor_structure = "flavor_diagonal" if result["tags"]["flavors"] else "not_stated"
                    current_types = []
                    if "unpolarized" in result["tags"]["polarizations"]:
                        current_types.append("vector")
                    if "helicity" in result["tags"]["polarizations"]:
                        current_types.append("axial_vector")
                    if "transversity" in result["tags"]["polarizations"]:
                        current_types.append("tensor")
                    if "gluon_operator" in result["tags"]["operators"] and "gluon" in result["tags"]["partons"]:
                        current_types.append("gluon")
                    result["tags"]["currents"] = [
                        {
                            "type": current_type,
                            "flavor_structure": flavor_structure,
                            "dirac_structure": None,
                        }
                        for current_type in current_types or ["not_stated"]
                    ]

            for field in ARRAY_ENUMS:
                result["tags"][field] = list(dict.fromkeys(result["tags"][field]))
            result["tags"]["currents"] = list(
                {json.dumps(current, sort_keys=True): current for current in result["tags"]["currents"]}.values()
            )
            if "three_point" not in result["tags"]["correlator_types"]:
                result["tags"]["currents"] = []

            review_topics = [f"paper_type={result['paper_type']}"]
            topic_fields = [
                ("observables", "target_observable"),
                ("kinematic_dependence", "kinematic_dependence"),
                ("partons", "parton"),
                ("hadrons", "hadron"),
                ("flavors", "flavor"),
                ("quark_sectors", "quark_sector"),
                ("polarizations", "polarization"),
                ("twist", "twist"),
                ("correlator_types", "correlator_type"),
                ("methods", "method"),
                ("workflow_stages", "stage"),
                ("renormalization_schemes", "scheme"),
                ("matching_orders", "matching"),
                ("gauge_choices", "gfix"),
                ("operators", "operator"),
                ("systematics", "systematic"),
            ]
            for field, prefix in topic_fields:
                review_topics.extend(f"{prefix}={value}" for value in result["tags"][field])
            review_topics.extend(
                f"current={current['type']}:{current['flavor_structure']}" for current in result["tags"]["currents"]
            )
            if result["lattice_setup"]["uses_lattice_data"]:
                review_topics.append("uses_lattice_data=true")
            for ensemble in result["lattice_setup"]["ensembles"]:
                review_topics.extend(f"lattice_spacing_fm={value:g}" for value in ensemble["lattice_spacings_fm"])
                review_topics.extend(f"volume={value}" for value in ensemble["volumes"])
                review_topics.extend(f"pion_mass_mev={value:g}" for value in ensemble["pion_masses_mev"])
                if ensemble["sea_quark_content"]:
                    review_topics.append(f"sea_quark_content={ensemble['sea_quark_content']}")
            review_topics.extend(f"momentum_gev={value:g}" for value in result["lattice_setup"]["momenta_gev"])

            classified[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "source_file": html_path.name,
                "source_scope": source_scope,
                **result,
                "review_topics": list(dict.fromkeys(review_topics)),
            }
            output = {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "papers": [classified[item["arxiv_id"]] for item in catalog_papers if item["arxiv_id"] in classified],
            }
            temporary_output = args.output.with_suffix(".json.tmp")
            temporary_output.write_text(
                json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary_output.replace(args.output)
            log(f"[{index}/{len(papers)}] {arxiv_id}: {result['relevance']} ({result['paper_type']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
