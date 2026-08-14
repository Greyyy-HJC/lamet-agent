"""Regression checks for the local literature tag taxonomy and output."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "lamet_literature" / "classify_arxiv.py"
OUTPUT = ROOT / "lamet_literature" / "arxiv.json"


def test_literature_taxonomy_and_regression_records() -> None:
    """Keep TMD, current structure, polarization, and flavor tags separated."""

    pytest.importorskip("bs4")
    pytest.importorskip("requests")
    spec = importlib.util.spec_from_file_location("classify_arxiv", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.SCHEMA_VERSION == 2
    assert module.ARRAY_ENUMS["flavors"] == ["u", "d", "s", "c", "b", "light", "heavy"]
    assert "tmd" not in module.ARRAY_ENUMS["observables"]
    assert "tmd" in module.ARRAY_ENUMS["kinematic_dependence"]
    assert "mentioned only in the introduction" in module.SYSTEM_PROMPT
    assert "number of independent gauge configurations" in module.SYSTEM_PROMPT

    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    papers = {paper["arxiv_id"]: paper for paper in data["papers"]}
    assert data["schema_version"] == 2
    assert len(papers) == len(data["papers"])

    for paper in papers.values():
        for field, allowed in module.ARRAY_ENUMS.items():
            assert set(paper["tags"][field]) <= set(allowed)
        if paper["tags"]["currents"]:
            assert "three_point" in paper["tags"]["correlator_types"]

    assert "isovector" not in papers["1810.05043"]["tags"]["flavors"]
    assert papers["1810.05043"]["tags"]["currents"][0]["flavor_structure"] == "isovector"
    assert "unpolarized" not in papers["2404.14525"]["tags"]["flavors"]
    assert "unpolarized" in papers["2404.14525"]["tags"]["polarizations"]
    assert "g" not in papers["2412.20461"]["tags"]["flavors"]
    assert "gluon" in papers["2412.20461"]["tags"]["partons"]
    assert "lattice_cross_section" not in papers["2412.20461"]["tags"]["methods"]
