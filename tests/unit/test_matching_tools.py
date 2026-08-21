from pathlib import Path
import tempfile

import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import validate_stage_diagnostics
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.matching.functions import (
    KERNEL_REGISTRY,
    apply_matching as _apply_matching,
    build_matching_kernel as _build_matching_kernel,
    load_quasi_pdf as _load_quasi_pdf,
    plot_matched_pdf,
    resolve_kernel_id,
)


def load_quasi_pdf(store, **kwargs):
    return _load_quasi_pdf(store, **kwargs)


def apply_matching(store, **kwargs):
    kwargs.setdefault("artifacts_dir", tempfile.mkdtemp())
    return _apply_matching(store, **kwargs)


def build_matching_kernel(store, **kwargs):
    kwargs.setdefault("mu", 2.0)
    y_ls = np.asarray(store["quasi_y_ls"], dtype=float)
    kwargs.setdefault("lc_x_ls", {"start": float(np.min(y_ls)), "stop": float(np.max(y_ls))})
    return _build_matching_kernel(store, **kwargs)


def _quasi_on(x_grid: np.ndarray, *, n_sample: int = 4) -> EnsembleData:
    """A smooth quasi-PDF over ``x_grid``, so interpolation error stays measurable."""
    return EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.exp(-(x_grid**2)) + 0.001 * idx for idx in range(n_sample)],
        dims=("x",),
        coords={"x": x_grid.tolist()},
        name="quasi_pdf",
    )


def test_resolve_registered_hybrid_kernel() -> None:
    kernel_id = "CG_gt_quark_PDF_hybrid_NLO"
    assert resolve_kernel_id(kernel_id, "hybrid") == kernel_id


def test_kernel_registry_ids_match_kernels_module_function_names() -> None:
    assert all(kernel_id == builder.__name__ for kernel_id, builder in KERNEL_REGISTRY.items())


def test_matching_grids_are_required() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}
    load_quasi_pdf(store)

    with pytest.raises(TypeError, match="lc_x_ls"):
        _build_matching_kernel(store, kernel_id="CG_gt_quark_PDF_ratio_NLO", momentum_gev=1.5)
    with pytest.raises(TypeError, match="mu"):
        _build_matching_kernel(
            store,
            kernel_id="CG_gt_quark_PDF_ratio_NLO",
            momentum_gev=1.5,
            lc_x_ls={"start": -2.0, "stop": 2.0},
        )


def test_load_quasi_pdf_uses_the_fourier_artifact_x_grid() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}

    load_quasi_pdf(store)

    assert np.array_equal(store["quasi_ed"].values, _quasi_on(native).values)
    assert np.array_equal(store["quasi_y_ls"], native)


def test_lc_x_ls_window_slices_the_quasi_nodes() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}
    load_quasi_pdf(store)
    build_matching_kernel(
        store,
        kernel_id="CG_gt_quark_PDF_ratio_NLO",
        momentum_gev=1.5,
        lc_x_ls={"start": -1.0, "stop": 1.0},
    )
    apply_matching(store)

    expected = native[(native >= -1.0) & (native <= 1.0)]
    assert store["kernel_matrix"].shape == (expected.size, native.size)
    assert store["lightcone_ed"].values.shape == (4, expected.size)
    assert np.allclose(store["lightcone_ed"].coords["x"], expected)


def test_formula_cache_does_not_serve_one_kernel_another_kernels_formula(tmp_path) -> None:
    from lamet_agent.stages.matching import reporting

    # CG_gt_... and GI_gt_... share an operator and a scheme but are different kernels
    # from different papers, so a cache keyed on the parsed fields would hand the second
    # one the first one's formula -- silently, since nothing downstream re-checks it.
    calls: list[str] = []

    def fake_request(*_args, **kwargs):
        prompt = "".join(m["content"] for m in kwargs["messages"])
        assert "request_timeout_seconds" not in kwargs
        assert "request_attempts" not in kwargs
        kernel = "GI" if "def C_ratio_gi" in prompt else "CG"
        calls.append(kernel)
        return f"formula for {kernel}"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(reporting, "request_llm_text", fake_request)
    monkey.setattr(reporting, "_fetch_paper_text", lambda *_a, **_k: "paper latex")
    # Both disk layers have to be neutralized, or the assertion below counts a hit on a
    # shipped/leftover formula instead of the call it means to count: the bundled
    # directory is redirected somewhere empty, and the writable one is switched off.
    monkey.setattr(reporting, "_BUNDLED_FORMULA_DIR", tmp_path / "bundled")
    monkey.setenv("LAMET_FORMULA_CACHE_DIR", "")
    reporting._FORMULA_CACHE.clear()
    try:
        llm = reporting.FormulaLlm(backend="api", provider="deepseek", api_key="k", model_name="m")
        cg, _ = reporting._llm_kernel_formula("CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm)
        gi, _ = reporting._llm_kernel_formula("GI_gt_quark_PDF_hybrid_NLO", language="en", llm=llm)
        assert cg == "formula for CG"
        assert gi == "formula for GI"
        assert calls == ["CG", "GI"], "each kernel must get its own call, not a cache hit"

        # The cache still has to work: the same kernel twice is one call.
        again, _ = reporting._llm_kernel_formula("CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm)
        assert again == "formula for CG" and calls == ["CG", "GI"]
    finally:
        monkey.undo()
        reporting._FORMULA_CACHE.clear()


def test_matching_formula_failure_keeps_numerical_report_available(monkeypatch) -> None:
    from lamet_agent.stages.matching import reporting

    def fail_formula(*_args, **_kwargs):
        raise RuntimeError("provider timeout")

    monkeypatch.setattr(reporting, "_llm_kernel_formula", fail_formula)

    with pytest.warns(UserWarning, match="numerical matching output is complete"):
        text = reporting._matching_formula_text(
            {"kernel_id": "CG_gt_quark_PDF_hybrid_NLO"},
            language="en",
            llm=reporting.FormulaLlm(),
        )

    assert "report-only LLM request failed" in text
    assert "numerical matching matrix and output artifact" in text


def test_formula_disk_cache_outlives_the_process_and_then_needs_no_llm(tmp_path) -> None:
    # The whole point of the disk layer: the second run of `lamet-agent` must not repeat
    # the paper download and the ~27k-token prompt. Simulate a fresh process by clearing
    # the in-process cache, then make any LLM call an error -- a hit is the only way through.
    from lamet_agent.stages.matching import reporting

    monkey = pytest.MonkeyPatch()
    monkey.setattr(reporting, "_BUNDLED_FORMULA_DIR", tmp_path / "bundled")
    monkey.setenv("LAMET_FORMULA_CACHE_DIR", str(tmp_path / "user"))
    monkey.setattr(reporting, "_fetch_paper_text", lambda *_a, **_k: "paper latex")
    monkey.setattr(reporting, "request_llm_text", lambda *_a, **_k: "generated formula")
    reporting._FORMULA_CACHE.clear()
    llm = reporting.FormulaLlm(backend="api", provider="deepseek", api_key="k", model_name="m")
    try:
        first, paper_used = reporting._llm_kernel_formula(
            "CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm
        )
        assert (first, paper_used) == ("generated formula", True)
        assert (tmp_path / "user" / "CG_gt_quark_PDF_hybrid_NLO.en.md").exists()

        reporting._FORMULA_CACHE.clear()  # a new process starts here

        def explode(*_args, **_kwargs):
            raise AssertionError("a cached formula must not be regenerated")

        monkey.setattr(reporting, "request_llm_text", explode)
        monkey.setattr(reporting, "_fetch_paper_text", explode)
        second, second_paper_used = reporting._llm_kernel_formula(
            "CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm
        )
        # The provenance flag has to survive too, or the report would credit the paper for
        # a formula derived from the code alone (or the other way round).
        assert (second, second_paper_used) == ("generated formula", True)
    finally:
        monkey.undo()
        reporting._FORMULA_CACHE.clear()


def test_editing_a_kernel_invalidates_its_cached_formula(tmp_path) -> None:
    # This is what makes shipping the formulas safe. The cached text is keyed by a digest
    # of the kernel's own source, so flipping one sign in kernels.py must miss the cache
    # rather than serve a formula that no longer describes the code.
    from lamet_agent.stages.matching import reporting

    monkey = pytest.MonkeyPatch()
    monkey.setattr(reporting, "_BUNDLED_FORMULA_DIR", tmp_path / "bundled")
    monkey.setenv("LAMET_FORMULA_CACHE_DIR", str(tmp_path / "user"))
    monkey.setattr(reporting, "_fetch_paper_text", lambda *_a, **_k: "paper latex")
    calls: list[str] = []

    def fake_request(*_args, **kwargs):
        calls.append("call")
        return f"formula v{len(calls)}"

    monkey.setattr(reporting, "request_llm_text", fake_request)
    reporting._FORMULA_CACHE.clear()
    llm = reporting.FormulaLlm(backend="api", provider="deepseek", api_key="k", model_name="m")
    try:
        real_source = reporting._kernel_source("CG_gt_quark_PDF_hybrid_NLO")
        first, _ = reporting._llm_kernel_formula("CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm)
        assert first == "formula v1" and len(calls) == 1

        # Same kernel_id, one character of its implementation changed.
        monkey.setattr(reporting, "_kernel_source", lambda _kid: real_source + "  # edited")
        reporting._FORMULA_CACHE.clear()
        second, _ = reporting._llm_kernel_formula("CG_gt_quark_PDF_hybrid_NLO", language="en", llm=llm)
        assert second == "formula v2", "an edited kernel must not reuse the old formula"
        assert len(calls) == 2
        # ...overwriting its own file rather than dropping a second one beside it: one
        # kernel is one file, so ordinary editing does not silt up the cache directory.
        files = list((tmp_path / "user").glob("CG_gt_quark_PDF_hybrid_NLO.en*.md"))
        assert [f.name for f in files] == ["CG_gt_quark_PDF_hybrid_NLO.en.md"]
        assert "formula v2" in files[0].read_text(encoding="utf-8")
    finally:
        monkey.undo()
        reporting._FORMULA_CACHE.clear()


def test_formula_llm_preserves_codex_model_name() -> None:
    from lamet_agent.stages.matching.reporting import FormulaLlm

    assert FormulaLlm(
        backend="cli",
        provider="codex",
        model_name="test-codex-model",
    ).resolved() == ("cli", "codex", None, "test-codex-model", None)


def test_kernel_source_carries_what_the_kernel_actually_calls() -> None:
    from lamet_agent.stages.matching.reporting import _kernel_source

    # The source handed to the formula LLM must follow the kernel's own call graph. A
    # hardcoded list of PDF coefficients left a DA kernel's V(x, y) out entirely, and the
    # model could only answer that it had no way to document the coefficient.
    da = _kernel_source("GI_gzg5_DA_hybrid_NLO")
    assert "def V_qq_p" in da and "def _da_matrix" in da

    # The PDF kernels pass their coefficient in as a lambda, so it is reachable only
    # through a nested code object -- easy to miss when walking the call graph.
    pdf = _kernel_source("CG_gt_quark_PDF_hybrid_NLO")
    assert "def C_hybrid" in pdf and "def C_ratio" in pdf

    # And neither should carry the other's physics as noise.
    assert "def C_ratio" not in da
    assert "def V_qq_p" not in pdf


def test_every_registered_kernel_declares_a_render_structure() -> None:
    # The report renders each kernel from its own `matching_structure` (attached in
    # kernels.py). If a kernel shipped without one, the report would silently fall back to
    # a generic factorization -- so require every registered kernel to declare it.
    for kernel_id, builder in KERNEL_REGISTRY.items():
        structure = getattr(builder, "matching_structure", None)
        assert isinstance(structure, dict), kernel_id
        assert structure.get("factorization"), kernel_id
        assert structure.get("notation"), kernel_id
        assert isinstance(structure.get("result_noun"), str), kernel_id


def test_report_formula_follows_the_kernel_structure_without_family_branches() -> None:
    # The formula section must render whatever the kernel declares -- an LRR kernel's
    # all-orders matrix exponential, a DA kernel's V(x, y) -- with no `if is_lrr/is_da` in
    # the report. Stub the LLM coefficient (the network call) via its cache so the test is
    # offline and exercises only the structure-driven scaffolding.
    from lamet_agent.stages.matching import reporting as R

    def formula_text(kernel_id: str) -> str:
        R._FORMULA_CACHE[(kernel_id, "en")] = ("STUB", False)
        return R._matching_formula_text({"kernel_id": kernel_id}, language="en", llm=None)

    # The renormalon-resummed kernel writes out its matrix-exponential structure...
    lrr = formula_text("GI_gt_quark_PDF_hybrid_LRR_NLO")
    assert r"M_{\mathrm{LRR}}" in lrr and "resums the leading Wilson-line renormalon" in lrr
    # ...and its formula prompt carries the instruction to document that resummation.
    lrr_structure = R._kernel_structure("GI_gt_quark_PDF_hybrid_LRR_NLO")
    assert "matrix exponential" in (lrr_structure.get("extra_note") or "")

    # The plain fixed-order kernel has no such structure and stays a PDF...
    fixed = formula_text("GI_gt_quark_PDF_hybrid_NLO")
    assert r"M_{\mathrm{LRR}}" not in fixed and "light-cone PDF" in fixed

    # ...and a DA kernel renders the DA factorization, not a PDF one, from the same code.
    da = formula_text("GI_gzg5_DA_hybrid_NLO")
    assert "light-cone DA" in da and r"\phi(x,\mu)" in da and r"\frac{dy}{|y|}" not in da


def test_report_text_follows_the_kernel_rather_than_assuming_a_pdf() -> None:
    from lamet_agent.stages.matching.reporting import _kernel_description, _scheme_explanation

    # The same Dirac structure serves a DA and a PDF, so the description must come from
    # the id's distribution field, not the operator alone.
    assert "distribution amplitude" in _kernel_description("GI_gzg5_DA_hybrid_NLO", language="en")
    assert "quark PDF" not in _kernel_description("GI_gzg5_DA_hybrid_NLO", language="en")
    assert "helicity" not in _kernel_description("GI_gzg5_DA_hybrid_NLO", language="en")
    assert "helicity" in _kernel_description("GI_gzg5_quark_PDF_ratio_NLO", language="en")

    # The scheme note cites the equations the selected kernel is tagged with, so a DA
    # kernel must not carry the Coulomb-gauge PDF paper's equation numbers.
    da_scheme = " ".join(_scheme_explanation({"kernel_id": "GI_gzg5_DA_hybrid_NLO"}, language="en"))
    assert "2405.20120" in da_scheme and "2602.11283" not in da_scheme
    pdf_scheme = " ".join(_scheme_explanation({"kernel_id": "CG_gt_quark_PDF_hybrid_NLO"}, language="en"))
    assert "2602.11283" in pdf_scheme


def test_lc_window_outside_quasi_is_rejected() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}
    load_quasi_pdf(store)

    with pytest.raises(ValueError, match="extends beyond"):
        build_matching_kernel(
            store,
            kernel_id="CG_gt_quark_PDF_ratio_NLO",
            momentum_gev=1.5,
            lc_x_ls={"start": -3.0, "stop": 3.0},
        )

    build_matching_kernel(
        store,
        kernel_id="CG_gt_quark_PDF_ratio_NLO",
        momentum_gev=1.5,
        lc_x_ls={"start": -1.0, "stop": 1.0},
    )
    expected = native[(native >= -1.0) & (native <= 1.0)]
    assert store["kernel_matrix"].shape == (expected.size, native.size)


def _matching_grid_payload(
    *,
    lc_x_ls: dict | None | object = ...,
    quasi_y_ls: dict | None | object = ...,
) -> dict:
    if lc_x_ls is ...:
        lc_x_ls = {"start": 0.0, "stop": 1.0}
    if quasi_y_ls is ...:
        quasi_y_ls = {"start": -2.0, "stop": 2.0, "num": 100}
    matching_defaults: dict = {"scheme": "ratio", "mu": 2.0}
    if lc_x_ls is not None:
        matching_defaults["lc_x_ls"] = lc_x_ls
    fourier_defaults: dict = {
        "order": ["LA"],
        "gfix": "GI",
        "sector": "valence",
        "Lambda0_gev": 0.0,
        "posterior_prior_error_scale": 3.0,
        "scheme_scan": {
            "zmin_fm": [0.1],
            "zmax_fm": [0.8],
            "zmax_ext_fm": 1.2,
            "smooth": "linear",
            "model_average": False,
        },
    }
    if quasi_y_ls is not None:
        fourier_defaults["quasi_y_ls"] = quasi_y_ls
    return {
        "metadata": {
            "run_id": "demo",
            "root_directory": ".",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform", "perturbative_matching"],
        },
        "inputs": {
            "correlators": [],
            "artifacts": [{"id": "rn", "stage": "renormalization", "path": "rn.nc"}],
            "kernels": [
                {
                    "stage": "perturbative_matching",
                    "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
                    "kernel_path": "kernels.py",
                    "kernel_parameters": {},
                }
            ],
        },
        "stages": {
            "fourier_transform": {
                "defaults": fourier_defaults,
                "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
            },
            "perturbative_matching": {
                "defaults": matching_defaults,
                "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}}],
            },
        },
    }


def test_matching_rejects_lc_window_outside_fourier_grid() -> None:
    manifest = AnalysisManifest.model_validate(
        _matching_grid_payload(lc_x_ls={"start": -3.0, "stop": 3.0})
    )
    job = manifest.stages["perturbative_matching"].jobs[0]
    issues = validate_stage_diagnostics("perturbative_matching", manifest, job)
    assert any(item.code == "matching.lc_x_ls.window" for item in issues)
    assert any("extends beyond" in item.message for item in issues)


def test_matching_accepts_lc_window_inside_fourier_grid() -> None:
    manifest = AnalysisManifest.model_validate(_matching_grid_payload())
    job = manifest.stages["perturbative_matching"].jobs[0]
    issues = validate_stage_diagnostics("perturbative_matching", manifest, job)
    assert not any(item.code == "matching.lc_x_ls.window" for item in issues)


@pytest.mark.parametrize(
    ("coords", "message"),
    [
        (np.linspace(-2.0, 2.0, 101), "must not contain 0"),
        (np.array([-1.0, -0.5, -0.1, 0.3, 1.2]), "uniformly spaced"),
    ],
)
def test_load_quasi_pdf_rejects_artifact_grids_kernels_cannot_integrate(coords, message: str) -> None:
    store = {"quasi": _quasi_on(coords)}

    with pytest.raises(ValueError, match=message):
        load_quasi_pdf(store)


def test_matching_consumes_in_memory_fourier_output_and_writes_primary_netcdf(tmp_path: Path) -> None:
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[np.array([1 + 0.1j, 2 + 0.2j]), np.array([1.2 + 0.1j, 2.2 + 0.2j])],
        dims=("x",),
        coords={"x": [-0.5, 0.5]},
        attrs={"bz_direction": "X"},
        name="fourier_transform",
    )
    store = {"quasi": data}
    loaded = load_quasi_pdf(store)
    store["kernel_matrix"] = np.eye(2)
    store["lc_x_ls"] = [-0.5, 0.5]

    result = apply_matching(store, artifacts_dir=tmp_path, job_id="mt_p5")

    assert loaded["n_sample"] == 2
    assert store["output"] is store["lightcone_ed"]
    assert store["output"].attrs["bz_direction"] == "X"
    assert Path(result["artifact"]).is_file()
    saved = EnsembleData.from_netcdf(result["artifact"])
    assert saved.dims == ["x"]
    assert np.allclose(saved.values, [[1, 2], [1.2, 2.2]])


def test_plot_matched_pdf_writes_pdf_and_svg(tmp_path: Path) -> None:
    quasi = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0, 2.0]), np.array([1.1, 2.1])],
        dims=("x",),
        coords={"x": [0.0, 1.0]},
        attrs={"sector": "valence"},
        name="quasi_pdf",
    )
    lightcone = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([0.9, 1.8]), np.array([1.0, 1.9])],
        dims=("x",),
        coords={"x": [0.0, 1.0]},
        name="lightcone_pdf",
    )
    store = {"quasi_y_ls": np.array([0.0, 1.0]), "quasi_ed": quasi, "lightcone_ed": lightcone}

    result = plot_matched_pdf(store, artifacts_dir=tmp_path, job_id="matched_pdf")

    assert Path(result["path"]).is_file()
    assert Path(result["plot_image"]).is_file()
    assert Path(result["path"]).suffix == ".pdf"
    assert Path(result["plot_image"]).suffix == ".svg"
    assert result["xlim"] == [-0.01, 1.01]
    assert result["ylim"] == pytest.approx([0.6792893218813452, 3.1207106781186544])
    assert store["matching_plot"] == result


def test_load_quasi_pdf_uses_fourier_artifact_channel() -> None:
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[np.array([1 + 2j, 3 + 4j]), np.array([1.1 + 2.1j, 3.1 + 4.1j])],
        dims=("x",),
        coords={"x": [-0.5, 0.5]},
        attrs={"component": "im"},
        name="fourier_transform",
    )
    store = {"quasi": data}

    loaded = load_quasi_pdf(store)

    assert loaded["component"] == "im"
    assert np.allclose(store["quasi_ed"].values, [[2.0, 4.0], [2.1, 4.1]])


# --- leading-renormalon resummation (LRR), arXiv:2305.05212 -------------------


def test_renormalon_pv_sum_matches_the_notebook_stored_outputs() -> None:
    """dPVasym reproduces LRR.nb's cached values once given the alpha_s it used.

    The notebook prints dPVasym[0.1/GeVfm, mu, 3, alpha_s] = 4.9371 at mu = 100 GeV and
    2.8577 at mu = 50 GeV, evaluated with its threshold-crossing four-loop coupling
    (alpha_s(100) ~ 0.1163, alpha_s(50) ~ 0.1297). Feeding those couplings back pins the
    transcription of the whole renormalon closed form.
    """
    from lamet_agent.kernels import GEV_FM, dPVasym

    z = 0.1 / GEV_FM
    assert dPVasym(z, 100.0, 3, 0.11628) == pytest.approx(4.937136, rel=2e-4)
    assert dPVasym(z, 50.0, 3, 0.12970) == pytest.approx(2.857725, rel=2e-4)


def test_lrr_kernel_reduces_to_fixed_order_without_the_renormalon() -> None:
    """With the renormalon numbers zeroed, M_LRR collapses to the fixed-order GI hybrid.

    (M_fix + r0 MCz) . exp(-MCz rsumPV) -> M_fix as r0, rsumPV -> 0, so the LRR kernel is a
    genuine correction *on top of* the fixed order, not a separate object.
    """
    import lamet_agent.kernels as K

    x = np.linspace(-2.0, 2.0, 60)
    x = x[np.abs(x) > 1e-6]
    kw = dict(momentum_gev=1.9, mu=2.0, zspz=4 * 0.06 / K.GEV_FM)
    m_fix = K.GI_gt_quark_PDF_hybrid_NLO(x, **kw)

    saved = (K.rnasym, K.dPVasym)
    try:
        K.rnasym = lambda *a, **k: 0.0
        K.dPVasym = lambda *a, **k: 0.0
        m_lrr = K.GI_gt_quark_PDF_hybrid_LRR_NLO(x, **kw)
    finally:
        K.rnasym, K.dPVasym = saved

    assert np.allclose(m_lrr, m_fix, atol=1e-12)


def test_lrr_kernel_registered_square_and_finite() -> None:
    """The four GI+LRR ids are wired in, need zspz, and produce a finite square matrix."""
    x = np.linspace(-2.0, 2.0, 60)
    x = x[np.abs(x) > 1e-6]
    store = {
        "quasi_y_ls": x,
        "quasi_ed": EnsembleData(
            ensemble=None, resample="bootstrap",
            values=[np.exp(-(x**2)) + 0.001 * i for i in range(4)],
            dims=("x",), coords={"x": x.tolist()}, name="quasi_pdf",
        ),
    }
    info = build_matching_kernel(
        store, kernel_id="GI_gt_quark_PDF_hybrid_LRR_NLO", momentum_gev=1.9, zs_fm=0.24,
    )
    assert info["shape"] == [x.size, x.size]
    assert np.isfinite(store["kernel_matrix"]).all()

    apply_matching(store)
    assert np.isfinite(store["lightcone_ed"].mean).all()

    # The renormalon is a Wilson-line property, so it extends to every GI hybrid operator
    # (transversity, meson DA) by swapping only the fixed-order builder.
    for kid in (
        "GI_gt_quark_PDF_hybrid_LRR_NLO",
        "GI_gtg5_quark_PDF_hybrid_LRR_NLO",
        "GI_gz_quark_PDF_hybrid_LRR_NLO",
        "GI_gzg5_quark_PDF_hybrid_LRR_NLO",
        "GI_gtgpg5_quark_PDF_hybrid_LRR_NLO",
        "GI_gtg5_DA_hybrid_LRR_NLO",
        "GI_gzg5_DA_hybrid_LRR_NLO",
    ):
        assert KERNEL_REGISTRY[kid].__name__ == kid


def test_rgr_alpha_s_running_reproduces_the_reference_points() -> None:
    # The whole RGR construction rides on alpha_s(mu), so pin it: the notebook starts from
    # alpha_s(m_Z) = 0.1179 and runs down with two-loop beta, switching nf at m_b and m_c.
    from lamet_agent import kernels as K

    assert K._alpha_s(K._M_Z) == pytest.approx(0.1179, abs=1e-6)
    # Monotonically growing towards the infrared, and in the right ballpark at the scales
    # this analysis actually uses.
    scales = [91.1876, 10.0, 4.18, 2.0, 1.27, 0.8]
    values = [K._alpha_s(m) for m in scales]
    assert all(a < b for a, b in zip(values, values[1:])), values
    assert K._alpha_s(2.0) == pytest.approx(0.30, abs=0.03)


def test_rgr_dglap_matrix_conserves_number_and_is_polarization_dependent() -> None:
    # The plus prescription is implemented as a diagonal subtraction of the column sums,
    # which is exactly what makes each column integrate to zero -- lose that and the
    # evolution stops conserving the quark number.
    import numpy as np

    from lamet_agent import kernels as K

    x = np.linspace(0.02, 1.0, 40)
    lo, nlo = K._dglap_evolution_matrices(x, K._p_nlo_full_unpolarized)
    assert np.abs(lo.sum(axis=0)).max() < 1e-12
    assert np.abs(nlo.sum(axis=0)).max() < 1e-10

    # RGR is not operator universal the way LRR is: the three polarizations carry three
    # different two-loop splitting functions, which is why gamma^t and gamma^t gamma5 need
    # separate RGR kernels even though they share one fixed-order coefficient.
    nu = np.array([0.2, 0.5, 0.8])
    unpol = K._p_nlo_full_unpolarized(nu)
    assert not np.allclose(unpol, K._p_nlo_full_helicity(nu))
    assert not np.allclose(unpol, K._p_nlo_transversity(nu))
    # Support is 0 <= nu < 1; outside it every splitting function is identically zero.
    outside = np.array([-0.3, 1.5])
    for fn in (K._p_qq_lo, K._p_nlo_full_unpolarized, K._p_nlo_full_helicity, K._p_nlo_transversity):
        assert np.all(fn(outside) == 0.0)


def test_rgr_kernel_zeroes_rows_below_the_perturbative_cutoff() -> None:
    # The paper fixes x_min by alpha_s(2 x P^z) ~ 1. In the matrix that shows up as whole
    # rows of zeros: a row whose own scale mu0(x) = 2 kappa x P^z is below the cutoff has no
    # number to report, and reporting one anyway would be the actual error.
    import numpy as np

    from lamet_agent import kernels as K

    momentum, mu_min, kappa = 3.04, 0.6, 1.0
    x = np.linspace(-1.0, 1.0, 41) + 1.0 / 40  # offset keeps 0 off the quasi grid
    matrix = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(
        x, momentum_gev=momentum, mu=2.0, quasi_y_ls=x, zspz=3.7
    )
    assert np.isfinite(matrix).all()

    alive = np.abs(matrix).sum(axis=1) > 0
    expected = (2.0 * kappa * x * momentum) >= mu_min
    assert np.array_equal(alive, expected)
    # Negative x can never clear the cutoff, so the whole antiquark half stays zero.
    assert not alive[x < 0].any()


def test_rgr_reduces_to_the_fixed_order_row_when_no_evolution_is_needed() -> None:
    # Sanity anchor for the evolution: when a row's own scale already equals the target mu,
    # the evolution operator is the identity and RGR must hand back that row of the plain
    # fixed-order matrix. Any scale-setting or ordering slip breaks this.
    import numpy as np

    from lamet_agent import kernels as K

    momentum, zspz = 3.04, 3.7
    x = np.linspace(-1.0, 1.0, 41) + 1.0 / 40
    index = int(np.argmax(x > 0.3))
    mu = 2.0 * float(x[index]) * momentum  # this row's mu0 is the target scale exactly

    rgr = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, momentum_gev=momentum, mu=mu, quasi_y_ls=x, zspz=zspz)
    fixed = K.CG_gt_quark_PDF_hybrid_NLO(x, momentum_gev=momentum, mu=mu, quasi_y_ls=x, zspz=zspz)
    np.testing.assert_allclose(rgr[index], fixed[index], atol=1e-10)


def test_rgr_kernels_are_registered_and_declare_their_resummation() -> None:
    # A new kernel must be reachable from a manifest and must tell the report that it is not
    # a plain fixed-order coefficient -- otherwise the formula section would document it as
    # one, which is precisely the kind of silent wrongness the structure declaration exists
    # to prevent.
    from lamet_agent.stages.matching.functions import KERNEL_REGISTRY, is_hybrid_kernel

    ids = sorted(k for k in KERNEL_REGISTRY if "RGR" in k.split("_"))
    # Five operators x two channels. The real part rides on the hybrid fixed order and the
    # imaginary part on MSbar, because the paper renormalizes the two parts differently.
    assert len(ids) == 10, ids
    assert sum(1 for k in ids if k.endswith("_re_NLO")) == 5, ids
    for kernel_id in ids:
        builder = KERNEL_REGISTRY[kernel_id]
        channel = "re" if kernel_id.endswith("_re_NLO") else "im"
        scheme = "hybrid" if channel == "re" else "msbar"
        assert scheme in kernel_id.split("_"), kernel_id
        assert is_hybrid_kernel(kernel_id) == (channel == "re"), kernel_id
        # The tag names where the RESUMMATION is derived (arXiv:2209.01236, appendix "A
        # Method Solving RG Equation"), not where the fixed order comes from -- the same
        # convention the LRR kernels follow. arXiv:2602.11283 only restates the procedure
        # in prose and has no numbered equation to cite.
        assert builder.arxiv_id == "2209.01236", kernel_id
        assert "RG Equation" in builder.equations, kernel_id
        structure = builder.matching_structure
        assert structure["extra_structure"], kernel_id
        assert "small-x" in structure["extra_note"] or "SMALL-x" in structure["extra_note"]


def test_rgr_helicity_does_not_reuse_the_unpolarized_splitting_function() -> None:
    # arXiv:2602.11283: the real part gives the VALENCE channel for the unpolarized PDF but
    # the FULL channel for helicity, "since the helicity quasi-distribution is even under
    # charge conjugation". Valence and full are the two C-parity non-singlet combinations,
    # and those differ from two loops on -- so the `re` kernels of the two polarizations
    # must carry different two-loop kernels even though they share one fixed-order
    # coefficient. Reusing one for the other is a silent physics error, not a refactor.
    import numpy as np

    from lamet_agent import kernels as K

    nu = np.linspace(0.05, 0.95, 19)
    unpolarized = K._p_nlo_full_unpolarized(nu)
    helicity = K._p_nlo_full_helicity(nu)
    assert not np.allclose(unpolarized, helicity)

    # The whole difference is the C-parity structure: 16 CF (CF - CA/2) [...] plus the
    # accompanying nf piece. At one loop the two combinations are identical, which is why
    # only the NLO kernel is polarization dependent here.
    assert np.allclose(K._p_qq_lo(nu), K._p_qq_lo(nu))
    difference = helicity - unpolarized
    assert np.all(np.abs(difference) > 0)

    # And the two RGR kernels really do propagate that difference into the matrix.
    x = np.linspace(-1.0, 1.0, 41) + 1.0 / 40
    kwargs = dict(momentum_gev=3.04, mu=2.0, quasi_y_ls=x, zspz=3.7)
    unpol_matrix = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **kwargs)
    heli_matrix = K.CG_gtg5_quark_PDF_hybrid_RGR_re_NLO(x, **kwargs)
    assert not np.allclose(unpol_matrix, heli_matrix)
    # The fixed order IS shared, so any difference has to come from the evolution alone.
    assert np.allclose(
        K.CG_gt_quark_PDF_hybrid_NLO(x, **kwargs), K.CG_gtg5_quark_PDF_hybrid_NLO(x, **kwargs)
    )


def test_rgr_channels_pair_the_scheme_and_the_c_parity_the_notebook_labelled() -> None:
    # CG_RGR_kernels.nb labels its own output: it exports
    # `matching_scale<c>_{valence|full}_<tag>_rgr_...csv`, and those labels -- not guesswork
    # -- fix which splitting function belongs to which channel. Combined with the paper
    # ("the real and imaginary parts ... correspond to the valence and full quark channels
    # respectively", and the helicity quasi-distribution being C-even) this pins the table
    # below. The pairing is invisible in the output if wrong, so assert it directly.
    import numpy as np

    from lamet_agent import kernels as K

    x = np.linspace(-1.0, 1.0, 41) + 1.0 / 40
    hybrid = dict(momentum_gev=3.04, mu=2.0, quasi_y_ls=x, zspz=3.7)
    msbar = dict(momentum_gev=3.04, mu=2.0, quasi_y_ls=x)

    # Unpolarized and helicity take OPPOSITE C-parity in the same channel, so their real
    # parts must differ even though their hybrid fixed order is one and the same function.
    assert np.allclose(
        K.CG_gt_quark_PDF_hybrid_NLO(x, **hybrid), K.CG_gtg5_quark_PDF_hybrid_NLO(x, **hybrid)
    )
    assert not np.allclose(
        K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **hybrid),
        K.CG_gtg5_quark_PDF_hybrid_RGR_re_NLO(x, **hybrid),
    )

    # gamma^z shares gamma^t's hybrid coefficient but has its own MSbar one (Eq. 2.15 adds
    # 2(1-ksi)_+), so the two agree in the real channel and must not in the imaginary one.
    assert np.allclose(
        K.CG_gz_quark_PDF_hybrid_RGR_re_NLO(x, **hybrid),
        K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **hybrid),
    )
    assert not np.allclose(
        K.CG_gz_quark_PDF_msbar_RGR_im_NLO(x, **msbar),
        K.CG_gt_quark_PDF_msbar_RGR_im_NLO(x, **msbar),
    )

    # The valence kernel is shared: the notebook produces the valence exports of BOTH
    # polarizations with one splitting function.
    nu = np.linspace(0.05, 0.95, 19)
    assert np.allclose(K._p_nlo_valence(nu), K._p_nlo_full_unpolarized(nu) + K._c_parity_term(nu))

    # Transversity has one evolution kernel and one fixed order across schemes, so its two
    # channels coincide numerically; they stay separate ids so a manifest names what it ran.
    assert np.allclose(
        K.CG_gtgpg5_quark_PDF_hybrid_RGR_re_NLO(x, **hybrid),
        K.CG_gtgpg5_quark_PDF_msbar_RGR_im_NLO(x, **msbar),
    )


def test_rgr_kappa_and_mu_min_move_x_min_together() -> None:
    # kappa and mu_min are not independent knobs: the cutoff is applied to mu0(x) = 2 kappa
    # x P^z, so the surviving window is x >= mu_min / (2 kappa P^z). A kappa scan for the
    # systematic budget therefore also moves x_min, and the three variants of a scan do not
    # share an x range -- worth pinning, because comparing them as if they did is a mistake
    # the numbers alone will not reveal.
    import numpy as np

    from lamet_agent import kernels as K

    momentum = 3.04
    x = np.linspace(-1.0, 1.0, 41) + 1.0 / 40
    base = dict(momentum_gev=momentum, mu=2.0, quasi_y_ls=x, zspz=3.7)

    def live_window(**overrides):
        matrix = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **base, **overrides)
        alive = np.abs(matrix).sum(axis=1) > 0
        return alive.sum(), x[alive].min()

    for kappa, mu_min in [(0.71, 0.6), (1.0, 0.6), (1.4, 0.6), (1.0, 0.4), (1.0, 1.0)]:
        count, x_min = live_window(kappa=kappa, mu_min=mu_min)
        predicted = mu_min / (2.0 * kappa * momentum)
        # The first surviving grid point is the first one at or above the predicted x_min.
        assert x_min >= predicted
        assert x[x < x_min].max() < predicted
        assert count > 0

    # Raising kappa lowers x_min; raising mu_min raises it.
    assert live_window(kappa=1.4)[1] < live_window(kappa=0.71)[1]
    assert live_window(mu_min=1.0)[1] > live_window(mu_min=0.4)[1]

    # kappa is not only a cutoff: it also changes where the fixed order is evaluated, so the
    # rows that survive both settings are still different numbers.
    central = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **base, kappa=1.0)
    varied = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, **base, kappa=1.4)
    shared = (np.abs(central).sum(axis=1) > 0) & (np.abs(varied).sum(axis=1) > 0)
    assert shared.any()
    assert not np.allclose(central[shared], varied[shared])


def test_rgr_parameters_go_inert_on_a_kernel_that_has_no_per_row_scale() -> None:
    # kappa and mu_min mean something only to RGR: a fixed-order kernel has no per-row scale
    # to vary. Such a job runs unchanged rather than failing, so a stage default can carry
    # the parameters while only some jobs select an RGR kernel. The kernel matrix must be
    # bit-identical to the one built without them, and the trace records what was dropped.
    import numpy as np

    from lamet_agent.stages.matching.functions import build_matching_kernel

    y = np.linspace(-1.0, 1.0, 41) + 1.0 / 40
    args = dict(
        kernel_id="CG_gt_quark_PDF_hybrid_NLO",
        momentum_gev=3.04,
        mu=2.0,
        zs_fm=0.18,
        lc_x_ls={"start": 0.0, "stop": 1.0},
    )
    plain: dict = {"quasi_y_ls": y}
    build_matching_kernel(plain, **args)

    with_rgr: dict = {"quasi_y_ls": y}
    build_matching_kernel(with_rgr, **args, rgr_kappa=1.4, rgr_mu_min_gev=0.9)

    np.testing.assert_array_equal(plain["kernel_matrix"], with_rgr["kernel_matrix"])
    assert "ignored_params" not in plain["matching_kernel_info"]
    assert with_rgr["matching_kernel_info"]["ignored_params"] == ["kappa", "mu_min"]

    # An RGR kernel does read them, so the same parameters are not inert everywhere.
    used: dict = {"quasi_y_ls": y}
    build_matching_kernel(
        used,
        kernel_id="CG_gt_quark_PDF_hybrid_RGR_re_NLO",
        momentum_gev=3.04, mu=2.0, zs_fm=0.18,
        lc_x_ls={"start": 0.0, "stop": 1.0},
        rgr_kappa=1.4, rgr_mu_min_gev=0.9,
    )
    assert "ignored_params" not in used["matching_kernel_info"]



def test_rgr_evolution_operator_keeps_the_notebook_factor_ordering() -> None:
    # The per-step factors do not commute (alpha_s differs between steps), and the source
    # notebook composes them with `Dot @@ Table[...]`, i.e. the EARLIEST step leftmost.
    # Reversing the product changes the operator by only ~1e-6, which reads as rounding --
    # so pin the ordering rather than trusting a smoke test to notice.
    import numpy as np
    from scipy.linalg import expm

    from lamet_agent import kernels as K

    x = np.linspace(0.025, 1.0, 20)
    evo_lo, evo_nlo = K._dglap_evolution_matrices(x, K._p_nlo_valence)
    mu_i, mu_f, steps = 4.0, 2.0, 20

    t0, t1 = np.log(mu_i**2), np.log(mu_f**2)
    dt = (t1 - t0) / steps
    expected = np.eye(x.size)
    for index in range(steps):
        a = K._alpha_s(float(np.exp((t0 + dt * (index + 0.5)) / 2.0))) / (4.0 * np.pi)
        expected = expected @ expm((a * evo_lo + a**2 * evo_nlo) * dt)

    operator = K._evolution_operator(mu_i, mu_f, evo_lo, evo_nlo, steps)
    np.testing.assert_allclose(operator, expected, rtol=1e-12, atol=1e-14)

    # The reversed product is genuinely a different matrix, so the assertion above has teeth.
    reversed_product = np.eye(x.size)
    for index in range(steps):
        a = K._alpha_s(float(np.exp((t0 + dt * (index + 0.5)) / 2.0))) / (4.0 * np.pi)
        reversed_product = expm((a * evo_lo + a**2 * evo_nlo) * dt) @ reversed_product
    assert not np.allclose(operator, reversed_product, rtol=1e-9, atol=0.0)


def test_lrr_accepts_a_light_cone_window_and_narrowing_it_only_selects_rows() -> None:
    # Two things at once, because they are the same requirement seen from both ends.
    #
    # 1) LRR works with lc_x_ls narrower than the quasi grid. C_z is both ADDED to the
    #    (nx, ny) fixed-order matrix and EXPONENTIATED, and only the second use needs a
    #    square matrix, so it is built twice -- (nx, ny) for the sum and (ny, ny) for the
    #    exponent. Which of the two is square is not a choice: LRR.nb writes the exponential
    #    on the RIGHT, so it contracts the left factor's column -- the quasi -- index.
    # 2) Narrowing the window SELECTS rows without changing them. Row x is
    #    f(x) = int dy K(x, y) f~(y): it cannot depend on which other output points were
    #    requested. That holds only because the plus-prescription subtraction is summed over
    #    the full quasi grid rather than over the rows that happen to be present.
    import numpy as np

    from lamet_agent import kernels as K

    y = np.linspace(-2.0, 2.0, 121) + 2.0 / 120
    x = y[(y > 0) & (y <= 1.0)]
    rows = np.flatnonzero(np.isin(y, x))
    kwargs = dict(momentum_gev=3.04, mu=2.0, zspz=2.77)

    with K._quiet_progress():
        narrow = K.GI_gt_quark_PDF_hybrid_LRR_NLO(x, quasi_y_ls=y, **kwargs)
        square = K.GI_gt_quark_PDF_hybrid_LRR_NLO(y, quasi_y_ls=y, **kwargs)
        fixed_narrow = K.GI_gt_quark_PDF_hybrid_NLO(x, quasi_y_ls=y, **kwargs)
        fixed_square = K.GI_gt_quark_PDF_hybrid_NLO(y, quasi_y_ls=y, **kwargs)

    assert narrow.shape == (x.size, y.size)
    assert np.isfinite(narrow).all()
    np.testing.assert_allclose(square[rows], narrow, rtol=0.0, atol=1e-12)
    # The fixed-order kernel carries the same guarantee -- it shares the plus prescription.
    np.testing.assert_allclose(fixed_square[rows], fixed_narrow, rtol=0.0, atol=1e-12)


def test_rgr_rows_are_coupled_through_dglap_unlike_the_matching_kernel() -> None:
    # LRR and the fixed order gained the guarantee that narrowing lc_x_ls only SELECTS rows.
    # RGR deliberately does not, and the difference is physical rather than a discretization
    # artifact: DGLAP is non-local in x -- evolving f(x) draws on f(x/z) for z < 1, i.e. on
    # LARGER x -- so the evolution operator genuinely couples the rows of whatever window it
    # is built on. That is why the paper stresses the evolution is "closed for x in
    # [x_min, 1]" and why the source notebook builds it on (0, 1] rather than on the full
    # Fourier range: the window IS part of the physics setup here, and lc_x_ls should be set
    # to the physical range, not used to crop a result after the fact.
    import numpy as np

    from lamet_agent import kernels as K

    y = np.linspace(-2.0, 2.0, 121) + 2.0 / 120
    x = y[(y > 0) & (y <= 1.0)]
    rows = np.flatnonzero(np.isin(y, x))
    kwargs = dict(momentum_gev=3.04, mu=2.0, zspz=2.77)

    with K._quiet_progress():
        narrow = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(x, quasi_y_ls=y, **kwargs)
        square = K.CG_gt_quark_PDF_hybrid_RGR_re_NLO(y, quasi_y_ls=y, **kwargs)

    live = np.abs(narrow).sum(axis=1) > 0
    assert live.any()
    assert not np.allclose(square[rows][live], narrow[live])

    # The fixed order the two share does obey row selection, so the difference above comes
    # from the evolution operator alone -- not from the matching kernel underneath it.
    with K._quiet_progress():
        fixed_narrow = K.CG_gt_quark_PDF_hybrid_NLO(x, quasi_y_ls=y, **kwargs)
        fixed_square = K.CG_gt_quark_PDF_hybrid_NLO(y, quasi_y_ls=y, **kwargs)
    np.testing.assert_allclose(fixed_square[rows], fixed_narrow, rtol=0.0, atol=1e-12)
