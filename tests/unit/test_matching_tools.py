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
    load_quasi_pdf(store, component="re")

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

    load_quasi_pdf(store, component="re")

    assert np.array_equal(store["quasi_ed"].values, _quasi_on(native).values)
    assert np.array_equal(store["quasi_y_ls"], native)


def test_lc_x_ls_window_slices_the_quasi_nodes() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}
    load_quasi_pdf(store, component="re")
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


def test_formula_cache_does_not_serve_one_kernel_another_kernels_formula() -> None:
    from lamet_agent.stages.matching import reporting

    # CG_gt_... and GI_gt_... share an operator and a scheme but are different kernels
    # from different papers, so a cache keyed on the parsed fields would hand the second
    # one the first one's formula -- silently, since nothing downstream re-checks it.
    calls: list[str] = []

    def fake_request(*_args, **kwargs):
        prompt = "".join(m["content"] for m in kwargs["messages"])
        kernel = "GI" if "def C_ratio_gi" in prompt else "CG"
        calls.append(kernel)
        return f"formula for {kernel}"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(reporting, "request_llm_text", fake_request)
    monkey.setattr(reporting, "_fetch_paper_text", lambda *_a, **_k: "paper latex")
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


def test_formula_llm_preserves_codex_model_name() -> None:
    from lamet_agent.stages.matching.reporting import FormulaLlm

    assert FormulaLlm(
        backend="codex",
        model_name="test-codex-model",
    ).resolved() == ("codex", None, None, "test-codex-model", None)


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
    load_quasi_pdf(store, component="re")

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
    matching_defaults: dict = {"scheme": "ratio", "component": "re", "mu": 2.0}
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


def test_endpoint_cut_drops_the_da_divergent_window_only_for_da_kernels() -> None:
    # Fine enough that points land inside 0.01 of x = 0 and x = 1; on a coarser grid the
    # cut has nothing to remove and would pass vacuously. The resolution has to come from
    # the quasi grid itself -- a denser light-cone grid is rejected, not a workaround.
    native = np.linspace(-1.0, 2.0, 300)

    def matched(kernel_id: str, cut: float | None) -> tuple[int, np.ndarray]:
        store = {"quasi": _quasi_on(native)}
        load_quasi_pdf(store, component="re")
        build_matching_kernel(store, kernel_id=kernel_id, momentum_gev=2.4, zs_fm=0.2)
        result = apply_matching(store, artifacts_dir=Path(tempfile.mkdtemp()), job_id="mt", endpoint_cut=cut)
        return result["endpoint_points_dropped"], np.asarray(store["lightcone_ed"].coords["x"])

    def in_window(x: np.ndarray) -> list[float]:
        return [float(v) for v in x if (0.0 < v < 0.01) or (0.99 < v < 1.0)]

    dropped, x = matched("GI_gzg5_DA_hybrid_NLO", 0.01)
    assert dropped == 2 and in_window(x) == []

    # Without the cut the divergent points ship, so the cut is what removes them.
    dropped, x = matched("GI_gzg5_DA_hybrid_NLO", None)
    assert dropped == 0 and len(in_window(x)) == 2

    # A PDF kernel has no endpoint divergence, so the cut must not touch its grid.
    dropped, x = matched("CG_gt_quark_PDF_hybrid_NLO", 0.01)
    assert dropped == 0 and len(in_window(x)) == 2


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
        load_quasi_pdf(store, component="re")


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
    loaded = load_quasi_pdf(store, component="re")
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


def test_plot_matched_pdf_honors_explicit_limits(tmp_path: Path) -> None:
    quasi = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0, 2.0]), np.array([1.1, 2.1])],
        dims=("x",),
        coords={"x": [-1.0, 1.0]},
        attrs={"sector": "singlet"},
        name="quasi_pdf",
    )
    lightcone = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([0.9, 1.8]), np.array([1.0, 1.9])],
        dims=("x",),
        coords={"x": [-1.0, 1.0]},
        name="lightcone_pdf",
    )
    store = {"quasi_y_ls": np.array([-1.0, 1.0]), "quasi_ed": quasi, "lightcone_ed": lightcone}

    result = plot_matched_pdf(
        store,
        artifacts_dir=tmp_path, job_id="matched_pdf",
        ylim=[-0.2, 2.5],
    )

    assert result["xlim"] == [-1.01, 1.01]
    assert result["ylim"] == [-0.2, 2.5]


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


def test_lrr_kernel_rejects_a_distinct_lightcone_grid() -> None:
    """The matrix exponential needs a square matrix, so lc and quasi grids must coincide."""
    import lamet_agent.kernels as K

    x = np.linspace(-2.0, 2.0, 60)
    x = x[np.abs(x) > 1e-6]
    with pytest.raises(ValueError, match="square grid|matching grids"):
        K.GI_gt_quark_PDF_hybrid_LRR_NLO(x, momentum_gev=1.9, zspz=1.0, quasi_y_ls=x[::2])
