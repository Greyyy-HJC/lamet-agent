import json
from pathlib import Path

from lamet_agent.manifest import validate_manifest_file


def test_ordinary_manifests_keep_declared_jobs() -> None:
    for path in (
        Path("examples/pion_da_gi_manifest.json"),
        Path("examples/pion_pdf_cg_manifest.json"),
        Path("examples/pion_pdf_gi_manifest.json"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = validate_manifest_file(path)
        for stage in manifest.metadata.stages:
            declared = [job["id"] for job in payload["stages"][stage]["jobs"]]
            actual_declared = [job.id for job in manifest.stages[stage].jobs if job.id in declared]
            assert actual_declared == declared


def test_systematics_manifest_expands_complete_branches(tmp_path: Path) -> None:
    payload = {
        "metadata": {
            "run_id": "sys",
            "root_directory": ".",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": [
                "renormalization",
                "fourier_transform",
                "perturbative_matching",
                "extrapolation",
            ],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {
            "renormalization": {
                "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": "0.17(2)"},
                "jobs": [{"id": "rn4"}, {"id": "rn5"}],
            },
            "fourier_transform": {
                "defaults": {"zmin_shift": 1},
                "jobs": [
                    {"id": "ft4", "inputs": {"input": "rn4"}},
                    {"id": "ft5", "inputs": {"input": "rn5"}},
                ],
            },
            "perturbative_matching": {
                "defaults": {"mu": 2.0, "r": 2.0},
                "jobs": [
                    {"id": "mt4", "inputs": {"quasi": "ft4"}},
                    {"id": "mt5", "inputs": {"quasi": "ft5"}},
                ],
            },
            "extrapolation": {
                "defaults": {"allow_order_1overp_sym": [2, 4]},
                "jobs": [{"id": "ex_main", "inputs": {"lightcone": ["mt4", "mt5"]}}],
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = validate_manifest_file(path)

    assert len(manifest.stages["renormalization"].jobs) == 6
    assert len(manifest.stages["fourier_transform"].jobs) == 10
    assert len(manifest.stages["perturbative_matching"].jobs) == 14
    assert [job.id for job in manifest.stages["extrapolation"].jobs] == [
        "ex_main",
        "ex_zs_low",
        "ex_zs_high",
        "ex_lambda_low",
        "ex_lambda_high",
        "ex_mu_low",
        "ex_mu_high",
        "ex_other",
        "ex_budget",
    ]
    assert manifest.stages["extrapolation"].jobs[-1].inputs["main"] == "ex_main"
