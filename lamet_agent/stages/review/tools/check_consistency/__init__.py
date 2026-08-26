"""Check cross-result identity, units, grids, and resampling provenance."""

from __future__ import annotations

from lamet_agent.agent import ToolContext


def run(context: ToolContext) -> dict[str, object]:
    """Create deterministic findings from the inspected result summaries."""
    summaries = context.state.get("result_summary")
    if not summaries:
        raise RuntimeError("inspect_results must run before check_consistency")
    findings: list[dict[str, object]] = []
    first = summaries[0]
    checks = set(context.params["checks"])
    for index, current in enumerate(summaries[1:], start=1):
        left = first.get("attrs", {})
        right = current.get("attrs", {})
        groups = {
            "identity": ("hadron", "current", "observable", "parton", "construction", "kernel_operator"),
            "units": ("units",),
            "kinematics": ("momentum_gev", "lattice_spacing_fm", "L_s", "m_pi_gev"),
            "schemes": ("renormalization_scheme", "strategy", "kernel_id"),
            "resampling": ("resample_id", "resample"),
            "extrapolation": ("physical_point", "physical_pion_mass_gev"),
        }
        for group, fields in groups.items():
            if group not in checks:
                continue
            for attr in fields:
                if (attr in left or attr in right) and left.get(attr) != right.get(attr):
                    findings.append(
                        {
                            "kind": "mismatch",
                            "group": group,
                            "index": index,
                            "field": attr,
                            "left": left.get(attr),
                            "right": right.get(attr),
                        }
                    )
        if "grids" in checks and (
            first.get("dims") != current.get("dims") or first.get("coords") != current.get("coords")
        ):
            findings.append(
                {
                    "kind": "mismatch",
                    "index": index,
                    "field": "dims",
                    "left": first.get("dims"),
                    "right": current.get("dims"),
                }
            )
    context.state["consistency"] = findings
    return {
        "summary": f"found {len(findings)} consistency findings",
        "metrics": {"finding_count": len(findings), "findings": findings},
        "state_keys": ["consistency"],
        "artifacts": [],
    }
