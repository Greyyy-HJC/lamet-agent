"""Derive Fourier conventions from upstream physics provenance."""

from __future__ import annotations

from collections.abc import Mapping


def derive_conventions(
    attrs: Mapping[str, object], *, target_observable: str, sector: str
) -> dict[str, object]:
    """Return the conventions selected implicitly by the reference pipeline."""
    target = str(target_observable).lower()
    parton = str(attrs.get("parton", "")).lower()
    gfix = str(attrs.get("gfix", "")).upper()
    polarization = str(attrs.get("polarization", "")).lower()
    if parton != "quark":
        raise ValueError("the migrated Fourier examples require parton='quark' provenance")
    if gfix not in {"GI", "CG"}:
        raise ValueError("Fourier input must carry GI or CG gfix provenance")
    if target == "da":
        if sector != "full":
            raise ValueError("DA Fourier transformation requires sector='full'")
        component, output_scale = "both", 1.0
    elif target == "pdf":
        if polarization not in {"unpolarized", "helicity", "transversity"}:
            raise ValueError("PDF Fourier input must carry supported polarization provenance")
        try:
            component, output_scale = {
                "valence": ("im" if polarization == "helicity" else "re", 2.0),
                "singlet": ("re" if polarization == "helicity" else "im", 2.0),
                "full": ("both", 1.0),
            }[sector]
        except KeyError as exc:
            raise ValueError("PDF Fourier sector must be valence, singlet, or full") from exc
    else:
        raise ValueError("the migrated Fourier conventions support PDF and DA targets")
    return {
        "parton": parton,
        "gfix": gfix,
        "symmetry": {"real": "even", "imag": "odd"},
        "transform": {"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        "tail_models": ["gi_nla" if gfix == "GI" else "cg_nla"],
        "component": component,
        "output_scale": output_scale,
    }
