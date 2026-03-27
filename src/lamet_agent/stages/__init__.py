"""Import stage modules so they register themselves with the global registry."""

from lamet_agent.stages.correlator_analysis import CorrelatorAnalysisStage
from lamet_agent.stages.fourier_transform import FourierTransformStage
from lamet_agent.stages.perturbative_matching import PerturbativeMatchingStage
from lamet_agent.stages.physical_limit import PhysicalLimitStage
from lamet_agent.stages.renormalization import RenormalizationStage

__all__ = [
    "CorrelatorAnalysisStage",
    "RenormalizationStage",
    "FourierTransformStage",
    "PerturbativeMatchingStage",
    "PhysicalLimitStage",
]
