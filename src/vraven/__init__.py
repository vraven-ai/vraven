"""VRAVEN: Visual Reasoning and Activation Visualisation for Explainable Networks."""

from ._version import __version__
from .adapters import load_pytorch_model
from .analysis import (
    DatasetItem,
    VRavenDatasetReport,
    VRavenEvolutionReport,
    VRavenModelComparison,
    compare_checkpoints,
    compare_models,
    explain_dataset,
)
from .attribution import LRPResult, explain_lrp
from .bundle import ModelBundle, load_model_bundle
from .causal import (
    CounterfactualResult,
    PerturbationResult,
    generate_counterfactual,
    run_perturbation_stability,
)
from .core.config import VRavenConfig
from .core.explainer import Explainer, explain
from .core.trace import NeuralTrace
from .darkside import DarkSideSession, TraceDataset, recommend_features
from .decompiler import (
    CaptureConfig,
    CausalEffectPropertySpec,
    DecompilationResult,
    DecompileConfig,
    FormalCausalMechanismCertificate,
    FormalCausalMechanismVerifier,
    FormalMechanismApplicabilityRecord,
    FormalRobustnessCertificate,
    InputDomainSpec,
    IntervalBoundVerifier,
    MechanismCertificate,
    MechanismIR,
    MechanismPropertySpec,
    VRavenDecompiler,
    reverse,
)
from .report.report import VRavenReport
from .visualisation import VRAVEN_COLORS, VRAVEN_IMPACT, VRavenTheme

__all__ = [
    "__version__",
    "CounterfactualResult",
    "DarkSideSession",
    "VRAVEN_COLORS",
    "VRAVEN_IMPACT",
    "VRavenTheme",
    "DatasetItem",
    "Explainer",
    "LRPResult",
    "ModelBundle",
    "NeuralTrace",
    "PerturbationResult",
    "TraceDataset",
    "VRavenConfig",
    "VRavenDatasetReport",
    "VRavenEvolutionReport",
    "VRavenModelComparison",
    "VRavenReport",
    "compare_checkpoints",
    "compare_models",
    "explain",
    "explain_dataset",
    "explain_lrp",
    "generate_counterfactual",
    "load_pytorch_model",
    "load_model_bundle",
    "recommend_features",
    "run_perturbation_stability",
    "CaptureConfig",
    "DecompileConfig",
    "DecompilationResult",
    "FormalCausalMechanismCertificate",
    "FormalCausalMechanismVerifier",
    "FormalMechanismApplicabilityRecord",
    "FormalRobustnessCertificate",
    "InputDomainSpec",
    "IntervalBoundVerifier",
    "CausalEffectPropertySpec",
    "MechanismPropertySpec",
    "MechanismCertificate",
    "MechanismIR",
    "VRavenDecompiler",
    "reverse",
]
