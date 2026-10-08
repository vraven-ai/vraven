"""VRAVEN DarkSide: advanced internal mechanism audits for deep networks.

This extension is deliberately separate from ordinary feature attribution. It
provides sparse circuitry, cohort geometry, failure, drift, evasion, interaction,
mediation, capacity and uncertainty analyses. Observational and intervention
claims remain explicitly separated.
"""

from vraven._version import __version__

from .adversarial import analyse_adversarial_rerouting
from .anatomy import inspect_internal_anatomy, print_internal_anatomy
from .capacity import audit_unused_capacity
from .circuit_multiplicity import analyse_circuit_multiplicity
from .class_conditional import compare_class_conditionals, map_class_conditional_unit_field
from .contracts import AnalysisResult, FeatureRecommendation, InputFeatureGroup, UnitGroup
from .cost_evidence import analyse_cost_evidence_tradeoff
from .dataset import TraceDataset
from .decision_circuitry import discover_decision_circuitry
from .disagreement import analyse_method_disagreement
from .edge_cards import enrich_circuit_wiring_edges
from .explanation_uncertainty import analyse_explanation_uncertainty
from .failure_anatomy import analyse_failure_anatomy
from .mechanism_transfer import analyse_mechanism_transfer
from .mediation import analyse_activation_mediation
from .ood import analyse_high_confidence_ood
from .registry import FEATURES, recommend_features
from .representation_drift import analyse_representation_drift
from .serialisation import load_result
from .session import DarkSideSession
from .shortcut import analyse_prediction_depth, detect_shortcut_reliance, layerwise_shortcut_probe
from .stability_geometry import analyse_stability_geometry
from .synergy import analyse_causal_synergy
from .topology import analyse_topological_shift, h0_persistence_deaths
from .wiring import build_circuit_wiring

__all__ = [
    "AnalysisResult",
    "DarkSideSession",
    "FeatureRecommendation",
    "InputFeatureGroup",
    "TraceDataset",
    "UnitGroup",
    "__version__",
    "FEATURES",
    "analyse_activation_mediation",
    "analyse_adversarial_rerouting",
    "analyse_causal_synergy",
    "analyse_circuit_multiplicity",
    "analyse_cost_evidence_tradeoff",
    "analyse_explanation_uncertainty",
    "analyse_failure_anatomy",
    "analyse_high_confidence_ood",
    "analyse_mechanism_transfer",
    "analyse_method_disagreement",
    "analyse_prediction_depth",
    "analyse_representation_drift",
    "analyse_stability_geometry",
    "analyse_topological_shift",
    "audit_unused_capacity",
    "build_circuit_wiring",
    "enrich_circuit_wiring_edges",
    "compare_class_conditionals",
    "detect_shortcut_reliance",
    "discover_decision_circuitry",
    "h0_persistence_deaths",
    "inspect_internal_anatomy",
    "layerwise_shortcut_probe",
    "load_result",
    "map_class_conditional_unit_field",
    "print_internal_anatomy",
    "recommend_features",
]
