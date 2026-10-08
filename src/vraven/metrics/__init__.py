"""Mathematically defined VRAVEN metrics."""

from .agreement import (
    AgreementMatrices,
    agreement_matrices,
    sign_agreement,
    spearman_correlation,
    topk_overlap,
)
from .basic import activation_statistics, gradient_statistics, histogram_entropy, robust_scale
from .evidence_audit import (
    CausalConcordanceAudit,
    EvidenceConcentrationAudit,
    ScientificEvidenceAudit,
    evidence_concentration,
    observational_causal_concordance,
    scientific_evidence_audit,
)
from .influence import (
    LayerInfluence,
    activation_gradient_totals,
    cumulative_evidence,
    layer_influence,
    pathway_concentration,
)
from .stability import cosine_similarity, perturbation_consistency, sign_flip_rate
from .uncertainty import ConflictProfile, conflict_profile, internal_opposition_ratio

__all__ = [
    "AgreementMatrices",
    "CausalConcordanceAudit",
    "ConflictProfile",
    "EvidenceConcentrationAudit",
    "LayerInfluence",
    "ScientificEvidenceAudit",
    "activation_gradient_totals",
    "activation_statistics",
    "agreement_matrices",
    "conflict_profile",
    "cosine_similarity",
    "cumulative_evidence",
    "evidence_concentration",
    "gradient_statistics",
    "histogram_entropy",
    "internal_opposition_ratio",
    "layer_influence",
    "observational_causal_concordance",
    "pathway_concentration",
    "perturbation_consistency",
    "robust_scale",
    "sign_agreement",
    "sign_flip_rate",
    "scientific_evidence_audit",
    "spearman_correlation",
    "topk_overlap",
]
