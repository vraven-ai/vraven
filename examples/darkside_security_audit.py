"""Security-oriented feature selection and paired evasion audit."""

from __future__ import annotations

from vraven.darkside import (
    analyse_adversarial_rerouting,  # noqa: F401 - used in the workflow below
    analyse_failure_anatomy,  # noqa: F401 - used in the workflow below
    analyse_high_confidence_ood,  # noqa: F401 - used in the workflow below
    recommend_features,
)

# ``clean_traces`` and ``evasive_traces`` are TraceDataset objects constructed
# from the same ordered malware samples before and after a permitted evasion test.
# ``audit_dataset`` contains labels and model predictions for TP/FP/FN/TN cohorts.

print("Recommended security modules:")
for feature in recommend_features("security"):
    print(f"  {feature.name}: {feature.purpose} [{feature.cost} cost]")

# reroute = analyse_adversarial_rerouting(clean_traces, evasive_traces)
# failure = analyse_failure_anatomy(audit_dataset, failure="fp", reference="tp")
# anomaly = analyse_high_confidence_ood(
#     reference_fingerprints,
#     query_fingerprints,
#     query_confidence=query_confidence,
# )
