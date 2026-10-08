"""Checkpoint and capacity audit for a training observer."""

from __future__ import annotations

from vraven.darkside import (  # noqa: F401 - used in the workflow below
    analyse_cost_evidence_tradeoff,
    analyse_representation_drift,
    analyse_stability_geometry,
    audit_unused_capacity,
)

# ``epoch_10`` and ``epoch_100`` must contain the same ordered evaluation samples.
# drift = analyse_representation_drift(epoch_10, epoch_100)
# stability = analyse_stability_geometry(epoch_10, epoch_100)
# capacity = audit_unused_capacity(epoch_100)
# efficiency = analyse_cost_evidence_tradeoff(epoch_100)
#
# drift.save_json("drift_epoch10_to_epoch100.json")
# drift.figure(mode="publication_light").write_html("drift_publication.html")
