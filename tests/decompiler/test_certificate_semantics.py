from __future__ import annotations

import torch

from vraven.decompiler.concepts import Concept


def test_association_label_is_not_counted_as_validated_semantics():
    concept = Concept(
        id="C0",
        node_id="n0",
        node_name="n0",
        depth=0.2,
        index=0,
        feature_axis=-1,
        direction=torch.ones(2),
        mean_activation=0.0,
        std_activation=1.0,
        activation_sparsity=0.0,
        score_variance=1.0,
        top_samples=(0,),
        semantic_label="Associated with permissions",
        semantic_confidence=0.49,
        metadata={"semantic_status": "association_only"},
    )
    assert concept.semantically_associated
    assert not concept.semantically_grounded
