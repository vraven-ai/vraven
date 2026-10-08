from __future__ import annotations

import numpy as np

from vraven.metrics import agreement_matrices
from vraven.visualisation.consistency import consistency_matrix_figure


def test_agreement_matrices_are_symmetric() -> None:
    methods = {
        "a": np.array([1.0, 2.0, -1.0, 0.2]),
        "b": np.array([1.1, 1.8, -0.8, 0.1]),
        "c": np.array([-1.0, -2.0, 1.0, -0.2]),
    }
    result = agreement_matrices(methods, top_k=2)
    assert np.allclose(result.rank_correlation, result.rank_correlation.T)
    assert np.allclose(np.diag(result.sign_agreement), 1.0)
    figure = consistency_matrix_figure(methods, mode="publication_light")
    assert figure.layout.paper_bgcolor == "#FFFFFF"
