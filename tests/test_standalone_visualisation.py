from __future__ import annotations

from vraven.core.trace import LayerTrace, NeuralTrace
from vraven.visualisation.energy import activation_energy_figure


def _trace() -> NeuralTrace:
    return NeuralTrace(
        layers=[
            LayerTrace(
                name="layer_1",
                layer_type="Linear",
                execution_index=0,
                output_shape=(1, 8),
                device="cpu",
                dtype="torch.float32",
                activation_energy=3.0,
            ),
            LayerTrace(
                name="layer_2",
                layer_type="ReLU",
                execution_index=1,
                output_shape=(1, 8),
                device="cpu",
                dtype="torch.float32",
                activation_energy=2.0,
            ),
        ]
    )


def test_publication_visual_is_white_and_standalone() -> None:
    figure = activation_energy_figure(_trace(), mode="publication_light")
    assert figure.layout.paper_bgcolor == "#FFFFFF"
    assert "VRAVEN activation energy" in figure.layout.title.text
    assert figure.layout.meta["vraven"]["standalone"] is True
    assert figure.layout.meta["vraven"]["render_mode"] == "publication_light"
    assert figure.layout.meta["scientific"]["evidence_tier"] == "observational"
    assert figure.layout.meta["scientific"]["scientific_question"]
    assert figure.layout.meta["scientific"]["visual_encodings"]
    assert figure.layout.meta["scientific"]["limitations"]
    assert len(figure.data) == 1


def test_dark_visual_uses_same_values() -> None:
    publication = activation_energy_figure(_trace(), mode="publication_light")
    interactive = activation_energy_figure(_trace(), mode="interactive_dark")
    assert list(publication.data[0].y) == list(interactive.data[0].y)
    assert interactive.layout.meta["vraven"]["render_mode"] == "interactive_dark"
    assert interactive.layout.paper_bgcolor != "#FFFFFF"
